// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dCrossWarpSync
{
    using Problem    = ck_tile::remove_cvref_t<Problem_>;
    using BlockShape = typename Problem::BlockShape;

    template <typename YDistributedTensor_>
    CK_TILE_DEVICE static constexpr int32_t GetReduceWarps()
    {
        using Dstr             = typename YDistributedTensor_::StaticTileDistribution;
        using DstrEncode       = typename Dstr::DstrEncode;
        using DstrEncodeDetail = typename DstrEncode::detail;

        constexpr int32_t NDimR = Dstr::get_num_of_dimension_r();

        constexpr int32_t idim_p_warp = 0;

        int32_t num_reduce_warps = 1;
        ck_tile::static_for<0, NDimR, 1>{}([&](auto idim_r) {
            if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_warp][idim_r])
            {
                constexpr int32_t r_length = DstrEncode::rs_lengths_[idim_r];
                num_reduce_warps *= r_length;
            }
        });

        return num_reduce_warps;
    }

    template <typename YDistributedTensor_>
    CK_TILE_DEVICE static constexpr int32_t GetColumnCount()
    {
        // using Dstr       = typename YDistributedTensor_::StaticTileDistribution;
        // using DstrEncode = typename Dstr::DstrEncode;

        // constexpr int32_t NDimP = Dstr::get_num_of_dimension_p();

        return 16;
    }

    // return in byte
    template <typename YDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSize()
    {
        using DataType   = typename YDistributedTensor_::DataType;

        constexpr int32_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();
        constexpr int32_t num_col = GetColumnCount<YDistributedTensor_>();

        // we need to store all data from every wave into smem
        // e.g. 2x2 reduce along N
        //     -------------> reduce N
        //    | w0 | w1 |   ___>      | w01 |
        //    | w2 | w3 |             | w23 |
        //
        //   -> store data from every wave into LDS
        //
        //
        //     -------------> reduce N
        //    | w0 | w1 | w2 | w3 |   ----->  | w0123 |
        //
        //   -> also store data from every wave into LDS
        constexpr int32_t num_warps = BlockShape::BlockSize / warpSize;
        return num_warps * num_col * thread_buf_size * sizeof(DataType);
    }

    template <typename YDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void
    operator()(YDistributedTensor_& y_tensor, void* smem, const ReduceFunc& reduce_func)
    {
        using DataType = typename YDistributedTensor_::DataType;

        constexpr int32_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        DataType* smem_ptr              = reinterpret_cast<DataType*>(smem);
        const int32_t lane_id           = ck_tile::get_lane_id();
        const int32_t warp_id           = ck_tile::get_warp_id();
        constexpr auto num_reduce_warps = GetReduceWarps<YDistributedTensor_>();
        constexpr int32_t num_warps     = BlockShape::BlockSize / warpSize;
        constexpr int32_t num_col       = GetColumnCount<YDistributedTensor_>();
        const int32_t col_idx           = lane_id % num_col;
        const int32_t smem_offset       = warp_id * num_col + col_idx;

        // skip if nonthing to do
        if constexpr(num_reduce_warps == 1)
            return;

        // store into smem only for lane-0 within one warp
        if(lane_id < num_col)
        {
            ck_tile::static_for<0, thread_buf_size, 1>{}([&](auto i) {
                smem_ptr[smem_offset + i * num_warps * num_col] = y_tensor.get_thread_buffer()[i];
            });
        }
        ck_tile::block_sync_lds();

        // load from smem. here we let everythread to do compute :)
        int32_t local_warp_id = warp_id / num_reduce_warps;
        int32_t local_smem_os = local_warp_id * num_reduce_warps;
        DataType all_scratch[thread_buf_size * num_reduce_warps];
        ck_tile::static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            ck_tile::static_for<0, num_reduce_warps, 1>{}([&](auto i_1) {
                all_scratch[i_0 * num_reduce_warps + i_1] =
                    smem_ptr[i_0 * num_warps * num_col + (local_smem_os + i_1) * num_col + col_idx];
            });
        });
        ck_tile::block_sync_lds(); // TODO: we don't need sync here

        ck_tile::static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            // TODO: use descriptor for this
            auto v_local = all_scratch[i_0 * num_reduce_warps];

            // further reduce mean/var
            ck_tile::static_for<0, num_reduce_warps - 1, 1>{}([&](auto i_1_n1) {
                constexpr auto i_1      = ck_tile::number<i_1_n1 + 1>{};
                const DataType v_remote = all_scratch[i_0 * num_reduce_warps + i_1];

                // reduce
                v_local = reduce_func(v_local, v_remote);
            });

            y_tensor.get_thread_buffer()(i_0) = v_local;
        });
    }
};