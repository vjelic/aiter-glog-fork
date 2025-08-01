// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_common.cuh"
#include "gemm_a8w8_bpreshuffle_lookup.h"
#include "gemm_a8w8_bpreshuffle_manifest.h"
#include "gemm_common.h"
#include <cmath>

using RowwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash
{
    size_t operator()(const std::tuple<int, int, int>& t) const
    {
        auto hash1 = std::hash<int>{}(std::get<0>(t));
        auto hash2 = std::hash<int>{}(std::get<1>(t));
        auto hash3 = std::hash<int>{}(std::get<2>(t));
        return hash1 ^ hash2 ^ hash3;
    }
};

using RowwiseKernelMap = std::unordered_map<std::tuple<int, int, int>, RowwiseKernel, IntTupleHash>;

template <typename DDataType, typename EDataType = DDataType>
RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K)
{
    if(K >= 1536)
    {
        if(M < 256 && K % 512 == 0)
        {
            if(N < 1536)
            {
                return a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
                    DDataType,
                    EDataType>;
            }
            else
            {
                return a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<
                    DDataType,
                    EDataType>;
            }
        }
        else
        {
            if(N < 1536)
            {
                return a8w8_bpreshuffle_256x128x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<
                    DDataType,
                    EDataType>;
            }
            else
            {
                return a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<
                    DDataType,
                    EDataType>;
            }
        }
    }
    else if(K >= 512)
    {
        if(M < 64)
        {
            return a8w8_bpreshuffle_128x16x32x128_16x16_16x16_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
                DDataType,
                EDataType>;
        }
        else if(M <= 256)
        {
            return a8w8_bpreshuffle_256x128x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<
                DDataType,
                EDataType>;
        }
        else
        {
            return a8w8_bpreshuffle_256x128x128x64_16x16_16x16_4x64x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v3<
                DDataType,
                EDataType>;
        }
    }
    else if(K >= 192 && K % 64 == 0)
    {
        if(M < 128)
        {
            return a8w8_bpreshuffle_128x16x256x64_16x16_16x16_4x16x1_4x32x1_1x16x1x8_8x8x1_1x2_intrawave_v1<
                DDataType,
                EDataType>;
        }
        else if(M <= 256)
        {
            return a8w8_bpreshuffle_256x32x256x64_16x16_16x16_4x32x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v1<
                DDataType,
                EDataType>;
        }
        else
        {
            return a8w8_bpreshuffle_256x128x128x64_16x16_16x16_4x64x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v3<
                DDataType,
                EDataType>;
        }
    }
    else
    {
        TORCH_CHECK(false,
                    "Unsupported K for heuristic dispatch: ",
                    K,
                    ". Supported K greater than 192 and K % 64 == 0.");
    }
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
    if(num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DDataType, typename EDataType = DDataType>
RowwiseKernel rowwise_dispatch(int M, int N, int K)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    static const auto lookup = [] {
        if constexpr(std::is_same_v<EDataType, F16>)
        {
            return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, F16)};
        }
        else if constexpr(std::is_same_v<EDataType, B16>)
        {
            return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, B16)};
        }
        else
        {
            static_assert(false, "rowwise_dispatch used with unsupported dtype!");
        }
    }();

    // First check if this shape(M,N,K) is available in the direct lookup.
    auto it = lookup.find({M, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }

    int padded_m = M;
  
    // Fine-grained search
    padded_m = getPaddedM(M, N, K, 0);
    // Second check if this shape(padded_m,N,K) is available in the direct lookup.
    it = lookup.find({padded_m, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }
  
    // Coarse-grained search
    padded_m = getPaddedM(M, N, K, 1);
    // Third check if this shape(padded_m,N,K) is available in the direct lookup.
    it = lookup.find({padded_m, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }

    // Otherwise, use heuristics.
    return rowwise_heuristic_dispatch<DDataType, EDataType>(M, N, K);
}

torch::Tensor gemm_a8w8_bpreshuffle(torch::Tensor& XQ,
                                    torch::Tensor& WQ,
                                    torch::Tensor& x_scale,
                                    torch::Tensor& w_scale,
                                    torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        rowwise_dispatch<F32, F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        rowwise_dispatch<F32, B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
