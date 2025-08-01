# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from typing import Optional
import aiter
from aiter import logger
from aiter import pertoken_quant, get_hip_quant
from aiter import ActivationType, QuantType, dtypes

BLOCK_SIZE_M = 32


def moe_sorting_ck(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty((1), dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    aiter.moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
        expert_mask,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def asm_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    a16=False,
    per_tensor_quant_scale=None,
    block_shape=None,
    expert_mask=None,
    activation=ActivationType.Silu,
):
    E, model_dim, inter_dim = w2.shape
    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype
    device = topk_ids.device
    lastdim_mul = 8 if w1.dtype in {dtypes.i32, torch.uint32} else 1
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting_ck(
            topk_ids, topk_weight, global_E, model_dim, dtype, BLOCK_SIZE_M, expert_mask
        )
    )

    if fc1_scale is None:
        # pure bf16
        aiter.fmoe(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
        )
    elif a16:
        # a16w8 smooth quant fmoe
        if w1.dtype in [dtypes.fp8, dtypes.i8] and inter_dim * 2 == w1.shape[1]:
            aiter.fmoe_g1u1_a16(
                moe_buf,
                hidden_states,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                activation,
            )
        elif w1.dtype == dtypes.i8 and inter_dim == w1.shape[1]:
            aiter.fmoe_int8_g1u0_a16(
                moe_buf,
                hidden_states,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
            )
        else:
            raise ValueError(f"Invalid args: {w1.dtype} {w1.shape=} {w2.shape=}")
    elif block_shape is not None:
        assert (
            dtype == torch.bfloat16
        ), "asm_moe for block_scale only support bfloat16 hidden_states"
        assert block_shape == (
            128,
            128,
        ), "asm_moe for block_scale only support (128, 128)"
        assert (
            w1.dtype == torch.float8_e4m3fnuz
        ), "asm_moe for block_scale only support float8_e4m3fnuz weight"
        assert (
            w2.shape[2] * 2 == w1.shape[1]
        ), "aiter moe for block_scale only support g1u1"
        scale_blk_n, scale_blk_k = block_shape
        hidden_states = hidden_states.view(M * model_dim // scale_blk_k, scale_blk_k)

        a1_q, a1_scale = pertoken_quant(
            hidden_states.view(-1, model_dim // scale_blk_k, scale_blk_k),
            quant_dtype=torch.float8_e4m3fnuz,
        )
        a1_q = a1_q.view(-1, model_dim)
        a1_scale = a1_scale.squeeze(-1).t().contiguous()

        scale_blk_n, scale_blk_k = block_shape
        aiter.fmoe_fp8_blockscale_g1u1(
            moe_buf,
            a1_q,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            fc1_scale,
            fc2_scale,
            scale_blk_n,
            scale_blk_k,
            None,
        )
    else:
        # a8w8 fmoe, opt: smooth quant
        a8_type = (
            w1.dtype
            if w1.dtype != dtypes.i32 and w1.dtype != torch.uint32
            else dtypes.fp8
        )
        if fc1_smooth_scale is not None:
            a8 = torch.empty((topk * M, model_dim), dtype=a8_type, device=device)
            a8_scale = torch.empty((topk * M), dtype=dtypes.fp32, device=device)

            # moe_smoothquant_fwd need topk_ids which contains local_expert_id
            if expert_mask is not None:
                local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
                local_expert_hash[local_expert_hash > 0] -= 1
                topk_ids = local_expert_hash[topk_ids]

            aiter.moe_smoothquant_fwd(
                a8, hidden_states, fc1_smooth_scale, topk_ids, a8_scale
            )
        else:
            if (
                w1.dtype == dtypes.fp8
                or w1.dtype == dtypes.i32
                and w1.dtype == torch.uint32
            ):
                a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
                a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
                if per_tensor_quant_scale is None:
                    aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)
                else:
                    aiter.static_per_tensor_quant(
                        a8, hidden_states, per_tensor_quant_scale
                    )
                    a8_scale.fill_(per_tensor_quant_scale)
            elif w1.dtype == dtypes.i8:
                a8 = torch.empty((M, model_dim), dtype=w1.dtype, device=device)
                a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
                fc1_smooth_scale = torch.ones(
                    model_dim, dtype=dtypes.fp32, device=device
                )
                aiter.smoothquant_fwd(a8, hidden_states, fc1_smooth_scale, a8_scale)
            else:
                logger.warning("FMOE fall into pure torch quant...")
                a8, a8_scale = aiter.pertoken_quant(hidden_states, quant_dtype=w1.dtype)
        if w2.shape[2] * lastdim_mul == w1.shape[1]:
            fmoe_func = aiter.fmoe_int8_g1u0
        elif w2.shape[2] * 2 * lastdim_mul == w1.shape[1]:
            fmoe_func = aiter.fmoe_g1u1
        else:
            raise ValueError(
                f"Invalid MoE weight: {w1.shape=} {w2.shape=} {lastdim_mul}"
            )

        fmoe_func(
            moe_buf,
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a8_scale,
            fc1_scale,
            fc2_scale,
            fc2_smooth_scale,
            activation,
        )
        #   fc2_smooth_scale)
    return moe_buf


def asm_moe_tkw1(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    a16=False,
    per_tensor_quant_scale=None,
    expert_mask=None,
    activation=ActivationType.Silu,
):
    E, model_dim, inter_dim = w2.shape
    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype
    device = topk_ids.device
    lastdim_mul = 8 if w1.dtype in {dtypes.i32, torch.uint32} else 1
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting_ck(
            topk_ids, topk_weight, global_E, model_dim, dtype, BLOCK_SIZE_M, expert_mask
        )
    )

    if fc1_scale is None:
        # pure bf16
        aiter.fmoe(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
        )
    elif a16:
        # a16w8 smooth quant fmoe
        if w1.dtype == dtypes.fp8 and inter_dim * 2 == w1.shape[1]:
            aiter.fmoe_fp8_g1u1_a16(
                moe_buf,
                hidden_states,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
            )
        elif w1.dtype == dtypes.i8 and inter_dim == w1.shape[1]:
            aiter.fmoe_int8_g1u0_a16(
                moe_buf,
                hidden_states,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
            )
        else:
            raise ValueError(f"Invalid args: {w1.dtype} {w1.shape=} {w2.shape=}")

    else:
        # a8w8 fmoe, opt: smooth quant
        a8_type = (
            w1.dtype
            if w1.dtype != dtypes.i32 and w1.dtype != torch.uint32
            else dtypes.fp8
        )
        if fc1_smooth_scale is not None:
            a8 = torch.empty((topk * M, model_dim), dtype=a8_type, device=device)
            a8_scale = torch.empty((topk * M), dtype=dtypes.fp32, device=device)

            # moe_smoothquant_fwd need topk_ids which contains local_expert_id
            if expert_mask is not None:
                local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
                local_expert_hash[local_expert_hash > 0] -= 1
                topk_ids = local_expert_hash[topk_ids]

            aiter.moe_smoothquant_fwd(
                a8, hidden_states, fc1_smooth_scale, topk_ids, a8_scale
            )
        else:
            if (
                w1.dtype == dtypes.fp8
                or w1.dtype == dtypes.i32
                and w1.dtype == torch.uint32
            ):
                a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
                a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
                if per_tensor_quant_scale is None:
                    aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)
                else:
                    aiter.static_per_tensor_quant(
                        a8, hidden_states, per_tensor_quant_scale
                    )
                    a8_scale.fill_(per_tensor_quant_scale)
            elif w1.dtype == dtypes.i8:
                a8 = torch.empty((M, model_dim), dtype=w1.dtype, device=device)
                a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
                fc1_smooth_scale = torch.ones(
                    model_dim, dtype=dtypes.fp32, device=device
                )
                aiter.smoothquant_fwd(a8, hidden_states, fc1_smooth_scale, a8_scale)
            else:
                logger.warning("FMOE fall into pure torch quant...")
                a8, a8_scale = aiter.pertoken_quant(hidden_states, quant_dtype=w1.dtype)
        if w2.shape[2] * 2 * lastdim_mul == w1.shape[1]:
            fmoe_func = aiter.fmoe_g1u1_tkw1

        else:
            raise ValueError(
                f"Invalid MoE weight: {w1.shape=} {w2.shape=} {lastdim_mul}"
            )

        fmoe_func(
            moe_buf,
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a8_scale,
            fc1_scale,
            fc2_scale,
            fc2_smooth_scale,
            activation,
        )
        #   fc2_smooth_scale)
    return moe_buf


def get_block_size(token, topk, expert):
    token_per_expert = token * topk / expert
    support_list = [32, 64, 128]
    for el in support_list:
        if token_per_expert <= el * 4:
            return el
    return support_list[-1]


# Only support fp8 per tensor quant
def ck_moe_2stages(
    a1,
    w1,  # [expert(local_expert:EP), inter_dim(*2), dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    quant_type=QuantType.No,
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [1]
    a2_scale=None,  # [1]
    block_size=None,
    expert_mask=None,
    activation=ActivationType.Silu,
    doweight_stage1=False,
):

    quant_func = get_hip_quant(quant_type)
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else torch.float8_e4m3fnuz

    # quant_func = get_torch_quant(quant_type)
    E, model_dim, inter_dim = w2.shape
    if w1.dtype is torch.uint32:
        inter_dim = inter_dim * 8

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    M, topk = topk_ids.shape
    dtype = a1.dtype
    device = topk_ids.device
    if block_size is None:
        block_size = get_block_size(M, topk, E)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting_ck(
            topk_ids, topk_weight, global_E, model_dim, dtype, block_size, expert_mask
        )
    )
    # print("block_size:", block_size, sorted_expert_ids)
    a1, a1_scale = quant_func(a1, scale=a1_scale, quant_dtype=q_dtype_a)

    a2 = torch.empty(
        (M, topk, inter_dim),
        dtype=dtype,
        device=device,
    )

    if activation == ActivationType.Silu:
        act_op = 1  # silu_and_mul
    else:
        act_op = 0  # gelu_and_mul

    aiter.ck_moe_stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        topk,
        fc1_scale,
        a1_scale,
        block_size,
        sorted_weights if doweight_stage1 else None,
        act_op,
    )

    if quant_type == QuantType.per_Token:
        a2 = a2.view(M, -1)
    a2, a2_scale = quant_func(a2, scale=a2_scale, quant_dtype=q_dtype_a)
    a2 = a2.view(M, topk, -1)

    aiter.ck_moe_stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        topk,
        fc2_scale,
        a2_scale,
        block_size,
        sorted_weights if not doweight_stage1 else None,
    )
    return moe_buf


def torch_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]
    if w2.shape[2] * 2 == w1.shape[1]:
        # g1u1(w1 include gate and up)
        moeType = "g1u1"
    else:
        # g1u0(w1 only include gate)
        moeType = "g1u0"

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(gate) * up
                else:
                    act_out = F.silu(gate) * up
            else:
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(act_input)
                else:
                    act_out = F.silu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


def torch_moe_tkw1(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]
    if w2.shape[2] * 2 == w1.shape[1]:
        # g1u1(w1 include gate and up)
        moeType = "g1u1"
    else:
        # g1u0(w1 only include gate)
        moeType = "g1u0"

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                gate = gate * (topk_weight.view(B, -1, 1)[mask])
                up = up * (topk_weight.view(B, -1, 1)[mask])
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(gate) * up
                else:
                    act_out = F.silu(gate) * up
            else:
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(act_input)
                else:
                    act_out = F.silu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            act_out, act_out_scale = pertoken_quant(
                act_out, quant_dtype=dtypes.fp8, dtypeMax=None
            )
            out[mask] = (
                act_out.to(computeType)
                @ (w2[E_id].transpose(0, 1))
                * act_out_scale.view(-1, 1)
            )

    return out.sum(dim=1).to(dtype)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    if topk_weights is None:
        topk_weights = torch.empty(
            M, topk, dtype=dtypes.fp32, device=hidden_states.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=dtypes.i32, device=hidden_states.device
    )

    aiter.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
        renormalize,
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    # if renormalize:
    #     topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
