# Integration layer for hybrid KV cache with unified attention
# This module provides the interface between vLLM's hybrid KV cache manager
# and the enhanced unified attention kernels

import os
import torch
from typing import Optional, Dict, Any, List
import logging
import triton
import triton.language as tl
from aiter.ops.triton.utils.arch_info import get_num_sms
import math

logger = logging.getLogger(__name__)


# Add the simple hybrid kernel functions directly to this file
@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


# Simplified hybrid kernel that just adds group awareness to the original
@triton.jit
def kernel_simple_hybrid_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    # Simple group parameters - just offsets, no complex mappings
    group_block_offset,  # int64 - simple offset to add to block table access
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    BLOCK_N: tl.constexpr,  # int
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    tl.assume(block_table_stride > 0)
    tl.assume(query_stride_0 > 0)
    tl.assume(query_stride_1 > 0)
    tl.assume(output_stride_0 > 0)
    tl.assume(output_stride_1 > 0)
    tl.assume(stride_k_cache_0 > 0)
    tl.assume(stride_k_cache_1 > 0)
    tl.assume(stride_k_cache_2 > 0)
    tl.assume(stride_v_cache_0 > 0)
    tl.assume(stride_v_cache_1 > 0)
    tl.assume(stride_v_cache_2 > 0)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = offs_d < HEAD_SIZE
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # Add simple group offset to block table access
    block_table_offset = seq_idx * block_table_stride + group_block_offset

    if sink_ptr is None:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    if SLIDING_WINDOW > 0:
        num_blocks_start = (max_seq_prefix_len - SLIDING_WINDOW - 1) // BLOCK_N
        num_blocks_start = max(0, num_blocks_start - 1)
    else:
        num_blocks_start = 0

    USE_SINGLE_KV_LOAD: tl.constexpr = BLOCK_N == BLOCK_SIZE

    if USE_SINGLE_KV_LOAD:
        loop_end = cdiv_fn(max_seq_prefix_len, BLOCK_N)
        loop_step = 1
        loop_start = num_blocks_start
    else:
        loop_end = max_seq_prefix_len
        loop_step = BLOCK_N
        loop_start = num_blocks_start * BLOCK_N

    # iterate through tiles
    for j in range(loop_start, loop_end, loop_step):
        offs_n = tl.arange(0, BLOCK_N)
        
        if USE_SINGLE_KV_LOAD:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

            v_offset = (
                physical_block_idx * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + offs_n[:, None] * stride_v_cache_1
            )

            k_offset = (
                physical_block_idx * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + offs_n[None, :] * stride_k_cache_1
            )
            seq_offset = j * BLOCK_N + offs_n

            # K : (HEAD_SIZE_PADDED, BLOCK_N)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None],
                other=0.0,
            )
            # V : (BLOCK_N, HEAD_SIZE_PADDED)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :],
                other=0.0,
            )

        else:
            j = tl.multiple_of(j, BLOCK_N)
            seq_offset = j + offs_n
            load_mask = seq_offset < max_seq_prefix_len
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
                mask=load_mask,
                other=0,
            )

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (offs_n[:, None] % BLOCK_SIZE) * stride_v_cache_1
            )

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (offs_n[None, :] % BLOCK_SIZE) * stride_k_cache_1
            )

            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None],
                other=0.0,
            )

            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & load_mask[:, None],
                other=0.0,
            )

        # seq_mask: (BLOCK_M, BLOCK_N)
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # Handle FP8 scaling
        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # S : (BLOCK_M, BLOCK_N)
        S = tl.zeros(shape=(BLOCK_M, BLOCK_N), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, BLOCK_N)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    l_recip = 1 / L[:, None]
    acc = acc * l_recip

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def simple_hybrid_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    sinks=None,
):
    """Simplified hybrid unified attention that uses minimal modifications."""
    
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert (
        q.element_size() >= 2 or block_size >= 32
    ), "Block size must be at least 32 for fp8"
    assert (
        sinks is None or sinks.shape[0] == q.shape[1]
    ), "Sinks must be num_query_heads size"
    
    use_alibi_slopes = alibi_slopes is not None
    SLIDING_WINDOW = 1 + window_size[0] if window_size[0] > 0 else 0
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    
    if BLOCK_Q == 0:
        BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    target_num_prgms = get_num_sms() * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads

    # Simple group offset - just add 0 for now (no actual group logic)
    group_block_offset = 0
    
    # Configure kernel parameters
    num_stages_2d = 4
    BLOCK_N = block_size
    num_warps = 2
    
    if num_2d_prgms >= 2 * target_num_prgms:
        num_warps = 4
        if num_2d_prgms <= 4 * target_num_prgms:
            BLOCK_M = 64
            num_stages_2d = 2 if SLIDING_WINDOW > 0 else 4
        else:
            BLOCK_M = 64
            num_stages_2d = 1
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
        
    if max_seqlen_q >= 256 and max_seqlen_k >= 256 and SLIDING_WINDOW == 0:
        num_warps = 4
        BLOCK_M = 128
        num_stages_2d = 1
        BLOCK_N = 64
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    logger.info(f"Launching simple hybrid kernel with grid ({num_kv_heads}, {total_num_q_blocks})")

    kernel_simple_hybrid_attention_2d[
        (num_kv_heads, total_num_q_blocks)
    ](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        group_block_offset=group_block_offset,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=SLIDING_WINDOW,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        waves_per_eu=2,
        num_warps=num_warps,
        num_stages=num_stages_2d,
    )


class HybridKVCacheGroupManager:
    """Manages KV cache group configurations for hybrid attention kernels."""
    
    def __init__(self, kv_cache_config, device):
        self.kv_cache_config = kv_cache_config
        self.device = device
        self.num_groups = len(kv_cache_config.kv_cache_groups)
        
        # Pre-compute mappings for efficient kernel access
        self._create_group_mappings()
        self._validate_configuration()
    
    def _create_group_mappings(self):
        """Create efficient mappings for kernel access."""
        # Create layer name to group ID mapping
        self.layer_to_group = {}
        self.group_to_layers = {}
        
        for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            self.group_to_layers[group_id] = group_spec.layer_names
            for layer_name in group_spec.layer_names:
                self.layer_to_group[layer_name] = group_id
        
        # Create head-to-group mapping (assumes sequential head assignment)
        self.head_to_group_mapping = {}
        current_head = 0
        
        for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            num_layers_in_group = len(group_spec.layer_names)
            # Assume each layer contributes equally to KV heads
            for _ in range(num_layers_in_group):
                self.head_to_group_mapping[current_head] = group_id
                current_head += 1
    
    def _validate_configuration(self):
        """Validate that the configuration is compatible with hybrid kernels."""
        if self.num_groups == 0:
            raise ValueError("No KV cache groups found in configuration")
        
        # Check that all groups have compatible specifications
        first_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        for group in self.kv_cache_config.kv_cache_groups[1:]:
            spec = group.kv_cache_spec
            if spec.block_size != first_spec.block_size:
                logger.warning(f"Inconsistent block sizes across groups: "
                             f"{spec.block_size} vs {first_spec.block_size}")
            if spec.head_size != first_spec.head_size:
                raise ValueError(f"Inconsistent head sizes across groups: "
                               f"{spec.head_size} vs {first_spec.head_size}")
    
    def create_kernel_tensors(self, num_kv_heads: int):
        """Create tensors needed for hybrid kernel execution."""
        
        # KV head to group mapping
        kv_group_mapping = torch.zeros(num_kv_heads, dtype=torch.int32, device=self.device)
        for head_idx in range(num_kv_heads):
            group_id = self.head_to_group_mapping.get(head_idx, 0)
            kv_group_mapping[head_idx] = group_id
        
        # Group offsets (cumulative layer counts)
        kv_group_offsets = torch.zeros(self.num_groups + 1, dtype=torch.int32, device=self.device)
        current_offset = 0
        for group_id in range(self.num_groups):
            kv_group_offsets[group_id] = current_offset
            current_offset += len(self.kv_cache_config.kv_cache_groups[group_id].layer_names)
        kv_group_offsets[self.num_groups] = current_offset
        
        # Block table base offsets for each group
        kv_group_block_tables = torch.zeros(self.num_groups, dtype=torch.int64, device=self.device)
        for group_id in range(self.num_groups):
            # Calculate block table offset based on group configuration
            # This is a simplified calculation - adjust based on actual block table layout
            kv_group_block_tables[group_id] = group_id * self.kv_cache_config.num_blocks
        
        return kv_group_mapping, kv_group_offsets, kv_group_block_tables
    
    def create_stride_tensors(self, k_cache, v_cache):
        """Create stride tensors for each KV cache group."""
        kv_group_strides = torch.zeros(self.num_groups, 8, dtype=torch.int64, device=self.device)
        
        for group_id in range(self.num_groups):
            group_spec = self.kv_cache_config.kv_cache_groups[group_id]
            
            # Get base strides from the cache tensors
            k_strides = k_cache.stride()
            v_strides = v_cache.stride()
            
            # Store K cache strides (first 4 elements)
            for i in range(min(4, len(k_strides))):
                kv_group_strides[group_id, i] = k_strides[i]
            
            # Store V cache strides (next 4 elements)
            for i in range(min(4, len(v_strides))):
                kv_group_strides[group_id, 4 + i] = v_strides[i]
        
        return kv_group_strides


def detect_hybrid_kv_cache_requirement(kv_cache_config) -> bool:
    """Detect if hybrid KV cache support is needed."""
    
    # Force hybrid mode if environment variable is set
    if os.getenv("VLLM_FORCE_HYBRID_ATTENTION") == "1":
        logger.info("Forcing hybrid KV cache mode due to VLLM_FORCE_HYBRID_ATTENTION=1")
        return True
    
    # TEMPORARY: Always use hybrid for debugging
    if os.getenv("VLLM_DEBUG_ALWAYS_HYBRID") == "1":
        logger.info("DEBUG: Always using hybrid mode")
        return True
    
    if not kv_cache_config:
        logger.info("No KV cache config provided")
        return False
        
    if not hasattr(kv_cache_config, 'kv_cache_groups'):
        logger.info("KV cache config has no kv_cache_groups attribute")
        return False
    
    num_groups = len(kv_cache_config.kv_cache_groups)
    logger.info(f"KV cache config has {num_groups} groups")
    
    # Always use hybrid if we have any groups (even single group to ensure compatibility)
    if num_groups == 0:
        logger.info("No KV cache groups found")
        return False
    
    # Log group information for debugging
    for i, group in enumerate(kv_cache_config.kv_cache_groups):
        logger.info(f"Group {i}: {len(group.layer_names)} layers, spec type: {type(group.kv_cache_spec).__name__}")
    
    # Use hybrid mode if:
    # 1. We have multiple groups, OR
    # 2. The --disable-hybrid-kv-cache-manager flag was NOT used (meaning hybrid cache is active)
    if num_groups > 1:
        logger.info("Multiple KV cache groups detected - using hybrid mode")
        return True
    
    # Even with single group, use hybrid if the group has multiple layers
    # This handles cases where hybrid cache manager is active but creates a single unified group
    single_group = kv_cache_config.kv_cache_groups[0]
    if len(single_group.layer_names) > 1:
        logger.info(f"Single group with {len(single_group.layer_names)} layers - using hybrid mode for compatibility")
        return True
    
    logger.info("Using standard mode - single group with single layer")
    return False


def enhanced_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    kv_cache_config=None,
    alibi_slopes=None,
    sinks=None,
):
    """Enhanced unified attention that automatically handles hybrid KV cache."""
    
    # Debug logging
    logger.info(f"enhanced_unified_attention called with kv_cache_config: {kv_cache_config is not None}")
    
    # Check if we need hybrid KV cache support
    use_hybrid = detect_hybrid_kv_cache_requirement(kv_cache_config)
    
    # Also check for force flag
    force_hybrid = os.getenv("VLLM_FORCE_HYBRID_ATTENTION") == "1"
    
    logger.info(f"use_hybrid={use_hybrid}, force_hybrid={force_hybrid}, kv_cache_config is not None={kv_cache_config is not None}")
    
    if (use_hybrid or force_hybrid):
        logger.info("Attempting to use hybrid KV cache compatible unified attention")
        
        # If kv_cache_config is None but we're forcing hybrid, create a dummy config
        if kv_cache_config is None and force_hybrid:
            logger.info("Creating dummy KV cache config for forced hybrid mode")
            # Create a minimal dummy config for testing
            from collections import namedtuple
            
            DummySpec = namedtuple('DummySpec', ['block_size', 'head_size', 'num_kv_heads', 'dtype'])
            DummyGroup = namedtuple('DummyGroup', ['layer_names', 'kv_cache_spec'])
            DummyConfig = namedtuple('DummyConfig', ['kv_cache_groups', 'num_blocks'])
            
            # Get basic info from tensors
            num_kv_heads = k.shape[2] if len(k.shape) > 2 else k.shape[1]
            head_size = q.shape[2] if len(q.shape) > 2 else q.shape[1]
            block_size = v.shape[1] if len(v.shape) > 1 else 128
            
            dummy_spec = DummySpec(
                block_size=block_size,
                head_size=head_size, 
                num_kv_heads=num_kv_heads,
                dtype=k.dtype
            )
            
            dummy_group = DummyGroup(
                layer_names=[f"layer_{i}" for i in range(num_kv_heads)],
                kv_cache_spec=dummy_spec
            )
            
            kv_cache_config = DummyConfig(
                kv_cache_groups=[dummy_group],
                num_blocks=1000  # dummy value
            )
            
            logger.info(f"Created dummy config with {len(kv_cache_config.kv_cache_groups)} groups")
        
        if kv_cache_config is not None:
            try:
                # Create hybrid cache manager
                cache_manager = HybridKVCacheGroupManager(kv_cache_config, q.device)
                
                # Get number of KV heads
                num_kv_heads = k.shape[2] if len(k.shape) > 2 else k.shape[1]
                logger.info(f"Number of KV heads: {num_kv_heads}")
                
                # Create kernel tensors
                kv_group_mapping, kv_group_offsets, kv_group_block_tables = \
                    cache_manager.create_kernel_tensors(num_kv_heads)
                kv_group_strides = cache_manager.create_stride_tensors(k, v)
                
                logger.info(f"Successfully created hybrid kernel tensors:")
                logger.info(f"  - kv_group_mapping: {kv_group_mapping.shape}")
                logger.info(f"  - kv_group_offsets: {kv_group_offsets.shape}")
                logger.info(f"  - kv_group_block_tables: {kv_group_block_tables.shape}")
                logger.info(f"  - kv_group_strides: {kv_group_strides.shape}")
                
                # Call simple hybrid kernel (without complex group management for now)
                logger.info("Calling simple hybrid kernel...")
                return simple_hybrid_unified_attention(
                    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                    softmax_scale, causal, window_size, block_table, softcap,
                    q_descale, k_descale, v_descale, alibi_slopes, sinks
                )
                
            except Exception as e:
                logger.error(f"Failed to use hybrid kernel: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # If we're forcing hybrid, re-raise the error instead of falling back
                if force_hybrid:
                    logger.error("VLLM_FORCE_HYBRID_ATTENTION=1 but hybrid kernel failed - raising error")
                    raise e
                else:
                    logger.info("Falling back to standard unified attention")
                    # Fall through to standard implementation
        else:
            logger.error("kv_cache_config is None but hybrid mode was requested")
            if force_hybrid:
                raise ValueError("VLLM_FORCE_HYBRID_ATTENTION=1 but kv_cache_config is None")
    
    logger.info("Using standard unified attention (no hybrid KV cache)")
    # Fall back to original unified attention
    try:
        from .unified_attention import unified_attention
        return unified_attention(
            q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
            softmax_scale, causal, window_size, block_table, softcap,
            q_descale, k_descale, v_descale, alibi_slopes, sinks
        )
    except ImportError:
        # Try different import path
        try:
            import sys
            # Add current directory to import path for testing
            sys.path.insert(0, '.')
            from unified_attention import unified_attention
            return unified_attention(
                q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size, block_table, softcap,
                q_descale, k_descale, v_descale, alibi_slopes, sinks
            )
        except ImportError:
            logger.error("Could not import original unified_attention function")
            raise RuntimeError("Both hybrid and standard unified attention unavailable")


def _call_hybrid_kernel(
    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap,
    q_descale, k_descale, v_descale, alibi_slopes, sinks,
    kv_group_mapping, kv_group_offsets, kv_group_block_tables, kv_group_strides,
    num_groups
):
    """Call the hybrid unified attention kernel with proper configuration."""
    
    from .hybrid_unified_attention import kernel_hybrid_unified_attention_2d
    from aiter.ops.triton.utils.arch_info import get_num_sms
    import triton
    
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert (
        q.element_size() >= 2 or block_size >= 32
    ), "Block size must be at least 32 for fp8"
    assert (
        sinks is None or sinks.shape[0] == q.shape[1]
    ), "Sinks must be num_query_heads size"
    
    use_alibi_slopes = alibi_slopes is not None
    SLIDING_WINDOW = 1 + window_size[0] if window_size[0] > 0 else 0
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    
    if BLOCK_Q == 0:
        BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    target_num_prgms = get_num_sms() * 4
    
    # Configure kernel parameters
    num_stages_2d = 4
    BLOCK_N = block_size
    num_warps = 2
    
    # Adjust parameters based on workload size
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    if num_2d_prgms >= 2 * target_num_prgms:
        num_warps = 4
        if num_2d_prgms <= 4 * target_num_prgms:
            BLOCK_M = 64
            num_stages_2d = 2 if SLIDING_WINDOW > 0 else 4
        else:
            BLOCK_M = 64
            num_stages_2d = 1
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    
    if max_seqlen_q >= 256 and max_seqlen_k >= 256 and SLIDING_WINDOW == 0:
        num_warps = 4
        BLOCK_M = 128
        num_stages_2d = 1
        BLOCK_N = 64
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Launch hybrid kernel
    kernel_hybrid_unified_attention_2d[
        (num_kv_heads, total_num_q_blocks)
    ](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        kv_group_mapping_ptr=kv_group_mapping,
        kv_group_offsets_ptr=kv_group_offsets,
        kv_group_block_tables_ptr=kv_group_block_tables,
        kv_group_strides_ptr=kv_group_strides,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        base_block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=SLIDING_WINDOW,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_groups=num_groups,
        waves_per_eu=2,
        num_warps=num_warps,
        num_stages=num_stages_2d,
    )


# Monkey patch detection for automatic integration
def should_use_enhanced_attention():
    """Determine if enhanced attention should be used based on environment."""
    return (
        os.getenv("VLLM_ROCM_USE_AITER") == "1" and 
        os.getenv("VLLM_USE_AITER_UNIFIED_ATTENTION") == "1" and
        os.getenv("VLLM_DISABLE_HYBRID_ATTENTION") != "1"
    )


# Integration hook for vLLM
def patch_unified_attention():
    """Patch the unified attention function to use hybrid version when needed."""
    try:
        import vllm.attention.ops.triton_unified_attention as triton_attention
        
        # Store original function
        triton_attention._original_unified_attention = triton_attention.unified_attention
        
        # Replace with enhanced version
        triton_attention.unified_attention = enhanced_unified_attention
        
        logger.info("Successfully patched unified attention for hybrid KV cache support")
        
    except ImportError as e:
        logger.warning(f"Could not patch unified attention: {e}")
    except AttributeError as e:
        logger.warning(f"Unified attention module structure has changed: {e}")


# Auto-patch when module is imported
if should_use_enhanced_attention():
    patch_unified_attention()