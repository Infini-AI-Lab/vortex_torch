"""
Sparse KV cache copy utilities for unified CPU/GPU memory management.

This module provides functions for copying sparse KV cache pages from CPU to GPU
with LRU eviction support, optimized for CUDA graph compatibility.
"""

import torch
from typing import Tuple


def allocate_pages_lru_block(
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    slot_ages: torch.Tensor,
    set_used_mask: torch.Tensor,
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    overflow_flag: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int,
    max_hash_attempts: int = 30,
) -> None:
    """
    LRU allocation: block-local shared-memory with 32-way set-associative cache.
    Same optimizations as lru_block_global (butterfly reduction, uint32 bitmask),
    but no global fallback — pages that can't be resolved block-locally overflow.
    """
    import vortex_torch_C

    indptr_last_idx = batch_size * num_kv_heads

    vortex_torch_C.allocate_pages_lru_block(
        sparse_kv_indices,
        sparse_kv_indptr,
        indptr_last_idx,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        slot_ages,
        set_used_mask,
        dst_gpu_slots,
        owners_bitmap,
        evicted_cpu_pages,
        overflow_flag,
        max_num_pages,
        max_hash_attempts
    )


def allocate_pages_lru_global(
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    slot_ages: torch.Tensor,
    set_used_mask: torch.Tensor,
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    overflow_flag: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int,
    max_hash_attempts: int = 30,
) -> None:
    """
    LRU allocation: global device semaphores for all sets.
    Same optimizations as lru_block_global (butterfly reduction, uint32 bitmask).
    """
    import vortex_torch_C

    indptr_last_idx = batch_size * num_kv_heads

    vortex_torch_C.allocate_pages_lru_global(
        sparse_kv_indices,
        sparse_kv_indptr,
        indptr_last_idx,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        slot_ages,
        set_used_mask,
        dst_gpu_slots,
        owners_bitmap,
        evicted_cpu_pages,
        overflow_flag,
        max_num_pages,
        max_hash_attempts
    )


def allocate_pages_lru_block_global(
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    slot_ages: torch.Tensor,
    set_used_mask: torch.Tensor,
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    overflow_flag: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int,
    max_hash_attempts: int = 30,
) -> None:
    """
    LRU allocation: block-local smem with relative ages (uint8, overflow-safe)
    + device semaphore global fallback.
    """
    import vortex_torch_C

    indptr_last_idx = batch_size * num_kv_heads

    vortex_torch_C.allocate_pages_lru_block_global(
        sparse_kv_indices,
        sparse_kv_indptr,
        indptr_last_idx,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        slot_ages,
        set_used_mask,
        dst_gpu_slots,
        owners_bitmap,
        evicted_cpu_pages,
        overflow_flag,
        max_num_pages,
        max_hash_attempts,
    )


def allocate_pages_block_global(
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    slot_state: torch.Tensor,
    set_used_mask: torch.Tensor,
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    overflow_flag: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int,
    max_hash_attempts: int = 30,
    cache_policy: int = 0,
) -> None:
    """
    Policy-agnostic allocation: block-local smem + global fallback.

    cache_policy: 0 = LRU, 1 = LFU, 2 = RANDOM
    """
    import vortex_torch_C

    indptr_last_idx = batch_size * num_kv_heads

    vortex_torch_C.allocate_pages_block_global(
        sparse_kv_indices,
        sparse_kv_indptr,
        indptr_last_idx,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        slot_state,
        set_used_mask,
        dst_gpu_slots,
        owners_bitmap,
        evicted_cpu_pages,
        overflow_flag,
        max_num_pages,
        max_hash_attempts,
        cache_policy,
    )


def copy_kv(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_buffer: torch.Tensor,
    gpu_v_buffer: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    page_size: int,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int
) -> None:
    """
    Copy kernel — grid-stride with auto-detected SM count.
    Copies KV data from CPU pinned memory to GPU based on allocation results.
    """
    import vortex_torch_C

    vortex_torch_C.copy_kv(
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_buffer,
        gpu_v_buffer,
        sparse_kv_indices,
        sparse_kv_indptr,
        dst_gpu_slots,
        owners_bitmap,
        evicted_cpu_pages,
        page_size,
        batch_size,
        num_kv_heads,
        max_num_pages
    )


def dequant_int8_cpu_to_bf16(
    cpu_int8_buffer: torch.Tensor,    # CPU pinned int8
    gpu_scale_buffer: torch.Tensor,   # GPU fp16 scales
    gpu_dst_buffer: torch.Tensor,     # GPU bf16 destination
    src_page_ids: torch.Tensor,       # int32 GPU: which CPU pages to read
    dst_page_ids: torch.Tensor,       # int32 GPU: where to write in destination
    page_size: int,
    head_dim: int,
) -> None:
    """
    Dequantize int8 pages from CPU pinned memory to GPU bf16 destination.

    Reads int8 data via CUDA UVA, multiplies by GPU-resident fp16 scales,
    writes bf16 result to GPU destination buffer.

    Supports separate source (CPU page IDs) and destination (GPU page IDs)
    for both in-place (forward_cache) and compact (extend gather) layouts.
    """
    import vortex_torch_C
    vortex_torch_C.dequant_int8_cpu_to_bf16(
        cpu_int8_buffer,
        gpu_scale_buffer,
        gpu_dst_buffer,
        src_page_ids,
        dst_page_ids,
        page_size,
        head_dim,
    )


def gather_pages_to_ragged(
    src_kv: torch.Tensor,         # paged KV buffer (CPU pinned or GPU)
    dst_buf: torch.Tensor,        # output ragged buffer [max_tokens, num_kv_heads, head_dim] bf16 GPU
    page_indices: torch.Tensor,   # [total_pages] int32 GPU
    kv_indptr: torch.Tensor,      # [bs * num_kv_heads + 1] int32 GPU
    dst_offsets: torch.Tensor,    # [bs] int32 GPU: token offset per request
    total_pages: int,
    num_kv_heads: int,
    page_size: int,
    head_dim: int,
    bs: int,
    quant_type: int = 0,          # 0=bf16, 1=int8, 2=fp8_e4m3, 3=fp8_e5m2
    kv_scale: float = 1.0,        # per-tensor scale (fp8 only)
    src_scale: torch.Tensor = None,  # per-token scales (int8 only)
) -> None:
    """
    Gather scattered per-head pages into contiguous multi-head ragged buffer.

    Reads from paged KV (CPU pinned via UVA or GPU), optionally dequantizes
    (int8/fp8 → bf16), and writes directly to the ragged output buffer with
    layout [total_tokens, num_kv_heads, head_dim].

    Eliminates temp GPU allocations and Python rearrangement loops.
    """
    import vortex_torch_C
    if src_scale is None:
        src_scale = torch.empty(0, dtype=torch.float16, device=dst_buf.device)
    vortex_torch_C.gather_pages_to_ragged(
        src_kv,
        dst_buf,
        page_indices,
        kv_indptr,
        dst_offsets,
        total_pages,
        num_kv_heads,
        page_size,
        head_dim,
        bs,
        quant_type,
        kv_scale,
        src_scale,
    )
