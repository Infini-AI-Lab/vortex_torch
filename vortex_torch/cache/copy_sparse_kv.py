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
    slots_used_bitmap: torch.Tensor,
    needs_eviction_bitmap: torch.Tensor,
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
    Pre-partitions pages into per-block buckets, then each CUDA block processes
    its pages using shared memory for all set operations.
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
        slots_used_bitmap,
        needs_eviction_bitmap,
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
    slots_used_bitmap: torch.Tensor,
    needs_eviction_bitmap: torch.Tensor,
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
    LRU allocation: global device semaphores, TryLock with blocking fallback.
    Split into 2 kernel launches: no_evict + with_evict.
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
        slots_used_bitmap,
        needs_eviction_bitmap,
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
