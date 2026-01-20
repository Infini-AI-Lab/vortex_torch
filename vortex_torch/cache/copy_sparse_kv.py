"""
Sparse KV cache copy utilities for unified CPU/GPU memory management.

This module provides functions for copying sparse KV cache pages from CPU to GPU
with LRU eviction support, optimized for CUDA graph compatibility.
"""

import torch
from typing import Tuple


def allocate_pages_lru(
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
    use_hive_lockfree: bool = False  # Changed default: old mutex-based kernel
) -> None:
    """
    LRU allocation kernel (old mutex-based version) - allocates GPU slots for requested pages.

    Args:
        sparse_kv_indices: Page IDs to allocate [max_num_pages]
        sparse_kv_indptr: Indptr array [batch_size * num_kv_heads + 1]
        cpu_to_gpu_slot_map: Mapping from CPU page ID to GPU slot
        gpu_to_cpu_page_map: Reverse mapping from GPU slot to CPU page ID
        slot_ages: LRU ages for each GPU slot (uint8)
        slots_used_bitmap: Temporary bitmap tracking slots used in current round (bool)
        needs_eviction_bitmap: Temporary bitmap for pages needing eviction (bool)
        dst_gpu_slots: Output GPU slots for each requested page
        owners_bitmap: Output bitmap indicating if page needs copying
        evicted_cpu_pages: Output CPU page IDs that were evicted
        overflow_flag: Output flag indicating allocation overflow
        batch_size: Batch size
        num_kv_heads: Number of KV heads
        max_num_pages: Maximum number of pages (fixed for CUDA graph)
        max_hash_attempts: Maximum hash attempts for eviction
        use_hive_lockfree: Deprecated - use allocate_pages_hive() for the new kernel
    """
    import vortex_torch_C

    indptr_last_idx = batch_size * num_kv_heads

    # Only use the old mutex-based kernel
    vortex_torch_C.allocate_pages_lru_warp_with_indptr(
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


def allocate_pages_hive(
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    slot_stamps: torch.Tensor,        # uint32: per-slot LRU timestamps
    set_clock: torch.Tensor,          # uint32: per-set monotonic clock
    set_version: torch.Tensor,        # uint32: seqlock version per set
    set_used_mask: torch.Tensor,      # uint32: per-set used bitmask
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    overflow_flag: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int,
    max_hash_attempts: int = 30
) -> None:
    """
    Hive-style lock-free LRU allocation kernel with seqlock + timestamp LRU.

    This kernel uses:
    - Seqlock per set to ensure only one warp mutates a set at a time
    - Timestamp-based LRU (monotonic counter per set)
    - Per-set uint32 bitmask for tracking used slots this round
    - Warp-uniform control flow (no lane-local branches)

    Args:
        sparse_kv_indices: Page IDs to allocate [max_num_pages]
        sparse_kv_indptr: Indptr array [batch_size * num_kv_heads + 1]
        cpu_to_gpu_slot_map: Mapping from CPU page ID to GPU slot (int32)
        gpu_to_cpu_page_map: Reverse mapping from GPU slot to CPU page ID (int32)
        slot_stamps: Per-slot LRU timestamps (uint32, [num_slots])
        set_clock: Per-set monotonic clock (uint32, [num_sets])
        set_version: Seqlock version per set (uint32, [num_sets])
        set_used_mask: Per-set used bitmask (uint32, [num_sets])
        dst_gpu_slots: Output GPU slots for each requested page (int32)
        owners_bitmap: Output bitmap indicating if page needs copying (bool)
        evicted_cpu_pages: Output CPU page IDs that were evicted (int32)
        overflow_flag: Output flag indicating allocation overflow (int32)
        batch_size: Batch size
        num_kv_heads: Number of KV heads
        max_num_pages: Maximum number of pages (fixed for CUDA graph)
        max_hash_attempts: Maximum hash attempts for eviction
    """
    import vortex_torch_C

    indptr_last_idx = batch_size * num_kv_heads

    vortex_torch_C.allocate_pages_hive_lockfree(
        sparse_kv_indices,
        sparse_kv_indptr,
        indptr_last_idx,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        slot_stamps,
        set_clock,
        set_version,
        set_used_mask,
        dst_gpu_slots,
        owners_bitmap,
        evicted_cpu_pages,
        overflow_flag,
        max_num_pages,
        max_hash_attempts
    )


def init_hive_structures(
    slot_stamps: torch.Tensor,
    set_clock: torch.Tensor,
    set_version: torch.Tensor,
    num_slots: int,
    num_sets: int
) -> None:
    """
    Initialize Hive data structures. Call once at setup time.

    Args:
        slot_stamps: Per-slot timestamps to initialize (uint32, [num_slots])
        set_clock: Per-set clocks to initialize (uint32, [num_sets])
        set_version: Per-set seqlock versions to initialize (uint32, [num_sets])
        num_slots: Total number of GPU staging slots
        num_sets: Number of sets (num_slots / 32)
    """
    import vortex_torch_C

    vortex_torch_C.init_hive_structures(
        slot_stamps,
        set_clock,
        set_version,
        num_slots,
        num_sets
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
    Copy kernel - copies KV data from CPU to GPU based on allocation results.

    Assumes allocate_pages_lru() or allocate_pages_hive() has been called first to set up:
    - dst_gpu_slots: destination GPU slots
    - owners_bitmap: which pages need copying
    - evicted_cpu_pages: which pages need eviction (GPU->CPU)

    Args:
        cpu_k_buffer: CPU K cache buffer, pinned memory
        cpu_v_buffer: CPU V cache buffer, pinned memory
        gpu_k_buffer: GPU K cache buffer
        gpu_v_buffer: GPU V cache buffer
        sparse_kv_indices: Page IDs to copy [max_num_pages]
        sparse_kv_indptr: Indptr array [batch_size * num_kv_heads + 1]
        dst_gpu_slots: GPU slots for each page (from allocation)
        owners_bitmap: Bitmap indicating if page needs copying (from allocation)
        evicted_cpu_pages: CPU pages that were evicted (from allocation)
        page_size: Number of tokens per page
        batch_size: Batch size
        num_kv_heads: Number of KV heads
        max_num_pages: Maximum number of pages (fixed for CUDA graph)
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
