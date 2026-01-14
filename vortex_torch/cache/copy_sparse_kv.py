"""
Sparse KV cache copy utilities for unified CPU/GPU memory management.

This module provides functions for copying sparse KV cache pages from CPU to GPU
with LRU eviction support, optimized for CUDA graph compatibility.
"""

import torch
from typing import Tuple


def copy_sparse_kv_to_gpu_with_indptr(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_buffer: torch.Tensor,
    gpu_v_buffer: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    slot_ages: torch.Tensor,
    dst_gpu_slots: torch.Tensor,
    owners_bitmap: torch.Tensor,
    slots_used_bitmap: torch.Tensor,
    needs_eviction_bitmap: torch.Tensor,
    evicted_cpu_pages: torch.Tensor,
    overflow_flag: torch.Tensor,
    page_size: int,
    batch_size: int,
    num_kv_heads: int,
    max_num_pages: int
) -> None:
    """
    Copy sparse KV cache pages from CPU to GPU with LRU eviction.

    This function is CUDA graph compatible - it reads the actual number of pages
    from sparse_kv_indptr on the GPU, avoiding CPU-GPU synchronization.

    The function performs two operations:
    1. Allocate GPU slots for requested pages using LRU eviction
    2. Copy KV data from CPU to GPU (and evict old pages if needed)

    Args:
        cpu_k_buffer: CPU K cache buffer, pinned memory. Layout flexible (head_dim is last dim)
        cpu_v_buffer: CPU V cache buffer, pinned memory. Layout flexible (head_dim is last dim)
        gpu_k_buffer: GPU K cache buffer
        gpu_v_buffer: GPU V cache buffer
        sparse_kv_indices: Page IDs to copy [max_num_pages], actual length in sparse_kv_indptr
        sparse_kv_indptr: Indptr array [batch_size * num_kv_heads + 1], last element = actual num_pages
        cpu_to_gpu_slot_map: Mapping from CPU page ID to GPU slot [-1 if not in GPU], [max_cpu_pages]
        gpu_to_cpu_page_map: Reverse mapping from GPU slot to CPU page ID, [num_gpu_slots]
        slot_ages: LRU ages for each GPU slot [num_gpu_slots], uint8
        dst_gpu_slots: Output GPU slots for each requested page [max_num_pages]
        owners_bitmap: Output bitmap indicating if page needs copying [max_num_pages]
        slots_used_bitmap: Temporary bitmap tracking slots used in current round [num_gpu_slots]
        needs_eviction_bitmap: Temporary bitmap for pages needing eviction [max_num_pages]
        evicted_cpu_pages: Output CPU page IDs that were evicted [max_num_pages], -1 if none
        overflow_flag: Output flag indicating allocation overflow [1], int32
        page_size: Number of tokens per page
        batch_size: Batch size
        num_kv_heads: Number of KV heads
        max_num_pages: Maximum number of pages (fixed for CUDA graph)

    Note:
        - CPU buffers must be pinned memory for fast GPU access
        - All tensors except CPU buffers must be on CUDA device
        - This function is designed for CUDA graph capture with fixed grid sizes
        - The actual number of pages to process is read from sparse_kv_indptr[-1] on GPU
    """
    import vortex_torch_C

    # Call the C++ extension directly (validation is done in C++ via TORCH_CHECK)
    vortex_torch_C.copy_sparse_kv_to_gpu_with_indptr(
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_buffer,
        gpu_v_buffer,
        sparse_kv_indices,
        sparse_kv_indptr,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        slot_ages,
        dst_gpu_slots,
        owners_bitmap,
        slots_used_bitmap,
        needs_eviction_bitmap,
        evicted_cpu_pages,
        overflow_flag,
        page_size,
        batch_size,
        num_kv_heads,
        max_num_pages
    )
