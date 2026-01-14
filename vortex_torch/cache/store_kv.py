"""
Unified KV storage for CPU/GPU hybrid memory management.

This module provides a high-level interface for storing KV cache to
unified CPU+GPU memory using CUDA Unified Virtual Addressing (UVA).
"""

import torch
from typing import Optional


def store_kv_unified(
    cpu_k_buffer: Optional[torch.Tensor],
    cpu_v_buffer: Optional[torch.Tensor],
    gpu_k_buffer: torch.Tensor,
    gpu_v_buffer: torch.Tensor,
    cache_k_input: torch.Tensor,
    cache_v_input: torch.Tensor,
    loc: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    page_size: int
) -> None:
    """
    Store KV cache to unified CPU/GPU memory.

    This function routes each token's KV data to either CPU pinned memory
    or GPU memory based on the slot mapping, enabling flexible page placement
    for memory management.

    Args:
        cpu_k_buffer: Pinned CPU memory for K cache [num_cpu_slots, page_size, head_dim].
                     Can be None or empty if all pages are in GPU.
        cpu_v_buffer: Pinned CPU memory for V cache [num_cpu_slots, page_size, head_dim].
                     Can be None or empty if all pages are in GPU.
        gpu_k_buffer: GPU memory for K cache [num_gpu_slots, page_size, head_dim]
        gpu_v_buffer: GPU memory for V cache [num_gpu_slots, page_size, head_dim]
        cache_k_input: Input K cache to store [num_tokens, num_heads, head_dim]
        cache_v_input: Input V cache to store [num_tokens, num_heads, head_dim]
        loc: Token positions [num_tokens], determines which page and offset
        cpu_to_gpu_slot_map: Routing table [total_pages] where:
                            - value >= 0: page is in CPU at that slot
                            - value == -1: page is in GPU (slot = page_id - num_cpu_slots)
        page_size: Number of tokens per page

    Example:
        >>> # Setup unified buffers
        >>> cpu_k = torch.empty((160, 16, 128), dtype=torch.bfloat16, pin_memory=True)
        >>> cpu_v = torch.empty((160, 16, 128), dtype=torch.bfloat16, pin_memory=True)
        >>> gpu_k = torch.empty((240, 16, 128), dtype=torch.bfloat16, device='cuda')
        >>> gpu_v = torch.empty((240, 16, 128), dtype=torch.bfloat16, device='cuda')
        >>>
        >>> # Slot map: first 160 pages in CPU, rest in GPU
        >>> slot_map = torch.cat([
        ...     torch.arange(160, dtype=torch.int32),
        ...     torch.full((240,), -1, dtype=torch.int32)
        ... ]).cuda()
        >>>
        >>> # Store new tokens
        >>> k_new = torch.randn((32, 8, 128), dtype=torch.bfloat16, device='cuda')
        >>> v_new = torch.randn((32, 8, 128), dtype=torch.bfloat16, device='cuda')
        >>> positions = torch.arange(32, dtype=torch.int64, device='cuda')
        >>>
        >>> store_kv_unified(cpu_k, cpu_v, gpu_k, gpu_v, k_new, v_new,
        ...                  positions, slot_map, page_size=16)

    Notes:
        - CPU buffers must be pinned memory (allocated with pin_memory=True)
        - The slot map size must match total_pages = num_tokens * num_heads
        - Uses CUDA Unified Virtual Addressing for efficient CPU memory access
        - No explicit synchronization needed - PyTorch manages stream ordering
    """
    import vortex_torch_C

    # Validate inputs
    assert cache_k_input.dtype == torch.bfloat16, "cache_k must be bfloat16"
    assert cache_v_input.dtype == torch.bfloat16, "cache_v must be bfloat16"
    assert loc.dtype == torch.int64, "loc must be int64"
    assert cpu_to_gpu_slot_map.dtype == torch.int32, "slot_map must be int32"

    # Handle empty CPU buffers
    if cpu_k_buffer is None or cpu_k_buffer.numel() == 0:
        cpu_k_buffer = torch.empty(0, dtype=torch.bfloat16)
    if cpu_v_buffer is None or cpu_v_buffer.numel() == 0:
        cpu_v_buffer = torch.empty(0, dtype=torch.bfloat16)

    # Validate CPU buffers are pinned if non-empty
    if cpu_k_buffer.numel() > 0:
        assert cpu_k_buffer.is_pinned(), (
            "cpu_k_buffer must be pinned memory for UVA access. "
            "Use: torch.empty(..., pin_memory=True)"
        )
    if cpu_v_buffer.numel() > 0:
        assert cpu_v_buffer.is_pinned(), (
            "cpu_v_buffer must be pinned memory for UVA access. "
            "Use: torch.empty(..., pin_memory=True)"
        )

    # Call C++ kernel
    vortex_torch_C.store_kv_unified(
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_buffer,
        gpu_v_buffer,
        cache_k_input,
        cache_v_input,
        loc,
        cpu_to_gpu_slot_map,
        page_size
    )
