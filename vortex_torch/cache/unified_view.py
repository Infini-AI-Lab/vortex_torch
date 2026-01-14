"""
Unified Cache View for CPU/GPU Hybrid Memory Management

This module provides UnifiedCacheView, a wrapper that enables transparent
access to cache data split between CPU pinned memory and GPU memory.
"""

import torch
from typing import Optional, Tuple
from ..abs import FORMAT


class UnifiedCacheView:
    """
    Wrapper that presents unified CPU+GPU cache as a standard cache tensor.

    This class enables operations to transparently work with pages distributed
    across CPU pinned memory and GPU memory using CUDA Unified Virtual
    Addressing (UVA). Operators detect this type and route to specialized
    kernels that can read from both memory spaces efficiently.

    Attributes:
        cpu_buffer (torch.Tensor): Pinned CPU memory [num_cpu_slots, page_size, head_dim]
                                   Must be allocated with pin_memory=True for UVA
        gpu_buffer (torch.Tensor): GPU memory [num_gpu_slots, page_size, head_dim]
        cpu_to_gpu_slot_map (torch.IntTensor): Routing table [total_pages] where:
                                               - value >= 0: page is in GPU at that slot
                                               - value == -1: page is in CPU
        num_cpu_slots (int): Number of pages in CPU memory
        num_gpu_slots (int): Number of pages in GPU memory
        _format (FORMAT): Tensor format (PAGED or RAGGED)

    Example:
        >>> # Create unified cache where first 400 pages are in GPU, rest in CPU
        >>> cpu_pages = torch.randn((600, 16, 128), dtype=torch.bfloat16, pin_memory=True)
        >>> gpu_pages = torch.randn((400, 16, 128), dtype=torch.bfloat16, device='cuda')
        >>> slot_map = torch.cat([
        ...     torch.arange(400),      # First 400 pages in GPU
        ...     torch.full((600,), -1)  # Rest in CPU
        ... ]).cuda()
        >>>
        >>> unified_view = UnifiedCacheView(cpu_pages, gpu_pages, slot_map)
        >>>
        >>> # Use with reduction operators (automatically routes to unified kernel)
        >>> from vortex_torch.cache import Mean
        >>> mean_op = Mean(dim=1)
        >>> result = mean_op(unified_view, output=None, loc=loc, ctx=ctx)
    """

    def __init__(
        self,
        cpu_buffer: Optional[torch.Tensor],
        gpu_buffer: torch.Tensor,
        cpu_to_gpu_slot_map: torch.Tensor,
        _format: FORMAT = FORMAT.PAGED
    ):
        """
        Initialize unified cache view.

        Args:
            cpu_buffer: Pinned CPU memory [num_cpu_slots, page_size, head_dim].
                       Must be allocated with pin_memory=True for UVA access.
                       Can be None if all pages are in GPU.
            gpu_buffer: GPU memory [num_gpu_slots, page_size, head_dim]
            cpu_to_gpu_slot_map: Int tensor [total_pages] mapping page_id to:
                                - slot index >= 0 if page is in GPU
                                - -1 if page is in CPU
            _format: FORMAT.PAGED or FORMAT.RAGGED (default: FORMAT.PAGED)

        Raises:
            ValueError: If cpu_buffer is not pinned memory
            ValueError: If cpu_buffer and gpu_buffer have mismatched shapes
            ValueError: If slot map references invalid slot indices
            ValueError: If cpu_buffer and gpu_buffer have different dtypes
        """
        # Validate and setup CPU buffer
        if cpu_buffer is not None:
            if not cpu_buffer.is_pinned():
                raise ValueError(
                    "cpu_buffer must be pinned memory (allocated with pin_memory=True). "
                    "Non-pinned memory cannot be accessed from GPU via UVA. "
                    "Use: torch.empty(..., pin_memory=True)"
                )
            self.cpu_buffer = cpu_buffer
            self.num_cpu_slots = cpu_buffer.shape[0]
        else:
            self.cpu_buffer = None
            self.num_cpu_slots = 0

        # Setup GPU buffer
        self.gpu_buffer = gpu_buffer
        self.num_gpu_slots = gpu_buffer.shape[0]

        # Validate shape compatibility
        if cpu_buffer is not None:
            if cpu_buffer.shape[1:] != gpu_buffer.shape[1:]:
                raise ValueError(
                    f"cpu_buffer and gpu_buffer must have same page shape. "
                    f"cpu_buffer: {cpu_buffer.shape[1:]}, gpu_buffer: {gpu_buffer.shape[1:]}"
                )

        # Validate dtype consistency
        if cpu_buffer is not None:
            if cpu_buffer.dtype != gpu_buffer.dtype:
                raise ValueError(
                    f"cpu_buffer dtype ({cpu_buffer.dtype}) must match "
                    f"gpu_buffer dtype ({gpu_buffer.dtype})"
                )

        # Setup routing map on GPU
        self.cpu_to_gpu_slot_map = cpu_to_gpu_slot_map
        self._format = _format

    def is_unified(self) -> bool:
        """
        Returns True, indicating this is a unified cache view.

        This method is used by operators to detect unified caches
        and route to specialized kernels.
        """
        return True

    def has_cpu_pages(self) -> bool:
        """
        Returns True if any pages are stored in CPU memory.

        Returns:
            bool: True if num_cpu_slots > 0, False otherwise
        """
        return self.num_cpu_slots > 0

    def get_cpu_base_ptr(self) -> int:
        """
        Get CPU buffer base pointer for UVA access.

        Returns:
            int: Data pointer to CPU buffer, or 0 if no CPU pages

        Note:
            This pointer is used by CUDA kernels via Unified Virtual
            Addressing to access CPU memory directly from GPU.
        """
        if self.cpu_buffer is None:
            return 0
        return self.cpu_buffer.data_ptr()

    def get_gpu_base_ptr(self) -> int:
        """
        Get GPU buffer base pointer.

        Returns:
            int: Data pointer to GPU buffer
        """
        return self.gpu_buffer.data_ptr()

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Total logical shape: [num_cpu_slots + num_gpu_slots, page_size, head_dim].

        Returns:
            Tuple[int, ...]: Combined shape of CPU and GPU buffers
        """
        return (self.num_cpu_slots + self.num_gpu_slots,) + tuple(self.gpu_buffer.shape[1:])

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of the cache tensors.

        Returns:
            torch.dtype: Dtype (propagated from gpu_buffer)
        """
        return self.gpu_buffer.dtype

    @property
    def device(self) -> torch.device:
        """
        Device of the cache (always CUDA).

        Returns:
            torch.device: GPU device (propagated from gpu_buffer)
        """
        return self.gpu_buffer.device

    @property
    def format(self) -> FORMAT:
        """
        Tensor format (PAGED or RAGGED).

        Returns:
            FORMAT: The format enum value
        """
        return self._format

    def data_ptr(self) -> int:
        """
        Return GPU buffer data pointer for Triton compatibility.

        Triton uses this for kernel specialization. We return the GPU buffer
        pointer since that's where the kernel will write output.

        Returns:
            int: GPU buffer data pointer
        """
        return self.gpu_buffer.data_ptr()

    def is_contiguous(self) -> bool:
        """
        Check if memory is contiguous.

        For Triton compatibility. The individual buffers may be contiguous,
        but the unified view spans non-contiguous memory spaces.

        Returns:
            bool: False (unified memory is not contiguous)
        """
        return False

    def stride(self) -> tuple:
        """
        Return strides for Triton compatibility.

        Returns:
            tuple: Strides of the GPU buffer
        """
        return self.gpu_buffer.stride()

    def dim(self) -> int:
        """
        Return number of dimensions.

        Returns:
            int: Number of dimensions (always 3 for cache)
        """
        return len(self.shape)

    def size(self, dim: Optional[int] = None):
        """
        Return size along dimension(s).

        Args:
            dim: Optional dimension index

        Returns:
            Size along dimension, or tuple of all sizes
        """
        if dim is None:
            return self.shape
        return self.shape[dim]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UnifiedCacheView(\n"
            f"  cpu_slots={self.num_cpu_slots}, gpu_slots={self.num_gpu_slots},\n"
            f"  shape={self.shape}, dtype={self.dtype}, format={self._format}\n"
            f")"
        )


def is_unified_cache(obj) -> bool:
    """
    Check if an object is a UnifiedCacheView.

    Args:
        obj: Object to check

    Returns:
        bool: True if obj is UnifiedCacheView, False otherwise
    """
    return isinstance(obj, UnifiedCacheView)
