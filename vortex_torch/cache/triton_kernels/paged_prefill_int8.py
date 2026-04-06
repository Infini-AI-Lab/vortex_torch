"""
OOM-safe bf16 fallback for int8 KV-cache prefill.

Instead of implementing full 2D-tiled Triton prefill with int8 dequantization,
this module dequantizes only the accessed KV pages into a compact temporary
bf16 buffer and remaps indices so FlashInfer can operate on the compact buffer.

This avoids dequantizing the entire global cache buffer.

Supports optional scale_page_map for CPU VTX: when int8 data page IDs differ
from scale page IDs (e.g., staging slots vs CPU flat page IDs).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_pages_kernel(
    src_int8,           # int8 paged buffer [num_pages, page_size, head_dim] flat
    src_scale,          # fp16 scale buffer [num_pages, page_size, 1] flat
    dst_bf16,           # bf16 compact buffer [num_accessed_pages, page_size, head_dim] flat
    page_indices,       # int32 [num_accessed_pages] — which global pages to dequant
    Scale_Page_Map,     # int32 [num_data_pages] → scale page ID, or nullptr
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    USE_SCALE_PAGE_MAP: tl.constexpr,
):
    """Dequantize selected int8 pages to bf16 compact buffer."""
    page_idx = tl.program_id(0)   # index into page_indices
    token_idx = tl.program_id(1)  # token within page [0, PAGE_SIZE)

    if page_idx >= NUM_PAGES:
        return

    # int64 to prevent overflow: page * PAGE_SIZE * HEAD_DIM can exceed INT32_MAX
    global_page_id = tl.load(page_indices + page_idx).to(tl.int64)
    dims = tl.arange(0, BLOCK_DIM).to(tl.int64)
    mask_dim = dims < HEAD_DIM

    # Source: global_page_id * PAGE_SIZE * HEAD_DIM + token_idx * HEAD_DIM + dims
    src_offset = (global_page_id * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    val_int8 = tl.load(src_int8 + src_offset, mask=mask_dim, other=0).to(tl.float32)

    # Scale: resolve page ID via optional indirection
    if USE_SCALE_PAGE_MAP:
        scale_page_id = tl.load(Scale_Page_Map + global_page_id).to(tl.int64)
    else:
        scale_page_id = global_page_id
    scale_offset = scale_page_id * PAGE_SIZE + token_idx
    scale = tl.load(src_scale + scale_offset).to(tl.float32)

    val_bf16 = (val_int8 * scale).to(tl.bfloat16)

    # Destination: page_idx is small (compact output), but use int64 for consistency
    page_idx_i64 = page_idx.to(tl.int64) if hasattr(page_idx, 'to') else tl.cast(page_idx, tl.int64)
    dst_offset = (page_idx_i64 * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    tl.store(dst_bf16 + dst_offset, val_bf16, mask=mask_dim)


def dequant_paged_int8_to_bf16(
    src_int8: torch.Tensor,       # int8 [num_pages, page_size, head_dim]
    src_scale: torch.Tensor,      # fp16 [num_pages, page_size, 1]
    page_indices: torch.Tensor,   # int32 [num_accessed_pages]
    page_size: int,
    head_dim: int,
    out: torch.Tensor = None,     # optional pre-allocated bf16 [>=num_accessed_pages, page_size, head_dim]
    scale_page_map: torch.Tensor = None,  # optional int32 [num_data_pages] → scale page ID
) -> torch.Tensor:
    """
    Dequantize only the accessed pages from int8 cache to a compact bf16 buffer.

    If `out` is provided, writes into it (must have room for num_accessed_pages).
    Otherwise allocates a new buffer.

    If `scale_page_map` is provided, scale lookups use indirection:
        scale_page = scale_page_map[data_page]
    This supports CPU VTX where int8 data is in staging (slot-indexed) but
    scales are in a persistent buffer (CPU-page-indexed).

    Returns:
        bf16 tensor of shape [num_accessed_pages, page_size, head_dim]
    """
    num_accessed_pages = page_indices.shape[0]
    if num_accessed_pages == 0:
        if out is not None:
            return out[:0]
        return torch.empty((0, page_size, head_dim), dtype=torch.bfloat16, device=src_int8.device)

    if out is not None:
        dst_bf16 = out[:num_accessed_pages]
    else:
        dst_bf16 = torch.empty(
            (num_accessed_pages, page_size, head_dim),
            dtype=torch.bfloat16,
            device=src_int8.device,
        )

    BLOCK_DIM = triton.next_power_of_2(head_dim)

    use_scale_page_map = scale_page_map is not None
    if scale_page_map is None:
        scale_page_map = page_indices  # dummy, won't be accessed

    grid = (num_accessed_pages, page_size)
    _dequant_pages_kernel[grid](
        src_int8,
        src_scale,
        dst_bf16,
        page_indices,
        scale_page_map,
        NUM_PAGES=num_accessed_pages,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
        USE_SCALE_PAGE_MAP=use_scale_page_map,
    )

    return dst_bf16


@triton.jit
def _dequant_pages_inplace_kernel(
    src_int8,           # int8 paged buffer flat
    src_scale,          # scale buffer flat (one scale per token slot)
    dst_bf16,           # bf16 destination buffer (same page layout as src)
    data_page_indices,  # int32 [num_pages] — data page IDs (index into src_int8)
    scale_page_indices, # int32 [num_pages] — scale page IDs (index into src_scale)
    dst_page_indices,   # int32 [num_pages] — destination page IDs (index into dst_bf16)
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Dequantize selected int8 pages to bf16, with separate page indices for
    data (src_int8), scales (src_scale), and destination (dst_bf16).

    This supports CPU VTX where:
    - data is in staging (staging-slot-indexed)
    - scales are in persistent buffer (CPU-page-indexed)
    - destination is _k_bf16_working (CPU-page-indexed for forward_cache)
    """
    page_idx = tl.program_id(0)   # index into page arrays
    token_idx = tl.program_id(1)  # token within page [0, PAGE_SIZE)

    if page_idx >= NUM_PAGES:
        return

    # int64 to prevent overflow: page * PAGE_SIZE * HEAD_DIM can exceed INT32_MAX
    data_page = tl.load(data_page_indices + page_idx).to(tl.int64)
    scale_page = tl.load(scale_page_indices + page_idx).to(tl.int64)
    dst_page = tl.load(dst_page_indices + page_idx).to(tl.int64)

    dims = tl.arange(0, BLOCK_DIM).to(tl.int64)
    mask_dim = dims < HEAD_DIM

    # Load int8 data at data page position
    src_offset = (data_page * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    val_int8 = tl.load(src_int8 + src_offset, mask=mask_dim, other=0).to(tl.float32)

    # Load scale at scale page position
    scale_offset = scale_page * PAGE_SIZE + token_idx
    scale = tl.load(src_scale + scale_offset).to(tl.float32)

    val_bf16 = (val_int8 * scale).to(tl.bfloat16)

    # Write to destination at dst page position
    dst_offset = (dst_page * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    tl.store(dst_bf16 + dst_offset, val_bf16, mask=mask_dim)


def dequant_paged_int8_to_bf16_inplace(
    src_int8: torch.Tensor,       # int8 paged cache (flat)
    src_scale: torch.Tensor,      # fp16 scale buffer (flat)
    dst_bf16: torch.Tensor,       # bf16 destination (same shape as src_int8)
    page_indices: torch.Tensor,   # int32 [num_pages] — which pages to dequant
    page_size: int,
    head_dim: int,
    scale_page_indices: torch.Tensor = None,  # optional separate scale page IDs
    dst_page_indices: torch.Tensor = None,    # optional separate destination page IDs
) -> None:
    """
    Dequantize selected pages from int8 cache to bf16 IN-PLACE.

    Unlike dequant_paged_int8_to_bf16 (which compacts into a dense buffer),
    this writes to the SAME page positions in dst_bf16, preserving the paged layout.
    Used to populate the bf16 working buffer for forward_cache (centroid computation).

    If scale_page_indices is provided, scales are looked up at different page positions
    than the int8 data. If dst_page_indices is provided, output is written to different
    page positions. This supports CPU VTX where data, scales, and destination use
    different page index spaces.
    """
    num_pages = page_indices.shape[0]
    if num_pages == 0:
        return

    if scale_page_indices is None:
        scale_page_indices = page_indices
    if dst_page_indices is None:
        dst_page_indices = page_indices

    BLOCK_DIM = triton.next_power_of_2(head_dim)

    grid = (num_pages, page_size)
    _dequant_pages_inplace_kernel[grid](
        src_int8,
        src_scale,
        dst_bf16,
        page_indices,
        scale_page_indices,
        dst_page_indices,
        NUM_PAGES=num_pages,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
    )
