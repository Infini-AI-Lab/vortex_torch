"""
OOM-safe bf16 fallback for int8 KV-cache prefill.

Instead of implementing full 2D-tiled Triton prefill with int8 dequantization,
this module dequantizes only the accessed KV pages into a compact temporary
bf16 buffer and remaps indices so FlashInfer can operate on the compact buffer.

This avoids dequantizing the entire global cache buffer.
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
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Dequantize selected int8 pages to bf16 compact buffer."""
    page_idx = tl.program_id(0)   # index into page_indices
    token_idx = tl.program_id(1)  # token within page [0, PAGE_SIZE)

    if page_idx >= NUM_PAGES:
        return

    global_page_id = tl.load(page_indices + page_idx)
    dims = tl.arange(0, BLOCK_DIM)
    mask_dim = dims < HEAD_DIM

    # Source: global_page_id * PAGE_SIZE * HEAD_DIM + token_idx * HEAD_DIM + dims
    src_offset = (global_page_id * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    val_int8 = tl.load(src_int8 + src_offset, mask=mask_dim, other=0).to(tl.float32)

    # Scale: global_page_id * PAGE_SIZE + token_idx
    scale_offset = global_page_id * PAGE_SIZE + token_idx
    scale = tl.load(src_scale + scale_offset).to(tl.float32)

    val_bf16 = (val_int8 * scale).to(tl.bfloat16)

    # Destination: page_idx * PAGE_SIZE * HEAD_DIM + token_idx * HEAD_DIM + dims
    dst_offset = (page_idx * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    tl.store(dst_bf16 + dst_offset, val_bf16, mask=mask_dim)


def dequant_paged_int8_to_bf16(
    src_int8: torch.Tensor,       # int8 [num_pages, page_size, head_dim]
    src_scale: torch.Tensor,      # fp16 [num_pages, page_size, 1]
    page_indices: torch.Tensor,   # int32 [num_accessed_pages]
    page_size: int,
    head_dim: int,
    out: torch.Tensor = None,     # optional pre-allocated bf16 [>=num_accessed_pages, page_size, head_dim]
) -> torch.Tensor:
    """
    Dequantize only the accessed pages from int8 cache to a compact bf16 buffer.

    If `out` is provided, writes into it (must have room for num_accessed_pages).
    Otherwise allocates a new buffer.

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

    grid = (num_accessed_pages, page_size)
    _dequant_pages_kernel[grid](
        src_int8,
        src_scale,
        dst_bf16,
        page_indices,
        NUM_PAGES=num_accessed_pages,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
    )

    return dst_bf16


@triton.jit
def _dequant_pages_inplace_kernel(
    src_int8,           # int8 paged buffer flat
    src_scale,          # scale buffer flat (one scale per token slot)
    dst_bf16,           # bf16 destination buffer (same page layout as src)
    page_indices,       # int32 [num_pages] — which global pages to dequant
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Dequantize selected int8 pages to bf16, writing to the SAME page positions in dst."""
    page_idx = tl.program_id(0)   # index into page_indices
    token_idx = tl.program_id(1)  # token within page [0, PAGE_SIZE)

    if page_idx >= NUM_PAGES:
        return

    global_page_id = tl.load(page_indices + page_idx)
    dims = tl.arange(0, BLOCK_DIM)
    mask_dim = dims < HEAD_DIM

    # Source and destination use the SAME offset (in-place layout)
    offset = (global_page_id * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    val_int8 = tl.load(src_int8 + offset, mask=mask_dim, other=0).to(tl.float32)

    scale_offset = global_page_id * PAGE_SIZE + token_idx
    scale = tl.load(src_scale + scale_offset).to(tl.float32)

    val_bf16 = (val_int8 * scale).to(tl.bfloat16)

    # Write to the SAME page position in dst (not compacted)
    tl.store(dst_bf16 + offset, val_bf16, mask=mask_dim)


def dequant_paged_int8_to_bf16_inplace(
    src_int8: torch.Tensor,       # int8 paged cache (flat)
    src_scale: torch.Tensor,      # fp16 scale buffer (flat)
    dst_bf16: torch.Tensor,       # bf16 destination (same shape as src_int8)
    page_indices: torch.Tensor,   # int32 [num_pages] — which pages to dequant
    page_size: int,
    head_dim: int,
) -> None:
    """
    Dequantize selected pages from int8 cache to bf16 IN-PLACE.

    Unlike dequant_paged_int8_to_bf16 (which compacts into a dense buffer),
    this writes to the SAME page positions in dst_bf16, preserving the paged layout.
    Used to populate the bf16 working buffer for forward_cache (centroid computation).
    """
    num_pages = page_indices.shape[0]
    if num_pages == 0:
        return

    BLOCK_DIM = triton.next_power_of_2(head_dim)

    grid = (num_pages, page_size)
    _dequant_pages_inplace_kernel[grid](
        src_int8,
        src_scale,
        dst_bf16,
        page_indices,
        NUM_PAGES=num_pages,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
    )
