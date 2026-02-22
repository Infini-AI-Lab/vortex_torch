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
    src_scale,          # float32 scale buffer [num_pages, page_size, 1] flat
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
    scale = tl.load(src_scale + scale_offset)

    val_bf16 = (val_int8 * scale).to(tl.bfloat16)

    # Destination: page_idx * PAGE_SIZE * HEAD_DIM + token_idx * HEAD_DIM + dims
    dst_offset = (page_idx * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    tl.store(dst_bf16 + dst_offset, val_bf16, mask=mask_dim)


def dequant_paged_int8_to_bf16(
    src_int8: torch.Tensor,       # int8 [num_pages, page_size, head_dim]
    src_scale: torch.Tensor,      # float32 [num_pages, page_size, 1]
    page_indices: torch.Tensor,   # int32 [num_accessed_pages]
    page_size: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Dequantize only the accessed pages from int8 cache to a compact bf16 buffer.

    Returns:
        bf16 tensor of shape [num_accessed_pages, page_size, head_dim]
    """
    num_accessed_pages = page_indices.shape[0]
    if num_accessed_pages == 0:
        return torch.empty((0, page_size, head_dim), dtype=torch.bfloat16, device=src_int8.device)

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
