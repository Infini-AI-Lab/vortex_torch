import torch
import triton
import triton.language as tl
from ..context import Context
from ...utils import ReduceType


# ---------------------------------------------------------------------------
# Helper: Load a page block from src_ptr, handling bf16 / int8 / fp8-stored-as-uint8.
#   QUANT_TYPE == 0  -> bf16 pointer, load normally
#   QUANT_TYPE == 1  -> int8 pointer, dequant with per-row scale from kv_scale_ptr
#   QUANT_TYPE == 2  -> uint8 pointer, bitcast to float8e4nv, dequant with per-tensor scale
#   QUANT_TYPE == 3  -> uint8 pointer, bitcast to float8e5, dequant with per-tensor scale
# All quantised paths return a float32 tensor ready for reduction.
# ---------------------------------------------------------------------------


@triton.jit
def reduce_pp_kernel(
x, output, loc,
x_D0: tl.constexpr,
x_D1: tl.constexpr,
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
REDUCE_TYPE: tl.constexpr,  # 0:Mean, 1:Max, 2:Min, 3:L2Norm
DIM: tl.constexpr,           # 1: over rows (axis=0) -> len x_D1; 2: over cols (axis=1) -> len x_D0
QUANT_TYPE: tl.constexpr,   # 0: bf16, 1: int8, 2: e4m3, 3: e5m2
scale,                        # float: 1.0 for bf16, kv_scale for fp8
kv_scale_ptr,                 # pointer to per-token int8 scales (unused when QUANT_TYPE != 1)
):

    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    token_position = tl.load(loc + token_id)

    if (token_position + 1) % PAGE_SIZE != 0:
        return

    page_id  = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
    x_offset = page_id * x_D0 * x_D1

    rows = tl.arange(0, x_D0)[:, None]     # [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]     # [1, x_D1]
    src_ptr = x + x_offset + rows * x_D1 + cols

    if QUANT_TYPE == 1:
        # int8: load int8 values, dequant with per-row scale
        raw = tl.load(src_ptr).to(tl.float32)
        # Per-row scales stored at kv_scale_ptr[page_id * x_D0 + row]
        scale_offset = page_id * x_D0 + tl.arange(0, x_D0)
        row_scales = tl.load(kv_scale_ptr + scale_offset).to(tl.float32)  # [x_D0]
        page_block = raw * row_scales[:, None]  # broadcast [x_D0, 1]
    elif QUANT_TYPE == 2:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif QUANT_TYPE == 3:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_block = tl.load(src_ptr)

    if DIM == 1:
        # reduce over rows -> axis=0 -> length x_D1
        if REDUCE_TYPE == 0:       # Mean
            reduce_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:     # Max
            reduce_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:     # Min
            reduce_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        else:                      # L2Norm
            s = tl.sum(page_block * page_block, axis=0).to(tl.float32)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        dst_ptr = output + page_id * x_D1 + tl.arange(0, x_D1)
        tl.store(dst_ptr, reduce_vec)

    else:
        # DIM == 2: reduce over cols -> axis=1 -> length x_D0
        if REDUCE_TYPE == 0:       # Mean
            reduce_vec = (tl.sum(page_block, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:     # Max
            reduce_vec = tl.max(page_block, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:     # Min
            reduce_vec = tl.min(page_block, axis=1).to(tl.bfloat16)
        else:                      # L2Norm
            s = tl.sum(page_block * page_block, axis=1).to(tl.float32)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        dst_ptr = output + page_id * x_D0 + tl.arange(0, x_D0)
        tl.store(dst_ptr, reduce_vec)




def reduce_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num

    reduce_pp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )


def _reduce_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads

    reduce_pp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )



@triton.jit
def reduce_rp_kernel(
    x, output, loc,
    x_D0: tl.constexpr,              # rows per token-page
    x_D1: tl.constexpr,              # cols per token-page
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    REDUCE_TYPE: tl.constexpr,       # 0: Mean, 1: Max, 2: Min, 3: L2Norm (not RMS)
    DIM: tl.constexpr,               # 1: reduce over rows -> len x_D1; 2: reduce over cols -> len x_D0
    QUANT_TYPE: tl.constexpr,        # 0: bf16, 1: int8, 2: e4m3, 3: e5m2
    scale,                            # float: 1.0 for bf16, kv_scale for fp8
    kv_scale_ptr,                     # pointer to per-token int8 scales (unused when QUANT_TYPE != 1)
):

    # Program IDs:
    #   pid0 = token index (0 .. num_tokens-1)
    #   pid1 = head  index (0 .. NUM_KV_HEAD-1)
    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    # Load the absolute position of this token (used to map to page index).
    token_position = tl.load(loc + token_id)

    # Only the last token of a page triggers the reduction.
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # Output page index:
    #   Logical page = token_position // PAGE_SIZE
    #   One vector per head, so linearize by NUM_KV_HEAD.
    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id

    # Input layout is [num_tokens, num_heads, x_D0, x_D1] (row-major).
    #   For this token/head, compute the base element offset in `x`.
    x_offset = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1

    # Build 2D indices within a page (row-major addressing).
    rows = tl.arange(0, x_D0)[:, None]      # shape [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]      # shape [1, x_D1]
    src_ptr = x + x_offset + rows * x_D1 + cols

    # Load the full page block for this (token_id, head_id).
    # Assumes the page is full; add masks here if you have partial tiles.
    if QUANT_TYPE == 1:
        # int8: load int8 values, dequant with per-row scale
        raw = tl.load(src_ptr).to(tl.float32)
        scale_offset = page_id * x_D0 + tl.arange(0, x_D0)
        row_scales = tl.load(kv_scale_ptr + scale_offset).to(tl.float32)
        page_block = raw * row_scales[:, None]
    elif QUANT_TYPE == 2:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif QUANT_TYPE == 3:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_block = tl.load(src_ptr)

    # Reduction:
    if DIM == 1:
        # Reduce over rows (axis=0) -> output vector length x_D1 (per-column reduce).
        if REDUCE_TYPE == 0:  # Mean
            # NOTE: precision-sensitive workloads may want fp32 accumulation:
            # s = tl.sum(page_block.to(tl.float32), axis=0)
            # reduce_vec = (s / x_D0).to(tl.bfloat16)
            reduce_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:  # Max
            reduce_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:  # Min
            reduce_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        else:                   # L2Norm (sqrt(sum(x*x))); NOT RMS
            # For RMS, use: tl.sqrt(tl.sum(page_block*page_block, axis=0) / x_D0)
            s = tl.sum(page_block * page_block, axis=0).to(tl.float32)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        # Write to output: layout [num_pages, x_D1] for DIM==1.
        dst_ptr = output + page_id * x_D1 + tl.arange(0, x_D1)
        tl.store(dst_ptr, reduce_vec)

    else:
        # DIM == 2: Reduce over cols (axis=1) -> output vector length x_D0 (per-row reduce).
        if REDUCE_TYPE == 0:  # Mean
            # s = tl.sum(page_block.to(tl.float32), axis=1)
            # reduce_vec = (s / x_D1).to(tl.bfloat16)
            reduce_vec = (tl.sum(page_block, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:  # Max
            reduce_vec = tl.max(page_block, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:  # Min
            reduce_vec = tl.min(page_block, axis=1).to(tl.bfloat16)
        else:                   # L2Norm (sqrt(sum(x*x))); NOT RMS
            s = tl.sum(page_block * page_block, axis=1).to(tl.float32)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        # Write to output: layout [num_pages, x_D0] for DIM==2.
        dst_ptr = output + page_id * x_D0 + tl.arange(0, x_D0)
        tl.store(dst_ptr, reduce_vec)



def reduce_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num

    reduce_rp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )


def _reduce_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads

    reduce_rp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )


@triton.jit
def reduce_pr_kernel(
x, output, loc,
x_D0: tl.constexpr,              # rows per page
x_D1: tl.constexpr,              # cols per page
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
REDUCE_TYPE: tl.constexpr,       # 0: Mean, 1: Max, 2: Min, 3: L2Norm (not RMS)
DIM: tl.constexpr,               # 1: reduce over rows -> len x_D1; 2: reduce over cols -> len x_D0
QUANT_TYPE: tl.constexpr,        # 0: bf16, 1: int8, 2: e4m3, 3: e5m2
scale,                            # float: 1.0 for bf16, kv_scale for fp8
kv_scale_ptr,                     # pointer to per-token int8 scales (unused when QUANT_TYPE != 1)
):
    """
    Layouts:
      x:      [num_pages * NUM_KV_HEAD, x_D0, x_D1]   (page-major, row-major inside page)
      output: [num_tokens * NUM_KV_HEAD, vec_len]     (token-major; vec_len = x_D1 if DIM==1 else x_D0)

    Behavior:
      - token_id comes from pid0; head_id comes from pid1.
      - Read loc[token_id] to get absolute position; only proceed at page end.
      - Map token -> page via page_idx = (token_position // PAGE_SIZE).
      - Read the whole page for this (page_idx, head_id), do reduction,
        then write a single vector to output at (token_id, head_id, :).
    """

    # --- Program IDs ---
    token_id = tl.program_id(0)             # [0 .. num_tokens-1]
    head_id  = tl.program_id(1)             # [0 .. NUM_KV_HEAD-1]

    # --- Trigger only at end-of-page token ---
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # --- Page indexing for x (page-major) ---
    # page linear id across heads
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id

    # Base element offset into x for this (page_id, head_id)
    # x is laid out as contiguous pages, each page is [x_D0, x_D1]
    x_offset = page_id * x_D0 * x_D1

    # 2D row-major addressing within the page
    rows = tl.arange(0, x_D0)[:, None]      # [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]      # [1, x_D1]
    src_ptr = x + x_offset + rows * x_D1 + cols

    # Load the full page block. Assumes full tiles; add masks if needed.
    if QUANT_TYPE == 1:
        # int8: load int8 values, dequant with per-row scale
        raw = tl.load(src_ptr).to(tl.float32)
        scale_offset = page_id * x_D0 + tl.arange(0, x_D0)
        row_scales = tl.load(kv_scale_ptr + scale_offset).to(tl.float32)
        page_block = raw * row_scales[:, None]
    elif QUANT_TYPE == 2:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif QUANT_TYPE == 3:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_block = tl.load(src_ptr)

    # --- Reduction & write-out ---
    if DIM == 1:
        # Reduce over rows (axis=0) -> per-column vector, length = x_D1
        if REDUCE_TYPE == 0:        # Mean
            # For better accuracy you may upcast: tl.sum(page_block.to(tl.float32), axis=0)
            reduce_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:      # Max
            reduce_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:      # Min
            reduce_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        else:                       # L2Norm (NOT RMS)
            s = tl.sum(page_block * page_block, axis=0).to(tl.float32)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        # output is token-major: [num_tokens, NUM_KV_HEAD, x_D1]
        out_base = (token_id * NUM_KV_HEAD + head_id) * x_D1
        dst_ptr  = output + out_base + tl.arange(0, x_D1)
        tl.store(dst_ptr, reduce_vec)

    else:
        # DIM == 2: Reduce over cols (axis=1) -> per-row vector, length = x_D0
        if REDUCE_TYPE == 0:        # Mean
            reduce_vec = (tl.sum(page_block, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:      # Max
            reduce_vec = tl.max(page_block, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:      # Min
            reduce_vec = tl.min(page_block, axis=1).to(tl.bfloat16)
        else:                       # L2Norm (NOT RMS)
            s = tl.sum(page_block * page_block, axis=1).to(tl.float32)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)


        # output is token-major: [num_tokens, NUM_KV_HEAD, x_D0]
        out_base = (token_id * NUM_KV_HEAD + head_id) * x_D0
        dst_ptr  = output + out_base + tl.arange(0, x_D0)
        tl.store(dst_ptr, reduce_vec)


def reduce_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num

    reduce_pr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )

def _reduce_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads

    reduce_pr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )


@triton.jit
def reduce_rr_kernel(
x, output, loc,
x_D0: tl.constexpr,              # rows per token-page
x_D1: tl.constexpr,              # cols per token-page
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
REDUCE_TYPE: tl.constexpr,       # 0: Mean, 1: Max, 2: Min, 3: L2Norm (not RMS)
DIM: tl.constexpr,               # 1: reduce over rows -> len x_D1; 2: reduce over cols -> len x_D0
QUANT_TYPE: tl.constexpr,        # 0: bf16, 1: int8, 2: e4m3, 3: e5m2
scale,                            # float: 1.0 for bf16, kv_scale for fp8
kv_scale_ptr,                     # pointer to per-token int8 scales (unused when QUANT_TYPE != 1)
):
    """
    Layouts:
      x:      [num_tokens * NUM_KV_HEAD, x_D0, x_D1]     (token-major)
      output: [num_tokens * NUM_KV_HEAD, vec_len]        (token-major; vec_len = x_D1 if DIM==1 else x_D0)

    Only the last token of each page performs the reduction and writes to output[token_id, head_id, :].
    """


    # program ids
    token_id = tl.program_id(0)   # 0..num_tokens-1
    head_id  = tl.program_id(1)   # 0..NUM_KV_HEAD-1

    # trigger only at end-of-page token
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # ---- read from x (token-major) ----
    x_base   = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1
    rows     = tl.arange(0, x_D0)[:, None]         # [x_D0, 1]
    cols     = tl.arange(0, x_D1)[None, :]         # [1, x_D1]
    src_ptr  = x + x_base + rows * x_D1 + cols

    if QUANT_TYPE == 1:
        # int8: load int8 values, dequant with per-row scale
        raw = tl.load(src_ptr).to(tl.float32)
        page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
        scale_offset = page_id * x_D0 + tl.arange(0, x_D0)
        row_scales = tl.load(kv_scale_ptr + scale_offset).to(tl.float32)
        page_blk = raw * row_scales[:, None]
    elif QUANT_TYPE == 2:
        raw = tl.load(src_ptr)
        page_blk = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif QUANT_TYPE == 3:
        raw = tl.load(src_ptr)
        page_blk = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_blk = tl.load(src_ptr)                    # assumes full page; add masks if needed

    # ---- reduce ----
    if DIM == 1:
        # over rows -> axis=0 -> vector len x_D1
        if REDUCE_TYPE == 0:       # Mean
            # For better accuracy you may upcast to fp32 before sum.
            vec = (tl.sum(page_blk, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:     # Max
            vec = tl.max(page_blk, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:     # Min
            vec = tl.min(page_blk, axis=0).to(tl.bfloat16)
        else:                      # L2Norm (NOT RMS)
            s = tl.sum(page_blk * page_blk, axis=0)
            vec = tl.sqrt(s).to(tl.bfloat16)

        # ---- write to output (token-major) ----
        out_base = (token_id * NUM_KV_HEAD + head_id) * x_D1
        tl.store(output + out_base + tl.arange(0, x_D1), vec)

    else:
        # DIM == 2: over cols -> axis=1 -> vector len x_D0
        if REDUCE_TYPE == 0:       # Mean
            vec = (tl.sum(page_blk, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:     # Max
            vec = tl.max(page_blk, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:     # Min
            vec = tl.min(page_blk, axis=1).to(tl.bfloat16)
        else:                      # L2Norm (NOT RMS)
            s = tl.sum(page_blk * page_blk, axis=1)
            vec = tl.sqrt(s).to(tl.bfloat16)

        out_base = (token_id * NUM_KV_HEAD + head_id) * x_D0
        tl.store(output + out_base + tl.arange(0, x_D0), vec)



def reduce_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num

    reduce_rr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )


def _reduce_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
quant_type: int = 0,
scale: float = 1.0,
kv_scale_ptr=None,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads

    reduce_rr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        REDUCE_TYPE=reduce_type.value,
        DIM=dim,
        QUANT_TYPE=quant_type,
        scale=scale,
        kv_scale_ptr=kv_scale_ptr if kv_scale_ptr is not None else x,
    )
