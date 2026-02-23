import torch
import triton
import triton.language as tl
from ..context import Context
from ...utils import ReduceType


# ---------------------------------------------------------------------------
# Helper: Load a page block from src_ptr, handling bf16 or fp8-stored-as-uint8.
#   FP8_TYPE == 0  -> bf16 pointer, load normally
#   FP8_TYPE == 1  -> uint8 pointer, bitcast to float8e4nv, dequant with scale
#   FP8_TYPE == 2  -> uint8 pointer, bitcast to float8e5, dequant with scale
# All paths return a float32 tensor ready for reduction.
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
FP8_TYPE: tl.constexpr,     # 0: bf16, 1: e4m3, 2: e5m2
scale,                        # float: 1.0 for bf16, kv_scale for fp8
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

    if FP8_TYPE == 1:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif FP8_TYPE == 2:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_block = tl.load(src_ptr).to(tl.float32)

    if DIM == 1:
        # reduce over rows -> axis=0 -> length x_D1
        if REDUCE_TYPE == 0:       # Mean
            reduce_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:     # Max
            reduce_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:     # Min
            reduce_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        else:                      # L2Norm
            s = tl.sum(page_block * page_block, axis=0)
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
            s = tl.sum(page_block * page_block, axis=1)
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
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )


def _reduce_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )



@triton.jit
def reduce_rp_kernel(
    x, output, loc,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    REDUCE_TYPE: tl.constexpr,
    DIM: tl.constexpr,
    FP8_TYPE: tl.constexpr,
    scale,
):

    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    token_position = tl.load(loc + token_id)

    if (token_position + 1) % PAGE_SIZE != 0:
        return

    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
    x_offset = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1

    rows = tl.arange(0, x_D0)[:, None]
    cols = tl.arange(0, x_D1)[None, :]
    src_ptr = x + x_offset + rows * x_D1 + cols

    if FP8_TYPE == 1:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif FP8_TYPE == 2:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_block = tl.load(src_ptr).to(tl.float32)

    if DIM == 1:
        if REDUCE_TYPE == 0:
            reduce_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:
            reduce_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:
            reduce_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        else:
            s = tl.sum(page_block * page_block, axis=0)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        dst_ptr = output + page_id * x_D1 + tl.arange(0, x_D1)
        tl.store(dst_ptr, reduce_vec)

    else:
        if REDUCE_TYPE == 0:
            reduce_vec = (tl.sum(page_block, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:
            reduce_vec = tl.max(page_block, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:
            reduce_vec = tl.min(page_block, axis=1).to(tl.bfloat16)
        else:
            s = tl.sum(page_block * page_block, axis=1)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        dst_ptr = output + page_id * x_D0 + tl.arange(0, x_D0)
        tl.store(dst_ptr, reduce_vec)


def reduce_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int,
reduce_type: ReduceType,
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )


def _reduce_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )


@triton.jit
def reduce_pr_kernel(
x, output, loc,
x_D0: tl.constexpr,
x_D1: tl.constexpr,
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
REDUCE_TYPE: tl.constexpr,
DIM: tl.constexpr,
FP8_TYPE: tl.constexpr,
scale,
):

    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id

    x_offset = page_id * x_D0 * x_D1

    rows = tl.arange(0, x_D0)[:, None]
    cols = tl.arange(0, x_D1)[None, :]
    src_ptr = x + x_offset + rows * x_D1 + cols

    if FP8_TYPE == 1:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif FP8_TYPE == 2:
        raw = tl.load(src_ptr)
        page_block = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_block = tl.load(src_ptr).to(tl.float32)

    if DIM == 1:
        if REDUCE_TYPE == 0:
            reduce_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:
            reduce_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:
            reduce_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        else:
            s = tl.sum(page_block * page_block, axis=0)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

        out_base = (token_id * NUM_KV_HEAD + head_id) * x_D1
        dst_ptr  = output + out_base + tl.arange(0, x_D1)
        tl.store(dst_ptr, reduce_vec)

    else:
        if REDUCE_TYPE == 0:
            reduce_vec = (tl.sum(page_block, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:
            reduce_vec = tl.max(page_block, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:
            reduce_vec = tl.min(page_block, axis=1).to(tl.bfloat16)
        else:
            s = tl.sum(page_block * page_block, axis=1)
            reduce_vec = tl.sqrt(s).to(tl.bfloat16)

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
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )

def _reduce_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )


@triton.jit
def reduce_rr_kernel(
x, output, loc,
x_D0: tl.constexpr,
x_D1: tl.constexpr,
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
REDUCE_TYPE: tl.constexpr,
DIM: tl.constexpr,
FP8_TYPE: tl.constexpr,
scale,
):

    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    x_base   = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1
    rows     = tl.arange(0, x_D0)[:, None]
    cols     = tl.arange(0, x_D1)[None, :]
    src_ptr  = x + x_base + rows * x_D1 + cols

    if FP8_TYPE == 1:
        raw = tl.load(src_ptr)
        page_blk = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    elif FP8_TYPE == 2:
        raw = tl.load(src_ptr)
        page_blk = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * scale
    else:
        page_blk = tl.load(src_ptr).to(tl.float32)

    if DIM == 1:
        if REDUCE_TYPE == 0:
            vec = (tl.sum(page_blk, axis=0) / x_D0).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:
            vec = tl.max(page_blk, axis=0).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:
            vec = tl.min(page_blk, axis=0).to(tl.bfloat16)
        else:
            s = tl.sum(page_blk * page_blk, axis=0)
            vec = tl.sqrt(s).to(tl.bfloat16)

        out_base = (token_id * NUM_KV_HEAD + head_id) * x_D1
        tl.store(output + out_base + tl.arange(0, x_D1), vec)

    else:
        if REDUCE_TYPE == 0:
            vec = (tl.sum(page_blk, axis=1) / x_D1).to(tl.bfloat16)
        elif REDUCE_TYPE == 1:
            vec = tl.max(page_blk, axis=1).to(tl.bfloat16)
        elif REDUCE_TYPE == 2:
            vec = tl.min(page_blk, axis=1).to(tl.bfloat16)
        else:
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
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )


def _reduce_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
dim: int,
reduce_type: ReduceType,
fp8_type: int = 0,
scale: float = 1.0,
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
        FP8_TYPE=fp8_type,
        scale=scale,
    )
