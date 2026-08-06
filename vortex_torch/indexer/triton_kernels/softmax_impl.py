import torch
import triton
import triton.language as tl
from ..context import Context
from .utils_impl import next_pow2

@triton.jit
def softmax_inplace_r_kernel(
x, 
indptr,
scale: tl.constexpr,
bos: tl.constexpr,
eos: tl.constexpr,
topk_val: tl.constexpr,
x_D0: tl.constexpr,
x_D1: tl.constexpr,      # real group size; may be any positive integer
x_D1_PAD: tl.constexpr,  # next_pow2(x_D1); tl.arange needs a power-of-two length
BLOCK_P: tl.constexpr = 256,
):
    pid = tl.program_id(0)

    start = tl.load(indptr + pid)
    end   = tl.load(indptr + pid + 1)
    num_pages_this_seq = end - start

    threshold: tl.constexpr = bos + eos + topk_val
    if num_pages_this_seq <= threshold:
        return

    num_pages_to_compute = num_pages_this_seq - bos - eos
    if num_pages_to_compute <= 0:
        return

    base_ptr = x + (start + bos) * (x_D0 * x_D1)

    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1_PAD)
    d1_mask = d1_idx < x_D1
    p_idx  = tl.arange(0, BLOCK_P)

    # --- One-pass accumulation of (m, s) ---
    neg_inf = -1e30
    m = tl.full((x_D0, x_D1_PAD), neg_inf, dtype=tl.float32)
    s = tl.zeros((x_D0, x_D1_PAD), dtype=tl.float32)

    for p in range(0, num_pages_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_pages_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * (x_D0 * x_D1)
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        mask = p_mask[:, None, None] & d1_mask[None, None, :]
        slab = tl.load(base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)
        slab = slab * scale
        mc = tl.max(slab, axis=0)
        sc = tl.sum(tl.exp(slab - mc[None, :, :]), axis=0)

        m_new = tl.maximum(m, mc)
        s = s * tl.exp(m - m_new) + sc * tl.exp(mc - m_new)
        m = m_new

    # --- Normalize and store ---
    for p in range(0, num_pages_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_pages_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * (x_D0 * x_D1)
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        mask = p_mask[:, None, None] & d1_mask[None, None, :]
        slab = tl.load(base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)
        slab = slab * scale
        slab = tl.exp(slab - m[None, :, :]) / s[None, :, :]
        slab = slab.to(tl.bfloat16)
        tl.store(base_ptr + offs, slab, mask=mask)

    


def softmax_inplace_r(
x: torch.Tensor,
dim: int,
scale: float,
ctx: Context
):  
    
    eff_batch_size = ctx.batch_size * ctx.num_kv_heads
    
    
    softmax_inplace_r_kernel[(eff_batch_size,)](
        x,
        ctx.dense_kv_indptr,
        scale,
        ctx.page_reserved_bos,
        ctx.page_reserved_eos,
        ctx.topk_val,
        x.shape[-2], 
        x.shape[-1], 
        next_pow2(x.shape[-1]),
        num_warps=4, 
        num_stages=1
    )
    

def _softmax_inplace_r(
x: torch.Tensor,
dim: int,
indptr: torch.Tensor,
scale: float,
page_reserved_bos: int, 
page_reserved_eos: int,
topk_val: int,
batch_size: int
):  
    
    
    softmax_inplace_r_kernel[(batch_size,)](
        x,
        indptr,
        scale,
        page_reserved_bos,
        page_reserved_eos,
        topk_val,
        x.shape[-2], 
        x.shape[-1], 
        next_pow2(x.shape[-1]),
        num_warps=4, 
        num_stages=1
    )