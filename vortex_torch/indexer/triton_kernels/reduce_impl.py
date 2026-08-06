import torch
import triton
import triton.language as tl
from ..context import Context
from typing import Literal
from ...utils import ReduceType
from .utils_impl import next_pow2

@triton.jit
def reduce_rr_kernel(
x,  # [*, D0, D1]
o,  # [*, 1 or D0, 1 or D1]
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,      # real extent of dim 0; may be any positive integer
x_D1: tl.constexpr,      # real extent of dim 1; may be any positive integer
x_D0_PAD: tl.constexpr,  # next_pow2(x_D0); tl.arange needs a power-of-two length
x_D1_PAD: tl.constexpr,  # next_pow2(x_D1)
DIM: tl.constexpr,
REDUCE_TYPE: tl.constexpr
):  
    
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    idx_ptr = tl.arange(0, max_chunk_size)
    dim0 = tl.arange(0, x_D0_PAD)
    dim1 = tl.arange(0, x_D1_PAD)
    dim0_mask = dim0 < x_D0
    dim1_mask = dim1 < x_D1
    
    for i in range(start, end):
        
        # Range of y rows for this workload
        x_len = tl.load(winfo_lens + i)
        x_off = tl.load(winfo_offsets + i)
        valid = idx_ptr < x_len
        x_i_ptr = x + x_off * x_D0 * x_D1 + \
                idx_ptr[:, None, None] * x_D0 * x_D1 + \
                dim0[None,:,None] * x_D1 + \
                dim1[None, None, :]

        # Lanes past the real extent are outside the tensor: mask them and
        # fill with the identity element of the reduction, so they cannot
        # affect the result. Mean divides by the real extent, not the padded
        # one, so 0.0 is the right filler there too.
        pad_val = -1e30 if REDUCE_TYPE == 1 else (1e30 if REDUCE_TYPE == 2 else 0.0)
        load_mask = valid[:, None, None] & dim0_mask[None, :, None] & dim1_mask[None, None, :]
        x_i = tl.load(x_i_ptr, mask=load_mask, other=pad_val).to(tl.float32)
        if DIM == 1:
            
            if REDUCE_TYPE == 0:
                x_i_reduce = tl.sum(x_i, axis=1) / x_D0
            
            elif REDUCE_TYPE == 1:
                x_i_reduce = tl.max(x_i, axis=1)
            
            elif REDUCE_TYPE ==  2:
                x_i_reduce = tl.min(x_i, axis=1)
                
            elif REDUCE_TYPE == 3:
                x_i_reduce = tl.sqrt(tl.sum(x_i * x_i, axis=1).to(tl.float32))
            
            elif REDUCE_TYPE == 4:
                x_i_reduce = tl.sum(x_i, axis=1)

            else:
                x_i_reduce = tl.zeros((max_chunk_size, x_D1_PAD), dtype=tl.bfloat16)
            
            x_i_reduce = x_i_reduce.to(tl.bfloat16)
            o_i_ptr = o + x_off * x_D1 + idx_ptr[:, None] * x_D1 + dim1[None,:]
            tl.store(o_i_ptr, x_i_reduce, mask=valid[:, None] & dim1_mask[None, :])
        
        elif DIM == 2:
            
            if REDUCE_TYPE == 0:
                x_i_reduce = tl.sum(x_i, axis=2) / x_D1
                
            elif REDUCE_TYPE == 1:
                x_i_reduce = tl.max(x_i, axis=2)
            
            elif REDUCE_TYPE == 2:
                x_i_reduce = tl.min(x_i, axis=2)
            
            elif REDUCE_TYPE == 3:
                x_i_reduce = tl.sqrt(tl.sum(x_i * x_i, axis=2).to(tl.float32))
            
            elif REDUCE_TYPE == 4:
                x_i_reduce = tl.sum(x_i, axis=2)

            else:
                x_i_reduce = tl.zeros((max_chunk_size, x_D0_PAD), dtype=tl.float32)

            x_i_reduce = x_i_reduce.to(tl.bfloat16)
            o_i_ptr = o + x_off * x_D0 + idx_ptr[:, None] * x_D0 + dim0[None,:]
            tl.store(o_i_ptr, x_i_reduce, mask=valid[:, None] & dim0_mask[None, :])




def reduce_rr(
x: torch.Tensor,
o: torch.Tensor,
dim: int,
reduce_type: ReduceType,
ctx: Context
):  
    
    reduce_rr_kernel[(8 * ctx.num_sms,)](
        x, o, 
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        x.shape[-1], 
        next_pow2(x.shape[-2]),
        next_pow2(x.shape[-1]),
        dim, 
        reduce_type.value,
        num_warps=4, 
        num_stages=1
    )


def _reduce_rr(
x: torch.Tensor,
o: torch.Tensor,
dim: int,
reduce_type: ReduceType,
winfo_kv_offsets: torch.Tensor,
winfo_kv_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: torch.Tensor,
num_sms: int,
):  
    
    reduce_rr_kernel[(8 * num_sms,)](
        x, o, 
        winfo_kv_offsets,
        winfo_kv_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        x.shape[-1], 
        next_pow2(x.shape[-2]),
        next_pow2(x.shape[-1]),
        dim, 
        reduce_type.value,
        num_warps=4, 
        num_stages=1
    )