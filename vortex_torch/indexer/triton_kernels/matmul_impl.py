import torch
import triton
import triton.language as tl
from ..context import Context
from .utils_impl import next_pow2

@triton.jit
def mm_bpr_kernel(
x,  # bf16, [B, G, D] ; D is the fastest-changing dim
y,  # bf16, [S, C, D] ; D is the fastest-changing dim
o,  # bf16, [*, C, G] ; flat buffer (each row writes a [C, G] tile)
indices,             # int32
winfo_x_indices,     # int32
winfo_y_offsets,     # int32
winfo_y_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
G: tl.constexpr,      # real group size; may be any positive integer
G_PAD: tl.constexpr,  # next_pow2(G); tl.arange needs a power-of-two length
C: tl.constexpr,
D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    # Index vectors
    d_ptr   = tl.arange(0, D)
    c_ptr   = tl.arange(0, C)
    g_ptr   = tl.arange(0, G_PAD)
    g_mask  = g_ptr < G
    idx_ptr = tl.arange(0, max_chunk_size)

    # Stride across B for x
    x_stride = G * D

    # Persistent cache: the current x[x_idx] as a whole [G_PAD, D] tile
    current_x_idx = tl.full((), -1, dtype=tl.int32)
    x_i = tl.zeros((G_PAD, D), dtype=tl.float32)

    
    for i in range(start, end):
        # Select x for this workload
        x_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
        if x_idx_i32 != current_x_idx:
            x_base = (x_idx_i32 * x_stride).to(tl.int32)
            # Load x_i: [G_PAD, D] (f32). Rows g >= G are outside the
            # tensor, so they are masked out and zero-filled; zeros contribute
            # nothing to the dot product below and are never stored.
            x_offs = x_base + (g_ptr[:, None] * D + d_ptr[None, :]).to(tl.int32)
            x_i = tl.load(x + x_offs, mask=g_mask[:, None], other=0.0).to(tl.float32) # f32
            current_x_idx = x_idx_i32

        # Range of y rows for this workload
        y_len = tl.load(winfo_y_lens + i)
        y_off = tl.load(winfo_y_offsets + i)
        valid = idx_ptr < y_len

        # Row indices for this chunk of y
        y_idx_i32 = tl.load(indices + y_off + idx_ptr, mask=valid, other=0).to(tl.int32)

        # Load y_tile: [rows, C, D] (f32)
        # Linear offset: row*C*D + c*D + d
        offs_y = (
            (y_idx_i32[:, None, None] * (C * D)) +
            (c_ptr[None, :, None]     * D) +
            d_ptr[None, None, :]
        ).to(tl.int32)
        y_tile = tl.load(y + offs_y, mask=valid[:, None, None], other=0.0).to(tl.float32)   # [rows, C, D], bf16

        # Reshape to [rows*C, D] as the left operand (f32)
        rows_total: tl.constexpr = max_chunk_size * C
        y_rc = tl.reshape(y_tile, (rows_total, D))  # [RC, D], f32

        # Use x without transpose: x_i is [G_PAD, D] (f32)
        # Elementwise multiply in f32, then cast to fp32 and reduce over D:
        # [RC, 1, D] * [1, G_PAD, D] -> [RC, G_PAD, D] (f32), then sum over D -> [RC, G_PAD] (fp32)
        prod_ = y_rc[:, None, :] * x_i[None, :, :]   # f32 mult
        acc = tl.sum(prod_, 2)             # fp32 reduction on D

        # Reshape back to [rows, C, G] and store (fp32)
        o_i = tl.reshape(acc, (max_chunk_size, C, G_PAD))  # [rows, C, G_PAD], fp32
        o_i = o_i.to(tl.bfloat16)
        # Linear output offset: row*C*G + c*G + g, where row starts at y_off
        offs_o = (
            ((y_off + idx_ptr[:, None, None]) * (C * G)) +
            (c_ptr[None, :, None] * G) +
            g_ptr[None, None, :]
        ).to(tl.int32)

        tl.store(o + offs_o, o_i, mask=valid[:, None, None] & g_mask[None, None, :])



def mm_bpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    mm_bpr_kernel[(ctx.num_sms,)](
        x, y, o, 
        ctx.dense_kv_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], next_pow2(x.shape[-2]), y.shape[-2], x.shape[-1],
        num_warps=32, num_stages=1
    )


def _mm_bpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_x_indices: torch.Tensor,
winfo_y_offsets: torch.Tensor,
winfo_y_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int,
):  
    
    mm_bpr_kernel[(num_sms,)](
        x, y, o, 
        indices,
        winfo_x_indices,
        winfo_y_offsets,
        winfo_y_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], next_pow2(x.shape[-2]), y.shape[-2], x.shape[-1],
        num_warps=32, num_stages=1
    )



@triton.jit
def mm_rrr_kernel(
x,  # [*, x_D0, x_D1]
y,  # [*, y_D0, y_D1]
o,  # [*, o_D0, o_D1]
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
y_D0: tl.constexpr,
o_D0: tl.constexpr,
x_D1: tl.constexpr,
y_D1: tl.constexpr,
o_D1: tl.constexpr
):  
    
    # o_D0 = y_D0, o_D1 = x_D0, x_D1 = y_D1
    # o = yx^t 
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    idx_ptr = tl.arange(0, max_chunk_size)
    
    x_dim0 = tl.arange(0, x_D0)
    x_dim1 = tl.arange(0, x_D1)
    
    y_dim0 = tl.arange(0, y_D0)
    y_dim1 = tl.arange(0, y_D1)
    
    o_dim0 = tl.arange(0, o_D0)
    o_dim1 = tl.arange(0, o_D1)
    
    
    for i in range(start, end):
        
        # Range of y rows for this workload
        _len = tl.load(winfo_lens + i)
        _off = tl.load(winfo_offsets + i)
        valid = idx_ptr < _len
        x_i_ptr = x + _off * x_D0 * x_D1 + \
                idx_ptr[:, None, None] * x_D0 * x_D1 + \
                x_dim0[None,:,None] * x_D1 + \
                x_dim1[None, None, :]

        x_i = tl.load(x_i_ptr, mask=valid[:,None,None], other=0.0).to(tl.float32)
        
        
        y_i_ptr = y + _off * y_D0 * y_D1 + \
                idx_ptr[:, None, None] * y_D0 * y_D1 + \
                y_dim0[None,:,None] * y_D1 + \
                y_dim1[None, None, :]

        y_i = tl.load(y_i_ptr, mask=valid[:,None,None], other=0.0).to(tl.float32)
        
        o_i = tl.sum((x_i[:,None,:,:] * y_i[:,:,None,:]), axis=3)
        o_i = o_i.to(tl.bfloat16)
        
        o_i_ptr = o + _off * o_D0 * o_D1 + \
                idx_ptr[:, None, None] * o_D0 * o_D1 + \
                o_dim0[None,:,None] * o_D1 + \
                o_dim1[None, None, :]
        
        
        tl.store(o_i_ptr, o_i, mask=valid[:, None, None])
        
        

def mm_rrr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    mm_rrr_kernel[(ctx.num_sms,)](
        x, y, o, 
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], y.shape[-2], o.shape[-2],
        x.shape[-1], y.shape[-1], o.shape[-1],
        num_warps=32, num_stages=1
    )


def _mm_rrr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int
):  
    
    mm_rrr_kernel[(8 * num_sms,)](
        x, y, o, 
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        num_warps=4, 
        num_stages=1
    )
    

@triton.jit
def mm_rpr_kernel(
x,  # [*, x_D0, x_D1]
y,  # [*, y_D0, y_D1]
o,  # [*, o_D0, o_D1]
indices,             # int32
winfo_offsets,     # int32
winfo_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
y_D0: tl.constexpr,
o_D0: tl.constexpr,
x_D1: tl.constexpr,
y_D1: tl.constexpr,
o_D1: tl.constexpr
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
    x_dim0 = tl.arange(0, x_D0)
    x_dim1 = tl.arange(0, x_D1)
    
    y_dim0 = tl.arange(0, y_D0)
    y_dim1 = tl.arange(0, y_D1)
    
    o_dim0 = tl.arange(0, o_D0)
    o_dim1 = tl.arange(0, o_D1)

    
    for i in range(start, end):
        

        # Range of y rows for this workload
        _len = tl.load(winfo_lens + i)
        _off = tl.load(winfo_offsets + i)
        valid = idx_ptr < _len

        # Row indices for this chunk of y
        y_idx_i32 = tl.load(indices + _off + idx_ptr, mask=valid, other=0).to(tl.int32)

        offs_y = (
            (y_idx_i32[:, None, None] * (y_D0 * y_D1)) +
            (y_dim0[None, :, None]     * y_D1) +
            y_dim1[None, None, :]
        ).to(tl.int32)
        
        y_i = tl.load(y + offs_y, mask=valid[:, None, None], other=0.0).to(tl.float32)  # [rows, C, D], f32

        x_i_ptr = x + _off * x_D0 * x_D1 + \
                idx_ptr[:, None, None] * x_D0 * x_D1 + \
                x_dim0[None,:,None] * x_D1 + \
                x_dim1[None, None, :]

        x_i = tl.load(x_i_ptr, mask=valid[:,None,None], other=0.0).to(tl.float32)
        
        
        o_i = tl.sum((x_i[:,None,:,:] * y_i[:,:,None,:]), axis=3)
        o_i = o_i.to(tl.bfloat16)
        
        o_i_ptr = o + _off * o_D0 * o_D1 + \
                idx_ptr[:, None, None] * o_D0 * o_D1 + \
                o_dim0[None,:,None] * o_D1 + \
                o_dim1[None, None, :]
        
        
        tl.store(o_i_ptr, o_i, mask=valid[:, None, None])
        


def mm_rpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    mm_rpr_kernel[(ctx.num_sms,)](
        x, y, o, 
        ctx.dense_kv_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], y.shape[-2], o.shape[-2],
        x.shape[-1], y.shape[-1], o.shape[-1],
        num_warps=32, num_stages=1
    )


def _mm_rpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_offsets: torch.Tensor,
winfo_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int
):  
    
    mm_rpr_kernel[(8 * num_sms,)](
        x, y, o, 
        indices,
        winfo_offsets,
        winfo_lens,
        winfo_num_workloads,
        max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-2], 
        x.shape[-1],
        y.shape[-1],
        o.shape[-1],
        num_warps=4, 
        num_stages=1
    )