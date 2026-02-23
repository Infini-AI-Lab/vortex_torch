import torch
import triton
import triton.language as tl

@triton.jit
def set_kv_buffer_kernel(
    k_cache,
    v_cache,
    new_k,
    new_v,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)    
    dim = tl.arange(0, HEAD_DIM)
    
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr)
    src_v = tl.load(new_v + src_ptr)
    
    token_position = tl.load(loc + token_id)
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    
    tl.store(dst_k_ptr, src_k)
    tl.store(dst_v_ptr, src_v)


@triton.jit
def set_kv_buffer_int8_kernel(
    k_cache,        # int8 paged K cache
    v_cache,        # int8 paged V cache
    k_scale_cache,  # fp16 per-token K scale [num_pages, page_size, 1]
    v_scale_cache,  # fp16 per-token V scale [num_pages, page_size, 1]
    new_k,          # bf16 input K [NNZ, NUM_KV_HEAD, HEAD_DIM]
    new_v,          # bf16 input V [NNZ, NUM_KV_HEAD, HEAD_DIM]
    loc,            # int64 token positions
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    """Quantize bf16 K/V to int8 with per-token absmax scaling and write to paged buffers."""
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load bf16 source values
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    # Compute per-token absmax scale: scale = absmax / 127
    absmax_k = tl.max(tl.abs(src_k), axis=0)
    absmax_v = tl.max(tl.abs(src_v), axis=0)
    # Avoid division by zero
    scale_k = absmax_k / 127.0 + 1e-10
    scale_v = absmax_v / 127.0 + 1e-10

    # Quantize to int8: round(x / scale), clamp to [-128, 127]
    q_k = tl.extra.cuda.libdevice.rint(src_k / scale_k)
    q_k = tl.minimum(tl.maximum(q_k, -128.0), 127.0).to(tl.int8)
    q_v = tl.extra.cuda.libdevice.rint(src_v / scale_v)
    q_v = tl.minimum(tl.maximum(q_v, -128.0), 127.0).to(tl.int8)

    # Compute paged destination offset (same layout as bf16 kernel)
    token_position = tl.load(loc + token_id)
    page_id = token_position // PAGE_SIZE
    in_page_offset = token_position % PAGE_SIZE
    position_trans = page_id * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + in_page_offset

    # Write int8 values
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    tl.store(dst_k_ptr, q_k)
    tl.store(dst_v_ptr, q_v)

    # Write per-token scales (fp16): shape [num_pages, page_size, 1]
    # Layout: page_id * PAGE_SIZE + in_page_offset (flat per-head, one scale per token per head)
    scale_offset = (page_id * NUM_KV_HEAD + head_id) * PAGE_SIZE + in_page_offset
    tl.store(k_scale_cache + scale_offset, scale_k.to(tl.float16))
    tl.store(v_scale_cache + scale_offset, scale_v.to(tl.float16))


def set_kv_buffer_int8_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int
):
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    set_kv_buffer_int8_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        new_k,
        new_v,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )


def set_kv_buffer_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]
    
    set_kv_buffer_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache,
        v_cache,
        new_k,
        new_v,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )

