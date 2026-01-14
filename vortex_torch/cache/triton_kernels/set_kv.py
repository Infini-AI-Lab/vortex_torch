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

@triton.jit
def set_kv_buffer_cpu_and_gpu_kernel(
    cpu_k_cache,
    cpu_v_cache,
    gpu_k_staging,
    gpu_v_staging,
    new_k,
    new_v,
    loc,
    cpu_to_gpu_slot_map_ptr,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE_ID: tl.constexpr,
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
    cpu_position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position % PAGE_SIZE

    cpu_dst_k_ptr = cpu_k_cache + cpu_position_trans * HEAD_DIM + dim
    cpu_dst_v_ptr = cpu_v_cache + cpu_position_trans * HEAD_DIM + dim
    tl.store(cpu_dst_k_ptr, src_k)
    tl.store(cpu_dst_v_ptr, src_v)

    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id

    if page_id < MAX_PAGE_ID:
        gpu_slot = tl.load(cpu_to_gpu_slot_map_ptr + page_id)
        if gpu_slot >= 0:
            gpu_position_trans = gpu_slot * PAGE_SIZE + (token_position % PAGE_SIZE)
            gpu_dst_k_ptr = gpu_k_staging + gpu_position_trans * HEAD_DIM + dim
            gpu_dst_v_ptr = gpu_v_staging + gpu_position_trans * HEAD_DIM + dim
            tl.store(gpu_dst_k_ptr, src_k)
            tl.store(gpu_dst_v_ptr, src_v)


def store_kv_cpu_and_gpu(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    cpu_to_gpu_slot_map: torch.Tensor,
    max_page_id: int,
):
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    set_kv_buffer_cpu_and_gpu_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_staging,
        gpu_v_staging,
        new_k,
        new_v,
        loc,
        cpu_to_gpu_slot_map,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size,
        max_page_id,
    )
