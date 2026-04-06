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


@triton.jit
def set_kv_buffer_fp8_kernel(
    k_cache,        # uint8 paged K cache
    v_cache,        # uint8 paged V cache
    new_k,          # bf16 input K [NNZ, NUM_KV_HEAD, HEAD_DIM]
    new_v,          # bf16 input V [NNZ, NUM_KV_HEAD, HEAD_DIM]
    loc,            # int64 token positions
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP8_TYPE: tl.constexpr,   # 1: e4m3 (max=448), 2: e5m2 (max=57344)
    k_scale,                   # float: per-tensor scale for K quantization
    v_scale,                   # float: per-tensor scale for V quantization
):
    """Quantize bf16 K/V to fp8, bitcast to uint8, and scatter into paged cache."""
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load bf16 source values
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    # Scale down: quantized = real_value / scale
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    scaled_k = src_k * inv_k_scale
    scaled_v = src_v * inv_v_scale

    # Clamp and cast to fp8, then bitcast to uint8 for storage
    if FP8_TYPE == 1:
        # e4m3: max = 448.0
        clamped_k = tl.minimum(tl.maximum(scaled_k, -448.0), 448.0)
        clamped_v = tl.minimum(tl.maximum(scaled_v, -448.0), 448.0)
        q_k = clamped_k.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        q_v = clamped_v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        # e5m2: max = 57344.0
        clamped_k = tl.minimum(tl.maximum(scaled_k, -57344.0), 57344.0)
        clamped_v = tl.minimum(tl.maximum(scaled_v, -57344.0), 57344.0)
        q_k = clamped_k.to(tl.float8e5).to(tl.uint8, bitcast=True)
        q_v = clamped_v.to(tl.float8e5).to(tl.uint8, bitcast=True)

    # Compute paged destination offset
    token_position = tl.load(loc + token_id)
    page_id = token_position // PAGE_SIZE
    in_page_offset = token_position % PAGE_SIZE
    position_trans = page_id * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + in_page_offset

    # Write uint8 values
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    tl.store(dst_k_ptr, q_k)
    tl.store(dst_v_ptr, q_v)


def set_kv_buffer_fp8_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    k_scale: float,
    v_scale: float,
    fp8_type: int = 1,
):
    """Quantize bf16 K/V to fp8, bitcast to uint8, and scatter into paged cache.

    Args:
        fp8_type: 1 for e4m3 (default), 2 for e5m2.
        k_scale: per-tensor scale used for K quantization.
        v_scale: per-tensor scale used for V quantization.
    """
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    set_kv_buffer_fp8_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache, v_cache,
        new_k, new_v,
        loc,
        NUM_KV_HEAD, NNZ, HEAD_DIM, page_size,
        FP8_TYPE=fp8_type,
        k_scale=k_scale,
        v_scale=v_scale,
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


# ============================================================
# INT8: quantize bf16→int8, write to CPU + GPU staging + scales
# ============================================================

@triton.jit
def set_kv_buffer_cpu_and_gpu_int8_kernel(
    cpu_k_cache,        # int8 CPU pinned K cache
    cpu_v_cache,        # int8 CPU pinned V cache
    gpu_k_staging,      # int8 GPU staging K cache
    gpu_v_staging,      # int8 GPU staging V cache
    gpu_k_scale,        # fp16 persistent K scale [num_pages_cpu, page_size, 1]
    gpu_v_scale,        # fp16 persistent V scale [num_pages_cpu, page_size, 1]
    new_k,              # bf16 input K [NNZ, NUM_KV_HEAD, HEAD_DIM]
    new_v,              # bf16 input V [NNZ, NUM_KV_HEAD, HEAD_DIM]
    loc,                # int64 token positions
    cpu_to_gpu_slot_map_ptr,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE_ID: tl.constexpr,
):
    """Quantize bf16→int8 and write to CPU pinned + GPU staging + persistent scales."""
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load bf16 source
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    # Quantize: scale = absmax / 127
    absmax_k = tl.max(tl.abs(src_k), axis=0)
    absmax_v = tl.max(tl.abs(src_v), axis=0)
    scale_k = absmax_k / 127.0 + 1e-10
    scale_v = absmax_v / 127.0 + 1e-10
    q_k = tl.extra.cuda.libdevice.rint(src_k / scale_k)
    q_k = tl.minimum(tl.maximum(q_k, -128.0), 127.0).to(tl.int8)
    q_v = tl.extra.cuda.libdevice.rint(src_v / scale_v)
    q_v = tl.minimum(tl.maximum(q_v, -128.0), 127.0).to(tl.int8)

    # Compute paged offsets
    token_position = tl.load(loc + token_id)
    page_id = token_position // PAGE_SIZE
    in_page_offset = token_position % PAGE_SIZE
    cpu_position_trans = page_id * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + in_page_offset

    # Write int8 to CPU pinned (always)
    cpu_dst_k_ptr = cpu_k_cache + cpu_position_trans * HEAD_DIM + dim
    cpu_dst_v_ptr = cpu_v_cache + cpu_position_trans * HEAD_DIM + dim
    tl.store(cpu_dst_k_ptr, q_k)
    tl.store(cpu_dst_v_ptr, q_v)

    # Write scales to persistent GPU buffer (always, indexed by CPU page layout)
    flat_page_id = page_id * NUM_KV_HEAD + head_id
    scale_offset = flat_page_id * PAGE_SIZE + in_page_offset
    tl.store(gpu_k_scale + scale_offset, scale_k.to(tl.float16))
    tl.store(gpu_v_scale + scale_offset, scale_v.to(tl.float16))

    # Write int8 to GPU staging (if page is cached)
    if flat_page_id < MAX_PAGE_ID:
        gpu_slot = tl.load(cpu_to_gpu_slot_map_ptr + flat_page_id)
        if gpu_slot >= 0:
            gpu_position_trans = gpu_slot * PAGE_SIZE + in_page_offset
            gpu_dst_k_ptr = gpu_k_staging + gpu_position_trans * HEAD_DIM + dim
            gpu_dst_v_ptr = gpu_v_staging + gpu_position_trans * HEAD_DIM + dim
            tl.store(gpu_dst_k_ptr, q_k)
            tl.store(gpu_dst_v_ptr, q_v)


def store_kv_cpu_and_gpu_int8(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    gpu_k_scale: torch.Tensor,
    gpu_v_scale: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    cpu_to_gpu_slot_map: torch.Tensor,
    max_page_id: int,
):
    """Quantize bf16→int8 and write to CPU + GPU staging + persistent scales."""
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    set_kv_buffer_cpu_and_gpu_int8_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer, cpu_v_buffer,
        gpu_k_staging, gpu_v_staging,
        gpu_k_scale, gpu_v_scale,
        new_k, new_v, loc,
        cpu_to_gpu_slot_map,
        NUM_KV_HEAD, NNZ, HEAD_DIM, page_size, max_page_id,
    )


# ============================================================
# FP8: quantize bf16→fp8 (uint8), write to CPU + GPU staging
# ============================================================

@triton.jit
def set_kv_buffer_cpu_and_gpu_fp8_kernel(
    cpu_k_cache,        # uint8 CPU pinned K cache
    cpu_v_cache,        # uint8 CPU pinned V cache
    gpu_k_staging,      # uint8 GPU staging K cache
    gpu_v_staging,      # uint8 GPU staging V cache
    new_k,              # bf16 input K [NNZ, NUM_KV_HEAD, HEAD_DIM]
    new_v,              # bf16 input V [NNZ, NUM_KV_HEAD, HEAD_DIM]
    loc,                # int64 token positions
    cpu_to_gpu_slot_map_ptr,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE_ID: tl.constexpr,
    FP8_TYPE: tl.constexpr,
    k_scale,
    v_scale,
):
    """Quantize bf16→fp8 (uint8) and write to CPU pinned + GPU staging."""
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load bf16 source
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    # Scale and clamp
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    scaled_k = src_k * inv_k_scale
    scaled_v = src_v * inv_v_scale

    if FP8_TYPE == 1:
        clamped_k = tl.minimum(tl.maximum(scaled_k, -448.0), 448.0)
        clamped_v = tl.minimum(tl.maximum(scaled_v, -448.0), 448.0)
        q_k = clamped_k.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        q_v = clamped_v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        clamped_k = tl.minimum(tl.maximum(scaled_k, -57344.0), 57344.0)
        clamped_v = tl.minimum(tl.maximum(scaled_v, -57344.0), 57344.0)
        q_k = clamped_k.to(tl.float8e5).to(tl.uint8, bitcast=True)
        q_v = clamped_v.to(tl.float8e5).to(tl.uint8, bitcast=True)

    # Compute paged offsets
    token_position = tl.load(loc + token_id)
    page_id = token_position // PAGE_SIZE
    in_page_offset = token_position % PAGE_SIZE
    cpu_position_trans = page_id * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + in_page_offset

    # Write uint8 to CPU pinned (always)
    cpu_dst_k_ptr = cpu_k_cache + cpu_position_trans * HEAD_DIM + dim
    cpu_dst_v_ptr = cpu_v_cache + cpu_position_trans * HEAD_DIM + dim
    tl.store(cpu_dst_k_ptr, q_k)
    tl.store(cpu_dst_v_ptr, q_v)

    # Write uint8 to GPU staging (if page is cached)
    flat_page_id = page_id * NUM_KV_HEAD + head_id
    if flat_page_id < MAX_PAGE_ID:
        gpu_slot = tl.load(cpu_to_gpu_slot_map_ptr + flat_page_id)
        if gpu_slot >= 0:
            gpu_position_trans = gpu_slot * PAGE_SIZE + in_page_offset
            gpu_dst_k_ptr = gpu_k_staging + gpu_position_trans * HEAD_DIM + dim
            gpu_dst_v_ptr = gpu_v_staging + gpu_position_trans * HEAD_DIM + dim
            tl.store(gpu_dst_k_ptr, q_k)
            tl.store(gpu_dst_v_ptr, q_v)


def store_kv_cpu_and_gpu_fp8(
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
    k_scale: float,
    v_scale: float,
    fp8_type: int = 1,
):
    """Quantize bf16→fp8 (uint8) and write to CPU pinned + GPU staging."""
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    set_kv_buffer_cpu_and_gpu_fp8_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer, cpu_v_buffer,
        gpu_k_staging, gpu_v_staging,
        new_k, new_v, loc,
        cpu_to_gpu_slot_map,
        NUM_KV_HEAD, NNZ, HEAD_DIM, page_size, max_page_id,
        FP8_TYPE=fp8_type,
        k_scale=k_scale,
        v_scale=v_scale,
    )


# ============================================================
# INT8 unified: quantize bf16→int8, route to EITHER CPU or GPU
# (decode path — mirrors store_kv_unified C++ routing logic)
# ============================================================

@triton.jit
def store_kv_unified_int8_kernel(
    cpu_k_cache,        # int8 CPU pinned K cache
    cpu_v_cache,        # int8 CPU pinned V cache
    gpu_k_staging,      # int8 GPU staging K cache
    gpu_v_staging,      # int8 GPU staging V cache
    gpu_k_scale,        # fp16 persistent K scale on GPU (always written)
    gpu_v_scale,        # fp16 persistent V scale on GPU (always written)
    new_k,              # bf16 input K [NNZ, NUM_KV_HEAD, HEAD_DIM]
    new_v,              # bf16 input V [NNZ, NUM_KV_HEAD, HEAD_DIM]
    loc,                # int64 token positions
    cpu_to_gpu_slot_map_ptr,  # int32: >=0 means GPU slot, <0 means CPU
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Quantize bf16→int8, route to EITHER CPU or GPU based on slot map.
    Scales always written to persistent GPU buffer."""
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load bf16 source
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    # Per-token-per-head quantize: scale = absmax / 127
    absmax_k = tl.max(tl.abs(src_k), axis=0)
    absmax_v = tl.max(tl.abs(src_v), axis=0)
    scale_k = absmax_k / 127.0 + 1e-10
    scale_v = absmax_v / 127.0 + 1e-10
    q_k = tl.extra.cuda.libdevice.rint(src_k / scale_k)
    q_k = tl.minimum(tl.maximum(q_k, -128.0), 127.0).to(tl.int8)
    q_v = tl.extra.cuda.libdevice.rint(src_v / scale_v)
    q_v = tl.minimum(tl.maximum(q_v, -128.0), 127.0).to(tl.int8)

    # Compute paged offsets
    token_position = tl.load(loc + token_id)
    page_id = token_position // PAGE_SIZE
    in_page_offset = token_position % PAGE_SIZE
    flat_page_id = page_id * NUM_KV_HEAD + head_id

    # Always write scales to persistent GPU buffer
    scale_offset = flat_page_id * PAGE_SIZE + in_page_offset
    tl.store(gpu_k_scale + scale_offset, scale_k.to(tl.float16))
    tl.store(gpu_v_scale + scale_offset, scale_v.to(tl.float16))

    # Route int8 data to CPU or GPU based on slot map
    gpu_slot = tl.load(cpu_to_gpu_slot_map_ptr + flat_page_id)
    if gpu_slot >= 0:
        # Page is in GPU staging at gpu_slot
        gpu_position_trans = gpu_slot * PAGE_SIZE + in_page_offset
        gpu_dst_k_ptr = gpu_k_staging + gpu_position_trans * HEAD_DIM + dim
        gpu_dst_v_ptr = gpu_v_staging + gpu_position_trans * HEAD_DIM + dim
        tl.store(gpu_dst_k_ptr, q_k)
        tl.store(gpu_dst_v_ptr, q_v)
    else:
        # Page is in CPU, use page layout directly
        cpu_position_trans = page_id * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + in_page_offset
        cpu_dst_k_ptr = cpu_k_cache + cpu_position_trans * HEAD_DIM + dim
        cpu_dst_v_ptr = cpu_v_cache + cpu_position_trans * HEAD_DIM + dim
        tl.store(cpu_dst_k_ptr, q_k)
        tl.store(cpu_dst_v_ptr, q_v)


def store_kv_unified_int8(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    gpu_k_scale: torch.Tensor,
    gpu_v_scale: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    cpu_to_gpu_slot_map: torch.Tensor,
):
    """Quantize bf16→int8, route to EITHER CPU or GPU. Scales always on GPU."""
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    store_kv_unified_int8_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer, cpu_v_buffer,
        gpu_k_staging, gpu_v_staging,
        gpu_k_scale, gpu_v_scale,
        new_k, new_v, loc,
        cpu_to_gpu_slot_map,
        NUM_KV_HEAD, NNZ, HEAD_DIM, page_size,
    )


# ============================================================
# FP8 unified: quantize bf16→fp8 (uint8), route to EITHER CPU or GPU
# (decode path — mirrors store_kv_unified C++ routing logic)
# ============================================================

@triton.jit
def store_kv_unified_fp8_kernel(
    cpu_k_cache,        # uint8 CPU pinned K cache
    cpu_v_cache,        # uint8 CPU pinned V cache
    gpu_k_staging,      # uint8 GPU staging K cache
    gpu_v_staging,      # uint8 GPU staging V cache
    new_k,              # bf16 input K [NNZ, NUM_KV_HEAD, HEAD_DIM]
    new_v,              # bf16 input V [NNZ, NUM_KV_HEAD, HEAD_DIM]
    loc,                # int64 token positions
    cpu_to_gpu_slot_map_ptr,  # int32: >=0 means GPU slot, <0 means CPU
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP8_TYPE: tl.constexpr,
    k_scale,
    v_scale,
):
    """Quantize bf16→fp8 (uint8), route to EITHER CPU or GPU based on slot map."""
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load bf16 source
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr).to(tl.float32)
    src_v = tl.load(new_v + src_ptr).to(tl.float32)

    # Per-tensor scale and clamp
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    scaled_k = src_k * inv_k_scale
    scaled_v = src_v * inv_v_scale

    if FP8_TYPE == 1:
        clamped_k = tl.minimum(tl.maximum(scaled_k, -448.0), 448.0)
        clamped_v = tl.minimum(tl.maximum(scaled_v, -448.0), 448.0)
        q_k = clamped_k.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        q_v = clamped_v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        clamped_k = tl.minimum(tl.maximum(scaled_k, -57344.0), 57344.0)
        clamped_v = tl.minimum(tl.maximum(scaled_v, -57344.0), 57344.0)
        q_k = clamped_k.to(tl.float8e5).to(tl.uint8, bitcast=True)
        q_v = clamped_v.to(tl.float8e5).to(tl.uint8, bitcast=True)

    # Compute paged offsets
    token_position = tl.load(loc + token_id)
    page_id = token_position // PAGE_SIZE
    in_page_offset = token_position % PAGE_SIZE
    flat_page_id = page_id * NUM_KV_HEAD + head_id

    # Route uint8 data to CPU or GPU based on slot map
    gpu_slot = tl.load(cpu_to_gpu_slot_map_ptr + flat_page_id)
    if gpu_slot >= 0:
        # Page is in GPU staging at gpu_slot
        gpu_position_trans = gpu_slot * PAGE_SIZE + in_page_offset
        gpu_dst_k_ptr = gpu_k_staging + gpu_position_trans * HEAD_DIM + dim
        gpu_dst_v_ptr = gpu_v_staging + gpu_position_trans * HEAD_DIM + dim
        tl.store(gpu_dst_k_ptr, q_k)
        tl.store(gpu_dst_v_ptr, q_v)
    else:
        # Page is in CPU, use page layout directly
        cpu_position_trans = page_id * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + in_page_offset
        cpu_dst_k_ptr = cpu_k_cache + cpu_position_trans * HEAD_DIM + dim
        cpu_dst_v_ptr = cpu_v_cache + cpu_position_trans * HEAD_DIM + dim
        tl.store(cpu_dst_k_ptr, q_k)
        tl.store(cpu_dst_v_ptr, q_v)


def store_kv_unified_fp8(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    cpu_to_gpu_slot_map: torch.Tensor,
    k_scale: float,
    v_scale: float,
    fp8_type: int = 1,
):
    """Quantize bf16→fp8 (uint8), route to EITHER CPU or GPU."""
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    store_kv_unified_fp8_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer, cpu_v_buffer,
        gpu_k_staging, gpu_v_staging,
        new_k, new_v, loc,
        cpu_to_gpu_slot_map,
        NUM_KV_HEAD, NNZ, HEAD_DIM, page_size,
        FP8_TYPE=fp8_type,
        k_scale=k_scale,
        v_scale=v_scale,
    )
