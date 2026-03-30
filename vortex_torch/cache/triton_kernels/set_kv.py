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


# ---------------------------------------------------------------------------
# Dequantization kernels (read direction: quantized paged cache → bf16)
# ---------------------------------------------------------------------------

@triton.jit
def _dequant_pages_kernel(
    src,                # quantized paged buffer flat
    src_scale,          # per-token scale buffer flat (int8 only)
    dst,                # bf16 destination buffer flat
    page_indices,       # int32 page indices to dequant
    NUM_PAGES,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    QUANT_TYPE: tl.constexpr,   # 1: int8, 2: e4m3, 3: e5m2
    tensor_scale,               # float: per-tensor scale (fp8 only)
    COMPACT: tl.constexpr,      # True: compact dst; False: in-place dst
):
    """Unified dequant kernel for selected pages → bf16.

    QUANT_TYPE==1: load int8, multiply by per-token scale from src_scale.
    QUANT_TYPE==2: load uint8, bitcast to float8e4nv, multiply by tensor_scale.
    QUANT_TYPE==3: load uint8, bitcast to float8e5, multiply by tensor_scale.
    COMPACT==True:  dst offset uses page_idx (compact buffer).
    COMPACT==False: dst offset uses global_page_id (in-place).
    """
    page_idx = tl.program_id(0)
    token_idx = tl.program_id(1)

    if page_idx >= NUM_PAGES:
        return

    global_page_id = tl.load(page_indices + page_idx)
    dims = tl.arange(0, BLOCK_DIM)
    mask_dim = dims < HEAD_DIM

    src_offset = (global_page_id * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    scale_offset = global_page_id * PAGE_SIZE + token_idx

    if QUANT_TYPE == 1:
        val = tl.load(src + src_offset, mask=mask_dim, other=0).to(tl.float32)
        scale = tl.load(src_scale + scale_offset).to(tl.float32)
        result = (val * scale).to(tl.bfloat16)
    elif QUANT_TYPE == 2:
        raw = tl.load(src + src_offset, mask=mask_dim, other=0)
        val = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        result = (val * tensor_scale).to(tl.bfloat16)
    else:  # QUANT_TYPE == 3
        raw = tl.load(src + src_offset, mask=mask_dim, other=0)
        val = raw.to(tl.float8e5, bitcast=True).to(tl.float32)
        result = (val * tensor_scale).to(tl.bfloat16)

    if COMPACT:
        dst_offset = (page_idx * PAGE_SIZE + token_idx) * HEAD_DIM + dims
    else:
        dst_offset = src_offset  # same position as source

    tl.store(dst + dst_offset, result, mask=mask_dim)


def dequant_pages_to_bf16(
    src: torch.Tensor,
    src_scale: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    head_dim: int,
    quant_type: int = 1,
    tensor_scale: float = 1.0,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Dequant selected pages to compact bf16 buffer.

    Args:
        quant_type: 1=int8 (per-token scale), 2=fp8 e4m3, 3=fp8 e5m2.
        tensor_scale: per-tensor scale (fp8 only, ignored for int8).
        out: optional pre-allocated bf16 buffer.
    """
    num_accessed_pages = page_indices.shape[0]
    if num_accessed_pages == 0:
        if out is not None:
            return out[:0]
        return torch.empty((0, page_size, head_dim), dtype=torch.bfloat16, device=src.device)

    if out is not None:
        dst = out[:num_accessed_pages]
    else:
        dst = torch.empty(
            (num_accessed_pages, page_size, head_dim),
            dtype=torch.bfloat16,
            device=src.device,
        )

    BLOCK_DIM = triton.next_power_of_2(head_dim)

    grid = (num_accessed_pages, page_size)
    _dequant_pages_kernel[grid](
        src, src_scale, dst, page_indices,
        NUM_PAGES=num_accessed_pages,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
        QUANT_TYPE=quant_type,
        tensor_scale=tensor_scale,
        COMPACT=True,
    )

    return dst


def dequant_pages_to_bf16_inplace(
    src: torch.Tensor,
    src_scale: torch.Tensor,
    dst: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    head_dim: int,
    quant_type: int = 1,
    tensor_scale: float = 1.0,
) -> None:
    """Dequant selected pages in-place (same page positions in dst).

    Args:
        quant_type: 1=int8 (per-token scale), 2=fp8 e4m3, 3=fp8 e5m2.
        tensor_scale: per-tensor scale (fp8 only, ignored for int8).
    """
    num_pages = page_indices.shape[0]
    if num_pages == 0:
        return

    BLOCK_DIM = triton.next_power_of_2(head_dim)

    grid = (num_pages, page_size)
    _dequant_pages_kernel[grid](
        src, src_scale, dst, page_indices,
        NUM_PAGES=num_pages,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
        QUANT_TYPE=quant_type,
        tensor_scale=tensor_scale,
        COMPACT=False,
    )


# ---------------------------------------------------------------------------
# Paged decode attention (unified quant_type-parameterized)
# ---------------------------------------------------------------------------

_MIN_BLOCK_KV = 32


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_paged_decode_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    K_Scale_Buffer,
    V_Scale_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    last_page_len,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    QUANT_TYPE: tl.constexpr,   # 0: bf16, 1: int8, 2: e4m3, 3: e5m2
    tensor_scale,               # per-tensor scale for fp8
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_num_pages = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    cur_last_page_len = tl.load(last_page_len + cur_batch)
    cur_batch_seq_len = (cur_batch_num_pages - 1) * PAGE_SIZE + cur_last_page_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0).to(tl.float32)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end

            page_indices_in_seq = offs_n // PAGE_SIZE
            in_page_offsets = offs_n % PAGE_SIZE
            page_ids = tl.load(
                kv_indices + cur_batch_kv_start_idx + page_indices_in_seq,
                mask=mask_n, other=0,
            )
            kv_loc = page_ids * PAGE_SIZE + in_page_offsets

            # Load K with quant-type-dependent dequantization
            offs_buf_k = kv_loc[:, None] * stride_buf_kbs + offs_d[None, :]
            if QUANT_TYPE == 0:
                k = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=mask_n[:, None] & mask_d[None, :], other=0,
                ).to(tl.float32)
            elif QUANT_TYPE == 1:
                k_int8 = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=mask_n[:, None] & mask_d[None, :], other=0,
                ).to(tl.float32)
                k_scale = tl.load(
                    K_Scale_Buffer + kv_loc, mask=mask_n, other=1.0,
                ).to(tl.float32)
                k = k_int8 * k_scale[:, None]
            elif QUANT_TYPE == 2:
                raw = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=mask_n[:, None] & mask_d[None, :], other=0,
                )
                k = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tensor_scale
            else:  # QUANT_TYPE == 3
                raw = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=mask_n[:, None] & mask_d[None, :], other=0,
                )
                k = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * tensor_scale

            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(mask_n, qk, float("-inf"))

            # Load V with quant-type-dependent dequantization
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :]
            if QUANT_TYPE == 0:
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=mask_n[:, None] & mask_dv[None, :], other=0,
                ).to(tl.float32)
            elif QUANT_TYPE == 1:
                v_int8 = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=mask_n[:, None] & mask_dv[None, :], other=0,
                ).to(tl.float32)
                v_scale = tl.load(
                    V_Scale_Buffer + kv_loc, mask=mask_n, other=1.0,
                ).to(tl.float32)
                v = v_int8 * v_scale[:, None]
            elif QUANT_TYPE == 2:
                raw = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=mask_n[:, None] & mask_dv[None, :], other=0,
                )
                v = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tensor_scale
            else:  # QUANT_TYPE == 3
                raw = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=mask_n[:, None] & mask_dv[None, :], other=0,
                )
                v = raw.to(tl.float8e5, bitcast=True).to(tl.float32) * tensor_scale

            # Online softmax accumulation
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _fwd_kernel_paged_decode_stage2(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    last_page_len,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Stage 2: Reduce split outputs via log-sum-exp merge."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_num_pages = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    cur_last_page_len = tl.load(last_page_len + cur_batch)
    cur_batch_seq_len = (cur_batch_num_pages - 1) * PAGE_SIZE + cur_last_page_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def paged_decode(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    page_size: int,
    quant_type: int = 0,
    k_scale_buffer: torch.Tensor = None,
    v_scale_buffer: torch.Tensor = None,
    tensor_scale: float = 1.0,
    logit_cap: float = 0.0,
    att_out: torch.Tensor = None,
    att_lse: torch.Tensor = None,
):
    """Unified paged decode attention.

    Args:
        quant_type: Controls K/V loading:
            0: bf16 (k_scale_buffer/v_scale_buffer unused)
            1: int8 with per-token scales (k_scale_buffer/v_scale_buffer required)
            2: fp8 e4m3 with per-tensor scale (tensor_scale required)
            3: fp8 e5m2 with per-tensor scale (tensor_scale required)
    """
    batch = q.shape[0]
    head_num = q.shape[1]
    Lk = q.shape[2]
    Lv = Lk

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = 128
    MAX_KV_SPLITS = max_kv_splits

    kv_group_num = head_num
    num_warps = 4

    if att_out is None:
        att_out = torch.empty(
            (batch, head_num, MAX_KV_SPLITS, Lv),
            dtype=torch.float32, device=q.device,
        )
    else:
        att_out = att_out[:batch]
    if att_lse is None:
        att_lse = torch.empty(
            (batch, head_num, MAX_KV_SPLITS),
            dtype=torch.float32, device=q.device,
        )
    else:
        att_lse = att_lse[:batch]

    stride_buf_kbs = k_buffer.shape[-1]
    stride_buf_vbs = v_buffer.shape[-1]

    # Use dummy tensors for scale buffers when not needed
    _k_scale = k_scale_buffer if k_scale_buffer is not None else k_buffer
    _v_scale = v_scale_buffer if v_scale_buffer is not None else v_buffer

    grid_stage1 = (batch, head_num, MAX_KV_SPLITS)
    _fwd_kernel_paged_decode_stage1[grid_stage1](
        q, k_buffer, v_buffer,
        _k_scale, _v_scale,
        sm_scale, kv_indptr, kv_indices, last_page_len,
        att_out, att_lse, num_kv_splits,
        q.stride(0), q.stride(1),
        stride_buf_kbs, stride_buf_vbs,
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk, Lv=Lv,
        PAGE_SIZE=page_size,
        QUANT_TYPE=quant_type,
        tensor_scale=tensor_scale,
    )

    grid_stage2 = (batch, head_num)
    _fwd_kernel_paged_decode_stage2[grid_stage2](
        att_out, att_lse, o,
        kv_indptr, last_page_len, num_kv_splits,
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        o.stride(0), o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        PAGE_SIZE=page_size,
        num_warps=4,
        num_stages=2,
    )

