"""
Custom Triton paged decode attention kernel for int8 KV cache.

Loads int8 K/V pages with per-token float32 scales, dequantizes inline in SRAM,
and computes standard multi-head attention with online softmax.

Adapted from SGLang's decode_attention.py for use with Vortex's paged layout
where each KV head is treated as a separate "batch" entry.
"""

import torch
import triton
import triton.language as tl

_MIN_BLOCK_KV = 32


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_int8_stage1(
    Q,                  # [batch, num_qo_heads, head_dim] bf16
    K_Buffer,           # int8 paged: flat
    V_Buffer,           # int8 paged: flat
    K_Scale_Buffer,     # fp16: flat (one scale per token slot)
    V_Scale_Buffer,     # fp16: flat
    sm_scale,
    kv_indptr,          # [batch + 1] int32, page-level
    kv_indices,         # page indices
    last_page_len,      # [batch] int32, tokens valid in last page
    Att_Out,            # [batch, num_qo_heads, max_kv_splits, head_dim]
    Att_Lse,            # [batch, num_qo_heads, max_kv_splits]
    num_kv_splits,      # [batch] int32
    stride_qbs,
    stride_qh,
    stride_buf_kbs,     # stride per token in K_Buffer (= head_dim)
    stride_buf_vbs,     # stride per token in V_Buffer (= head_dim)
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
):
    """
    Stage 1: For each (batch, head, kv_split), compute partial attention output and LSE.

    kv_indptr is page-level. Total tokens for batch i:
        (num_pages - 1) * PAGE_SIZE + last_page_len[i]
    """
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
    # Correct token count accounting for partial last page
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

            # Convert token offsets to page_id + in-page offset
            page_indices_in_seq = offs_n // PAGE_SIZE
            in_page_offsets = offs_n % PAGE_SIZE

            # Load page indices from kv_indices (physical page IDs)
            page_ids = tl.load(
                kv_indices + cur_batch_kv_start_idx + page_indices_in_seq,
                mask=mask_n,
                other=0,
            )

            # Flat token location: physical_page * PAGE_SIZE + in_page_offset
            kv_loc = page_ids * PAGE_SIZE + in_page_offsets

            # Load int8 K and dequantize
            offs_buf_k = kv_loc[:, None] * stride_buf_kbs + offs_d[None, :]
            k_int8 = tl.load(
                K_Buffer + offs_buf_k,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.float32)

            k_scale = tl.load(
                K_Scale_Buffer + kv_loc,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)
            k = k_int8 * k_scale[:, None]

            # Compute QK
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_n, qk, float("-inf"))

            # Load int8 V and dequantize
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :]
            v_int8 = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0,
            ).to(tl.float32)

            v_scale = tl.load(
                V_Scale_Buffer + kv_loc,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)
            v = v_int8 * v_scale[:, None]

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

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=mask_dv,
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_kernel_int8_stage2(
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


def paged_decode_int8(
    q: torch.Tensor,                # [batch, num_qo_heads, head_dim] bf16
    k_buffer: torch.Tensor,         # int8 paged K cache
    v_buffer: torch.Tensor,         # int8 paged V cache
    k_scale_buffer: torch.Tensor,   # fp16 scale for K
    v_scale_buffer: torch.Tensor,   # fp16 scale for V
    o: torch.Tensor,                # [batch, num_qo_heads, head_dim] bf16 output
    kv_indptr: torch.Tensor,        # [batch + 1] int32, page-level
    kv_indices: torch.Tensor,       # page indices
    last_page_len: torch.Tensor,    # [batch] int32
    num_kv_splits: torch.Tensor,    # [batch] int32
    max_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    att_out: torch.Tensor = None,   # optional pre-allocated [batch, head_num, max_kv_splits, Lv]
    att_lse: torch.Tensor = None,   # optional pre-allocated [batch, head_num, max_kv_splits]
):
    """
    Paged decode attention with int8 KV cache and inline dequantization.

    kv_indptr is page-level. last_page_len specifies valid tokens in the last page
    for each batch entry. Total tokens = (num_pages - 1) * page_size + last_page_len.
    """
    batch = q.shape[0]
    head_num = q.shape[1]
    Lk = q.shape[2]
    Lv = Lk

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = 64
    MAX_KV_SPLITS = max_kv_splits

    kv_group_num = head_num

    num_warps = 4 if kv_group_num == 1 else 2

    # Use pre-allocated buffers if provided, otherwise allocate
    if att_out is None:
        att_out = torch.empty(
            (batch, head_num, MAX_KV_SPLITS, Lv),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        att_out = att_out[:batch]
    if att_lse is None:
        att_lse = torch.empty(
            (batch, head_num, MAX_KV_SPLITS),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        att_lse = att_lse[:batch]

    stride_buf_kbs = k_buffer.shape[-1]
    stride_buf_vbs = v_buffer.shape[-1]

    grid_stage1 = (batch, head_num, MAX_KV_SPLITS)
    _fwd_kernel_int8_stage1[grid_stage1](
        q,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        last_page_len,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        stride_buf_kbs,
        stride_buf_vbs,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        PAGE_SIZE=page_size,
    )

    grid_stage2 = (batch, head_num)
    _fwd_kernel_int8_stage2[grid_stage2](
        att_out,
        att_lse,
        o,
        kv_indptr,
        last_page_len,
        num_kv_splits,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        PAGE_SIZE=page_size,
        num_warps=4,
        num_stages=2,
    )
