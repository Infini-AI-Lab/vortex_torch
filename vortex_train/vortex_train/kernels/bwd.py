"""Block-sparse attention backward, in Triton.

Two kernels, on purpose, because the two gradients want opposite traversal orders:

* ``dq`` is **q-major** — a `dq` tile sums over the KV blocks that query block
  selected, which is exactly the forward's loop. Balanced, reuses the forward's
  index.
* ``dk``/``dv`` are **kv-major** — a `dk` tile sums over the query blocks that
  selected that KV block, which needs the transposed index from
  :mod:`.transpose`.

Fusing them would force one order and pay the other's imbalance. Keeping them
separate also means each is a clean `one program owns one output tile` mapping, so
**no atomics touch the gradients**: each program accumulates its tile in fp32
registers and stores once. (Atomic bf16 accumulation would be both slow and
non-deterministic, which would make every downstream numerical comparison
unfalsifiable.)

Neither kernel materializes `p`: both recompute `p = exp(qk - lse)` per block from
the saved `lse`, the standard FlashAttention trade of a little math for a lot of
memory.

`delta = rowsum(do * o)` is precomputed once (a cheap elementwise pass) because
both kernels need it for the `ds = p * (dp - delta)` softmax-Jacobian term.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..pattern import SparsePattern
from .fwd import warps_for_tile


@triton.jit
def _delta_kernel(
    Out, DO, Delta,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_db, stride_dh, stride_dm,
    seqlen_q,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask = offs_m < seqlen_q
    base = pid_b * stride_ob + pid_h * stride_oh
    o = tl.load(Out + base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(Delta + pid_b * stride_db + pid_h * stride_dh + offs_m * stride_dm,
             tl.sum(o * do, axis=1), mask=mask)


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, DO, DQ, Lse, Delta,
    CNT, IDX,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_lb, stride_lh, stride_lm,
    stride_cb, stride_ch, stride_cm,
    stride_ib, stride_ih, stride_im, stride_ik,
    seqlen_q, seqlen_kv,
    GROUP: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr, MAX_SEL: tl.constexpr, CAUSAL: tl.constexpr,
    PACK_G: tl.constexpr, TILE_Q: tl.constexpr,
):
    """dq: one program owns a [TILE_Q, D] tile. Same traversal as the forward.

    ``PACK_G`` query heads of one GQA group share each K/V load, for the same reason
    as the forward: this kernel is HBM-bound on K/V, the group shares one selection,
    so serving ``PACK_G * BLOCK_Q`` rows per load cuts traffic by ``PACK_G``. The rows
    stay separate (each needs its own ``lse``/``delta``); packing buys the shared load.
    """
    pid_m = tl.program_id(0); pid_hs = tl.program_id(1); pid_b = tl.program_id(2)
    pid_kvh = (pid_hs * PACK_G) // GROUP

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_KV)
    offs_p = tl.arange(0, TILE_Q)
    p_h = pid_hs * PACK_G + offs_p // BLOCK_Q
    offs_m = pid_m * BLOCK_Q + offs_p % BLOCK_Q
    m_mask = (offs_m < seqlen_q) & (offs_p < PACK_G * BLOCK_Q)

    qrow = pid_b * stride_qb + p_h[:, None] * stride_qh
    lrow = pid_b * stride_lb + p_h * stride_lh
    q = tl.load(Q + qrow + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + qrow + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                 mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + lrow + offs_m * stride_lm, mask=m_mask, other=0.0)
    delta = tl.load(Delta + lrow + offs_m * stride_lm, mask=m_mask, other=0.0)

    dq = tl.zeros((TILE_Q, HEAD_DIM), dtype=tl.float32)
    cnt = tl.load(CNT + pid_b * stride_cb + pid_kvh * stride_ch + pid_m * stride_cm)
    idx_base = IDX + pid_b * stride_ib + pid_kvh * stride_ih + pid_m * stride_im

    for s in range(0, MAX_SEL):
        if s < cnt:
            kv_blk = tl.load(idx_base + s * stride_ik)
            offs_kv = kv_blk * BLOCK_KV + offs_n
            n_mask = offs_kv < seqlen_kv
            kvh = pid_b * stride_kb + pid_kvh * stride_kh
            k = tl.load(K + kvh + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=n_mask[:, None], other=0.0)
            v = tl.load(V + kvh + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=n_mask[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            p = tl.exp(qk - lse[:, None])
            p = tl.where(n_mask[None, :], p, 0.0)
            if CAUSAL:
                qpos = offs_m + (seqlen_kv - seqlen_q)
                p = tl.where(offs_kv[None, :] <= qpos[:, None], p, 0.0)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    tl.store(DQ + qrow + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
             dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def _bwd_dkv_packed_kernel(
    Q, K, V, DO, DK, DV, Lse, Delta,
    T_OFF, T_IND,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_lb, stride_lh, stride_lm,
    stride_ob_, stride_oh_, stride_on_,
    stride_tb, stride_th, stride_tn,
    seqlen_q, seqlen_kv,
    GROUP: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr, TILE_P: tl.constexpr,
    BATCH_R: tl.constexpr,
):
    """dk/dv with the GQA group packed into the contracted axis. Small ``block_q``.

    The insight: dk/dv already *sum* over the group, and a sum of outer products is
    one bigger outer product —

        sum_g A_g^T B_g  ==  [A_0; ...; A_{G-1}]^T [B_0; ...; B_{G-1}]

    — so stacking the group's rows along the query axis makes the group reduction
    fall out of the MMA for free. Two things follow, and both matter:

    * **The MMA's ``K >= 16`` is satisfied by real work, not padding.** At
      ``block_q=4, group=4`` the contracted dim is exactly 16 with zero waste, where
      padding a bare ``block_q=4`` to 16 would have thrown away 3/4 of the tile.
      Padding is still needed when ``GROUP * BLOCK_Q < 16`` (e.g. block_q=1, group=4
      → 4 real rows), but the waste drops from 16x to 4x.
    * **The per-query-head staging buffers disappear.** The general kernel writes
      ``dk_g``/``dv_g`` of shape ``[B, Hq, Skv, D]`` in fp32 and reduces them in a
      second launch; here the accumulator already holds the group sum, so it writes
      ``dk``/``dv`` directly. That removes two fp32 tensors the size of ``Hq`` KV
      caches and one kernel launch.

    Grid is over **KV heads**, not query heads, since one program now owns the whole
    group's contribution to a KV tile.
    """
    pid_n = tl.program_id(0)          # kv block
    pid_kvh = tl.program_id(1)        # kv head -- the whole group lives here now
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_KV + tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n < seqlen_kv

    # Packed row r <-> (segment entry r // (GROUP*BLOCK_Q),
    #                   group member (r // BLOCK_Q) % GROUP,
    #                   query token r % BLOCK_Q).
    #
    # BATCH_R segment entries are handled per iteration, so a tile holds
    # BATCH_R * GROUP * BLOCK_Q real rows. At block_q=1, group=4 that is 4 rows per
    # entry -- a quarter of the 16-row MMA minimum -- so batching 4 entries fills the
    # tile exactly and cuts the iteration count 4x. The segment can be up to 512 long
    # at block_q=1 (every query row selects the block), which is what made this loop
    # 67% of the step.
    offs_p = tl.arange(0, TILE_P)
    per_entry: tl.constexpr = GROUP * BLOCK_Q
    p_e = offs_p // per_entry                    # which segment entry this row reads
    p_g = (offs_p // BLOCK_Q) % GROUP            # group member
    p_t = offs_p % BLOCK_Q                       # query token within the block
    p_live = offs_p < (BATCH_R * per_entry)      # excludes MMA padding, if any

    kvh = pid_b * stride_kb + pid_kvh * stride_kh
    k = tl.load(K + kvh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0)
    v = tl.load(V + kvh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0)

    dk = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)

    seg = T_OFF + pid_b * stride_ob_ + pid_kvh * stride_oh_
    lo = tl.load(seg + pid_n * stride_on_)
    hi = tl.load(seg + (pid_n + 1) * stride_on_)

    # per-row head offset: the group member this packed row belongs to
    row_h = (pid_kvh * GROUP + p_g) * stride_qh + pid_b * stride_qb
    row_lh = (pid_kvh * GROUP + p_g) * stride_lh + pid_b * stride_lb

    for i in tl.range(lo, hi, BATCH_R, num_stages=1):  # see the general kernel on num_stages
        # Gather BATCH_R segment entries. They are NOT guaranteed contiguous or
        # sorted -- the transpose's scatter uses an atomic cursor -- so each packed
        # row looks up its own entry rather than assuming consecutive query blocks.
        e_idx = i + p_e
        e_ok = e_idx < hi
        m_blk = tl.load(T_IND + pid_b * stride_tb + pid_kvh * stride_th
                        + e_idx * stride_tn, mask=e_ok, other=0)
        offs_m = m_blk * BLOCK_Q + p_t
        m_mask = (offs_m < seqlen_q) & p_live & e_ok

        q = tl.load(Q + row_h[:, None] + offs_m[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd, mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + row_h[:, None] + offs_m[:, None] * stride_qm
                     + offs_d[None, :] * stride_qd, mask=m_mask[:, None], other=0.0)
        lse = tl.load(Lse + row_lh + offs_m * stride_lm, mask=m_mask, other=float("inf"))
        delta = tl.load(Delta + row_lh + offs_m * stride_lm, mask=m_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale            # [TILE_P, BLOCK_KV]
        p = tl.exp(qk - lse[:, None])
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
        if CAUSAL:
            qpos = offs_m + (seqlen_kv - seqlen_q)
            p = tl.where(offs_n[None, :] <= qpos[:, None], p, 0.0)

        # Contracting over TILE_P sums over BOTH the query tokens and the group, so
        # these accumulators are already the group-reduced gradients.
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * sm_scale

    out = kvh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    tl.store(DK + out, dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, DO, DK, DV, Lse, Delta,
    T_OFF, T_IND,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_gb, stride_gh, stride_gn, stride_gd,      # dk_g / dv_g: [B, Hq, Skv, D]
    stride_lb, stride_lh, stride_lm,
    stride_ob_, stride_oh_, stride_on_,
    stride_tb, stride_th, stride_tn,
    seqlen_q, seqlen_kv,
    GROUP: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr, TILE_Q: tl.constexpr,
):
    """dk/dv: one program owns one [BLOCK_KV, D] tile, walking its CSR segment.

    Because a program owns the tile exclusively, the fp32 accumulators live in
    registers and are stored once — no atomics, and bitwise-deterministic given a
    fixed CSR order.

    GQA: `dk` is accumulated over every query head in the group, so the grid is
    over query heads and the caller reduces the group dimension. Doing it here
    would need cross-program accumulation, i.e. the atomics we are avoiding.
    """
    pid_n = tl.program_id(0)          # kv block
    pid_h = tl.program_id(1)          # query head
    pid_b = tl.program_id(2)
    pid_kvh = pid_h // GROUP

    offs_n = pid_n * BLOCK_KV + tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, HEAD_DIM)
    # TILE_Q == BLOCK_Q here: this kernel is only dispatched for block_q >= 16, which
    # is what the MMA's contracted dimension needs. Smaller block_q goes to
    # _bwd_dkv_packed_kernel, which fills that dimension with the GQA group instead
    # of with padding.
    offs_q = tl.arange(0, TILE_Q)
    q_in_block = offs_q < BLOCK_Q
    n_mask = offs_n < seqlen_kv

    kvh = pid_b * stride_kb + pid_kvh * stride_kh
    k = tl.load(K + kvh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0)
    v = tl.load(V + kvh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0)

    dk = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)

    seg = T_OFF + pid_b * stride_ob_ + pid_kvh * stride_oh_
    lo = tl.load(seg + pid_n * stride_on_)
    hi = tl.load(seg + (pid_n + 1) * stride_on_)

    qh = pid_b * stride_qb + pid_h * stride_qh
    # num_stages=1 is LOAD-BEARING, not a tuning choice. Triton 3.6 miscompiles
    # this loop when it software-pipelines it: with two fp32 accumulators and a
    # data-dependent trip count, any program whose CSR segment is SHORTER than the
    # pipeline depth gets `dk` silently zeroed (depth 2 kills trip counts <= 2,
    # depth 3 kills <= 3). `dv` is never affected, and dropping either accumulator
    # makes it go away -- which is why it presents as "dk is wrong in proportion to
    # how many query blocks selected this KV block" rather than as an obvious break.
    #
    # Pinned per-loop rather than via the kernel's `num_stages` so the rest of the
    # kernel keeps its pipelining; measured cost is within 3% of a pipelined
    # split-kernel version at S=4k/16k, group 1/4. See test_dkv_short_segments.
    for i in tl.range(lo, hi, num_stages=1):
        m_blk = tl.load(T_IND + pid_b * stride_tb + pid_kvh * stride_th + i * stride_tn)
        offs_m = m_blk * BLOCK_Q + offs_q
        # `q_in_block` excludes the MMA padding: without it a pad row would read the
        # NEXT query block's tokens (offs_m keeps counting past BLOCK_Q) and add a
        # spurious contribution to this kv block's gradient.
        m_mask = (offs_m < seqlen_q) & q_in_block

        q = tl.load(Q + qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                     mask=m_mask[:, None], other=0.0)
        lse = tl.load(Lse + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_lm,
                      mask=m_mask, other=float("inf"))
        delta = tl.load(Delta + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_lm,
                        mask=m_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale                  # [BLOCK_Q, BLOCK_KV]
        p = tl.exp(qk - lse[:, None])
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
        if CAUSAL:
            qpos = offs_m + (seqlen_kv - seqlen_q)
            p = tl.where(offs_n[None, :] <= qpos[:, None], p, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * sm_scale

    # dk_g/dv_g are indexed by QUERY head (the group reduction happens after), so
    # they need their own strides -- reusing K's silently mis-strides the batch
    # dimension whenever Hq != Hkv.
    dkv = pid_b * stride_gb + pid_h * stride_gh
    tl.store(DK + dkv + offs_n[:, None] * stride_gn + offs_d[None, :] * stride_gd,
             dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + dkv + offs_n[:, None] * stride_gn + offs_d[None, :] * stride_gd,
             dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


@triton.jit
def _group_reduce_kernel(
    DK_G, DV_G, DK, DV,
    stride_gb, stride_gh, stride_gn, stride_gd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    seqlen_kv,
    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Sum dk_g/dv_g over the GQA group and cast to the KV dtype, in one pass.

    Accumulates in fp32: the sum is over GROUP terms, and doing it in bf16 loses
    precision that matters at large group sizes (the reason dk_g is fp32 at all).
    """
    pid_n = tl.program_id(0); pid_kvh = tl.program_id(1); pid_b = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n < seqlen_kv

    acc_k = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    for g in range(0, GROUP):
        h = pid_kvh * GROUP + g
        off = (pid_b * stride_gb + h * stride_gh
               + offs_n[:, None] * stride_gn + offs_d[None, :] * stride_gd)
        acc_k += tl.load(DK_G + off, mask=n_mask[:, None], other=0.0)
        acc_v += tl.load(DV_G + off, mask=n_mask[:, None], other=0.0)

    out = (pid_b * stride_kb + pid_kvh * stride_kh
           + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    tl.store(DK + out, acc_k.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + out, acc_v.to(DV.dtype.element_ty), mask=n_mask[:, None])


def sparse_attn_bwd(
    do: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, lse: torch.Tensor,
    pattern: SparsePattern,
    t_offsets: torch.Tensor, t_indices: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dq, dk, dv)``. ``do`` must be contiguous with ``q``'s layout."""
    b, hq, sq, d = q.shape
    hkv, skv = k.shape[1], k.shape[2]
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    do = do.contiguous()

    delta = torch.empty_like(lse)
    BLOCK_M = 128
    _delta_kernel[(triton.cdiv(sq, BLOCK_M), hq, b)](
        out, do, delta, *out.stride(), *delta.stride(), sq,
        BLOCK_M=BLOCK_M, HEAD_DIM=d,
    )

    dq = torch.zeros_like(q)
    # Same K/V-sharing rationale as the forward; see _bwd_dq_kernel's docstring.
    pack = min(group, max(1, 16 // pattern.block_q)) if pattern.block_q < 16 else 1
    # A packed head slice must lie entirely inside ONE GQA group: `pid_kvh` is derived
    # from the slice's first head, so a slice straddling two groups would read the
    # wrong group's selection for its later heads -- silently, with O(1) relative
    # error. That cannot happen for power-of-two `group` (which is all this project
    # supports): both `group` and `16 // block_q` are powers of two, so their min
    # divides `group`. Asserted rather than assumed because the failure is quiet.
    assert group % pack == 0, (
        f"packed head slice of {pack} does not divide the GQA group of {group}; "
        f"num_kv_heads * group must give a power-of-two group size"
    )
    _bwd_dq_kernel[(pattern.num_q_blocks, hq // pack, b)](
        q, k, v, do, dq, lse, delta, pattern.cnt, pattern.idx, scale,
        *q.stride(), *k.stride(), *lse.stride(),
        *pattern.cnt.stride(), *pattern.idx.stride(),
        sq, skv,
        GROUP=group, BLOCK_Q=pattern.block_q, BLOCK_KV=pattern.block_kv,
        HEAD_DIM=d, MAX_SEL=pattern.max_selected, CAUSAL=causal,
        PACK_G=pack, TILE_Q=max(triton.next_power_of_2(pack * pattern.block_q), 16),
        # dq carries more live state than the forward, so it wants a floor of 2 warps
        # even at a 16-row tile (measured 12.29 vs 13.73 ms).
        num_warps=warps_for_tile(max(triton.next_power_of_2(pack * pattern.block_q), 16),
                                 min_warps=2),
        num_stages=2,
    )

    # dk/dv: two kernels, dispatched on whether `block_q` alone fills the MMA's
    # contracted dimension (>= 16).
    #
    # `block_q >= 16`: one program per QUERY head, group-reduced afterwards. The
    # per-head tile is already big enough, and keeping the group on the grid gives
    # more parallelism than packing would.
    #
    # `block_q < 16`: pack the group into the contracted axis instead of padding it.
    # dk/dv already sum over the group, and a sum of outer products is one bigger
    # outer product, so the group reduction comes free from the MMA -- and the
    # `[B, Hq, Skv, D]` fp32 staging buffers plus their reduction launch disappear
    # with it. At block_q=4, group=4 the tile is exactly 16 rows of real work.
    if pattern.block_q >= 16:
        # fp32 staging: the group sum is over `group` terms and a bf16 accumulation
        # there measurably degrades gradients at large group sizes.
        dk_g = torch.zeros((b, hq, skv, d), dtype=torch.float32, device=q.device)
        dv_g = torch.zeros((b, hq, skv, d), dtype=torch.float32, device=q.device)
        _bwd_dkv_kernel[(pattern.num_kv_blocks, hq, b)](
            q, k, v, do, dk_g, dv_g, lse, delta, t_offsets, t_indices, scale,
            *q.stride(), *k.stride(), *dk_g.stride(), *lse.stride(),
            *t_offsets.stride(), *t_indices.stride(),
            sq, skv,
            GROUP=group, BLOCK_Q=pattern.block_q, BLOCK_KV=pattern.block_kv,
            HEAD_DIM=d, CAUSAL=causal, TILE_Q=pattern.block_q,
            # dk/dv deliberately keeps 8 warps regardless of head_dim: its tile is
            # BLOCK_KV rows with a long segment loop, and the extra warps hide the
            # gather latency (measured 11.23 ms at 8 warps vs 43.84 at 1).
            num_warps=8, num_stages=2,
        )
        # Reduce the GQA group dimension in ONE fused kernel rather than a
        # view -> sum(2) -> to(dtype) torch chain (which would launch a reduction
        # and a cast, and materialize an fp32 intermediate the size of dk_g).
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BLOCK_N = 64
        _group_reduce_kernel[(triton.cdiv(skv, BLOCK_N), hkv, b)](
            dk_g, dv_g, dk, dv,
            *dk_g.stride(), *dk.stride(),
            skv, GROUP=group, BLOCK_N=BLOCK_N, HEAD_DIM=d,
        )
        return dq, dk, dv

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    per_entry = group * pattern.block_q
    batch_r = max(1, -(-16 // per_entry))       # ceil(16 / per_entry)
    _bwd_dkv_packed_kernel[(pattern.num_kv_blocks, hkv, b)](
        q, k, v, do, dk, dv, lse, delta, t_offsets, t_indices, scale,
        *q.stride(), *k.stride(), *lse.stride(),
        *t_offsets.stride(), *t_indices.stride(),
        sq, skv,
        GROUP=group, BLOCK_Q=pattern.block_q, BLOCK_KV=pattern.block_kv,
        HEAD_DIM=d, CAUSAL=causal,
        # real rows = GROUP * BLOCK_Q; pad only if that is still under 16.
        # Batch enough segment entries to fill the MMA tile with real work: each
        # entry contributes group*block_q rows, so batch ceil(16 / that).
        BATCH_R=batch_r,
        TILE_P=max(triton.next_power_of_2(batch_r * group * pattern.block_q), 16),
        # see the general dk/dv path: 8 warps hides the gathered segment loop's latency.
        num_warps=8, num_stages=2,
    )
    return dq, dk, dv
