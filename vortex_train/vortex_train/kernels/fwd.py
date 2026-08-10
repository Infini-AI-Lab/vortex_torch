"""Block-sparse attention forward, in Triton.

One program per (batch, query head, query block). Each walks only the KV blocks
its query block selected, keeping the running softmax in registers — so the
`[BLOCK_Q, cnt*BLOCK_KV]` probability matrix never exists in HBM. That is the
whole point: at T=128k those probs would be ~16 GB per layer.

Parallelization note. The forward is naturally load-balanced: with a fixed top-k
budget every query block visits `min(cnt, topk)` KV blocks, so programs do near-
equal work and a flat 3-D grid is enough. (The *backward* is not balanced — a KV
block is selected by an unpredictable number of query blocks, 1..129 measured at
S=512 blocks / topk=32 — which is why `bwd.py` splits differently.)

GQA: selection is per KV head, so all `Hq/Hkv` query heads in a group traverse the
same block list. We index the pattern with `head // group`, which lets the same
loaded KV tile serve the whole group.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..pattern import SparsePattern


@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse,
    CNT, IDX,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_cb, stride_ch, stride_cm,
    stride_ib, stride_ih, stride_im, stride_ik,
    seqlen_q, seqlen_kv,
    GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_SEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    PACK_G: tl.constexpr,             # query heads sharing one K/V load
    TILE_Q: tl.constexpr,             # = PACK_G * BLOCK_Q, padded to >= 16
):
    """One program per (query block, head *slice*, batch).

    With ``PACK_G > 1`` the program owns ``PACK_G`` query heads of one GQA group and
    loads each selected K/V block **once** for all of them. That is sound because the
    whole group shares one selection, and it is what makes small ``block_q`` viable:
    the kernel is HBM-bound on K/V, so serving `PACK_G * BLOCK_Q` query rows per load
    instead of `BLOCK_Q` cuts traffic by exactly `PACK_G`.

    Unlike the dk/dv backward, the group cannot be folded into a contraction here --
    each head needs its own softmax normalisation -- so the packed rows are kept
    *separate* and the online-softmax state is per row. Packing buys the shared load,
    not a bigger MMA.

    **Two alternatives were measured and rejected**, recorded so they are not retried:

    * *Union neighbouring query blocks* so a program's tile is full instead of 4-real-in-16
      at ``block_q=1``. Measured on a real ``block_topk`` pattern (seqlen 16384, topk 18):
      the union of R=16 consecutive tokens' selections is **102.7 blocks**, not the ~18 a
      strong-overlap assumption would give, so total row-block work is **1.47x worse**
      (R=32: 1.74x). Neighbouring tokens agree far less than they appear to if you sample
      only early tokens -- those are causally limited to few KV blocks, which is the trap:
      an initial sample over the first 4096 tokens suggested a union of ~32 and a 0.44x
      *win*. Unioning also forfeits ``PACK_G`` head packing, since the union is shared
      across the GQA group but each query head still needs its own softmax rows.
    * *KV-major forward* (as Flash-Sparse-Attention and flash-moba do, gathering scattered
      query rows into a dense tile). That eliminates padding entirely, but a KV-major
      forward cannot finish the online softmax in one pass -- a row's normaliser spans the
      KV blocks *it* selected, which live in different programs -- so it needs partial
      ``(o, m, l)`` and a second reduce. The partials are ``nnz x D`` fp32 per query head:
      ~4 GB at seqlen 16k across 32 heads, which is worse than the padding it removes.
    """
    pid_m = tl.program_id(0)          # query block
    pid_hs = tl.program_id(1)         # head slice: PACK_G query heads
    pid_b = tl.program_id(2)          # batch
    pid_kvh = (pid_hs * PACK_G) // GROUP   # kv head owning this group's selection

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_KV)

    # packed row r <-> (head pid_hs*PACK_G + r // BLOCK_Q, token r % BLOCK_Q)
    offs_p = tl.arange(0, TILE_Q)
    p_h = pid_hs * PACK_G + offs_p // BLOCK_Q
    offs_m = pid_m * BLOCK_Q + offs_p % BLOCK_Q
    m_mask = (offs_m < seqlen_q) & (offs_p < PACK_G * BLOCK_Q)

    q = tl.load(
        Q + pid_b * stride_qb + p_h[:, None] * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None], other=0.0,
    )

    # running softmax state (fp32 accumulators — bf16 in, fp32 accumulate)
    acc = tl.zeros((TILE_Q, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((TILE_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((TILE_Q,), dtype=tl.float32)

    cnt = tl.load(CNT + pid_b * stride_cb + pid_kvh * stride_ch + pid_m * stride_cm)
    idx_base = IDX + pid_b * stride_ib + pid_kvh * stride_ih + pid_m * stride_im

    # Loop over the SELECTED kv blocks only. MAX_SEL is a constexpr so the loop
    # unrolls; the `s < cnt` guard skips the padded tail without a host read.
    for s in range(0, MAX_SEL):
        if s < cnt:
            kv_blk = tl.load(idx_base + s * stride_ik)
            start_n = kv_blk * BLOCK_KV
            offs_kv = start_n + offs_n
            n_mask = offs_kv < seqlen_kv

            k_ptr = K + pid_b * stride_kb + pid_kvh * stride_kh
            k = tl.load(
                k_ptr + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * sm_scale          # [BLOCK_Q, BLOCK_KV]

            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            if CAUSAL:
                # absolute positions; handles seqlen_q != seqlen_kv (decode-style
                # offset) as well as the square training case.
                qpos = offs_m + (seqlen_kv - seqlen_q)
                qk = tl.where(offs_kv[None, :] <= qpos[:, None], qk, float("-inf"))

            # online softmax (FlashAttention): rescale the accumulator instead of
            # ever materializing the full probability matrix.
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptr = V + pid_b * stride_vb + pid_kvh * stride_vh
            vt = tl.load(
                v_ptr + offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=n_mask[:, None], other=0.0,
            )
            acc += tl.dot(p.to(vt.dtype), vt)
            m_i = m_new

    # A row with cnt == 0 attends to nothing: define it as zero output and
    # lse = -inf, rather than 0/0. The oracle makes the same choice.
    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]
    lse = tl.where(l_i > 0, m_i + tl.log(safe_l), float("-inf"))

    tl.store(
        Out + pid_b * stride_ob + p_h[:, None] * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(Out.dtype.element_ty), mask=m_mask[:, None],
    )
    tl.store(Lse + pid_b * stride_lb + p_h * stride_lh + offs_m * stride_lm,
             lse, mask=m_mask)


def warps_for_tile(tile_rows: int, *, min_warps: int = 1) -> int:
    """Pick ``num_warps`` from the TILE ROW COUNT, not from ``head_dim``.

    The kernels were launching ``4 if head_dim <= 64 else 8``, which is the wrong
    variable: at ``block_q=1`` the query tile is only 16 rows, so 8 warps (256 threads)
    puts 2 rows per warp and most of the MMA sits idle. Measured on a real pattern
    (seqlen 16384, topk 18, D=128):

        block_q=1  (tile 16):  fwd 8.17 / 10.23 / 11.60 / 14.21 ms at 1 / 2 / 4 / 8 warps
        block_q=64 (tile 64):  fwd 5.05 /  1.64 /  1.07 /  1.21 ms at 1 / 2 / 4 / 8 warps

    So the optimum tracks tile rows: one warp per 16 rows. That is also why the old
    ``head_dim`` rule looked fine -- it happened to give 8 warps for the 64-row tiles it
    was tuned on, and nobody re-tuned when ``block_q=1`` made the tile 16 rows.

    ``min_warps`` exists because ``dq`` wants 2 warps at a 16-row tile (12.29 vs 13.73):
    it carries more live state than the forward, so a little extra parallelism still
    helps. ``dk/dv`` is excluded entirely -- its tile is ``TILE_P x BLOCK_KV`` with a long
    gathered segment loop, and it prefers 8 warps (11.23 vs 43.84 at one) to hide that
    latency.
    """
    return max(min_warps, min(8, max(1, tile_rows // 16)))


def sparse_attn_fwd(
    q: torch.Tensor,               # [B, Hq, Sq, D]
    k: torch.Tensor,               # [B, Hkv, Skv, D]
    v: torch.Tensor,               # [B, Hkv, Skv, D]
    pattern: SparsePattern,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention forward. Returns ``(out, lse)``.

    ``lse`` is what makes the backward possible without saving the probabilities:
    the backward recomputes ``p = exp(qk - lse)`` per block.
    """
    b, hq, sq, d = q.shape
    hkv = k.shape[1]
    assert hq % hkv == 0
    assert d == k.shape[3] == v.shape[3], "head_dim must match across q/k/v"
    assert d in (16, 32, 64, 128, 256), f"unsupported head_dim {d}"
    assert pattern.num_kv_heads == hkv, (
        f"pattern has {pattern.num_kv_heads} kv heads, tensors have {hkv}"
    )
    assert pattern.num_q_blocks == triton.cdiv(sq, pattern.block_q)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device=q.device)

    group = hq // hkv
    # Pack query heads of a group into one program when `block_q` alone gives a thin
    # tile. This kernel is HBM-bound on K/V (at block_q=1, topk=8, D=128 each program
    # loads 256 KB to serve ONE query row -- 128 GB of traffic at 16k vs 2 GB at
    # block_q=64), and the whole group shares one selection, so `pack` heads can share
    # a single load. Capped at `group` because selection is only shared within a group,
    # and chosen so the tile reaches 16 rows: the MMA wants that anyway.
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
    tile_q = max(triton.next_power_of_2(pack * pattern.block_q), 16)

    grid = (pattern.num_q_blocks, hq // pack, b)
    _fwd_kernel[grid](
        q, k, v, out, lse,
        pattern.cnt, pattern.idx,
        scale,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
        *pattern.cnt.stride(), *pattern.idx.stride(),
        sq, k.shape[2],
        GROUP=group,
        BLOCK_Q=pattern.block_q,
        BLOCK_KV=pattern.block_kv,
        HEAD_DIM=d,
        MAX_SEL=pattern.max_selected,
        CAUSAL=causal,
        PACK_G=pack,
        TILE_Q=tile_q,
        num_warps=warps_for_tile(tile_q),
        num_stages=2,
    )
    return out, lse
