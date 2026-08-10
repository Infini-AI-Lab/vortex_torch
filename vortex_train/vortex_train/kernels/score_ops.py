"""Device functions for the scorer's memory-touching ops.

Kept in a normal module (rather than inside the generated kernel text) so they are
reviewable as ordinary code and are compiled from a real file, which
``@triton.jit`` requires. The generated kernel in :mod:`vortex_train.kernels.select`
imports these; only the *chain* is generated, not the op bodies.

Adding an op to ``flow/ops.py`` that reads memory means adding a function here and
one emit line in ``select._emit_body``. Ops that are pure arithmetic on existing
scores need neither — they are one emit line.

**Everything here works on a TILE of KV blocks, not all of them.** The state tile a
program holds is ``[TILE_N, HEAD_DIM]`` fp32; sizing it by the full
``next_pow2(Nkv)`` instead would be 512 KB per program at 64k tokens, which spills
to local memory and cost 43-72% of the step before this was tiled. See
``select._TEMPLATE``.

**Sub-blocks.** A field may keep several summaries per KV block. Scoring maxes over
them: one relevant sub-region is enough to select a block, which is the point of
keeping sub-block state at all. Each call site is specialised to *its own field's*
sub-count (a compile-time constant), so the padding slots that exist when fields
disagree on ``sub_block`` are never read.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def q_summary(
    Q,
    stride_qb, stride_qh, stride_qm, stride_qd,
    pid_b, pid_h, offs_m, offs_d, m_mask, cnt_m,
    GROUP: tl.constexpr, BLOCK_Q: tl.constexpr, HEAD_DIM: tl.constexpr,
    Q_HOW: tl.constexpr,
):
    """Collapse a query block to one vector per GQA group member -> ``[GROUP, D]``.

    Hoisted out of the KV-tile loop deliberately: ``q`` is the same for every tile,
    so loading it per tile would re-read ``BLOCK_Q x D`` once per tile (256 KB of
    redundant traffic per program at 64k). ``[GROUP, D]`` is ~2 KB and stays live
    across the loop.
    """
    qs = tl.zeros((GROUP, HEAD_DIM), dtype=tl.float32)
    for g in tl.static_range(GROUP):
        hq = pid_h * GROUP + g
        qt = tl.load(
            Q + pid_b * stride_qb + hq * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.float32)
        if Q_HOW == 0:                                  # mean over the block
            red = tl.sum(qt, axis=0) / cnt_m
        else:                                           # max
            red = tl.max(tl.where(m_mask[:, None], qt, float("-inf")), axis=0)
        # Place `red` in row g. A static index keeps this in registers.
        qs += tl.where((tl.arange(0, GROUP) == g)[:, None], red[None, :], 0.0)
    return qs


@triton.jit
def _group_of(QS, g, GROUP: tl.constexpr):
    """Row ``g`` of the ``[GROUP, D]`` query summary, as ``[D]``."""
    return tl.sum(tl.where((tl.arange(0, GROUP) == g)[:, None], QS, 0.0), axis=0)


@triton.jit
def score_dot(
    QS, STATE,
    stride_sb, stride_sh, stride_sn, stride_sf, stride_su, stride_sd,
    pid_b, pid_h, offs_n, offs_d, valid_n,
    FSLOT: tl.constexpr, GRED: tl.constexpr,
    GROUP: tl.constexpr, TILE_N: tl.constexpr, NSUB: tl.constexpr,
):
    """``<q_summary, state[FSLOT]>`` per KV block, over group and sub-blocks.

    ``GRED``: 0 = max, 1 = mean, 2 = sum, applied across the GQA group. Max is the
    frontend default because selection is *shared* across the group — a block any
    head in the group wants badly should survive, where averaging lets one
    strongly-interested head be outvoted.

    Sub-blocks are always reduced by **max**, regardless of ``GRED``: they are
    alternative locations within one block, not alternative opinions about it, so
    the block's score is that of its best-matching region.

    Sub-block loop is **outer** so each state tile is loaded once and reused across
    the group; see :func:`score_envelope` for what the other order costs.
    """
    if GRED == 0:                      # max over group: commutes with the sub-max
        acc = tl.full((TILE_N,), float("-inf"), dtype=tl.float32)
        for u in tl.static_range(NSUB):
            st = tl.load(
                STATE + pid_b * stride_sb + pid_h * stride_sh
                + offs_n[:, None] * stride_sn + FSLOT * stride_sf + u * stride_su
                + offs_d[None, :] * stride_sd,
                mask=valid_n[:, None], other=0.0,
            )
            for g in tl.static_range(GROUP):
                d = tl.sum(st * _group_of(QS, g, GROUP)[None, :], axis=1)
                acc = d if (u == 0 and g == 0) else tl.maximum(acc, d)
        return acc

    # sum / mean over the group: each member contributes its own best sub-block, so
    # the group reduction cannot be folded into the sub-block max.
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    for g in tl.static_range(GROUP):
        qg = _group_of(QS, g, GROUP)
        gbest = tl.full((TILE_N,), float("-inf"), dtype=tl.float32)
        for u in tl.static_range(NSUB):
            st = tl.load(
                STATE + pid_b * stride_sb + pid_h * stride_sh
                + offs_n[:, None] * stride_sn + FSLOT * stride_sf + u * stride_su
                + offs_d[None, :] * stride_sd,
                mask=valid_n[:, None], other=0.0,
            )
            d = tl.sum(st * qg[None, :], axis=1)
            gbest = d if u == 0 else tl.maximum(gbest, d)
        acc += gbest
    if GRED == 1:
        acc = acc / GROUP
    return acc


@triton.jit
def score_envelope(
    QS, STATE,
    stride_sb, stride_sh, stride_sn, stride_sf, stride_su, stride_sd,
    pid_b, pid_h, offs_n, offs_d, valid_n,
    FMAX: tl.constexpr, FMIN: tl.constexpr, GRED: tl.constexpr,
    GROUP: tl.constexpr, TILE_N: tl.constexpr, NSUB: tl.constexpr,
):
    """The exact QUEST bound ``sum_d max(q_d*M_d, q_d*m_d)``, per KV block.

    The per-channel ``max`` happens **before** the reduction over ``D``. That
    ordering is the whole point: ``Dot(kmax) + Dot(kmin)`` collapses ``D`` first and
    so cannot make the per-channel endpoint choice, giving a different and looser
    quantity. Doing it here keeps the bound tight, which is what makes it a
    meaningful upper bound on ``max_k <q,k>`` within the block.

    **Loop order is sub-block OUTER, group INNER**, so each state tile is loaded once
    and reused across the group. The natural order (group outer) reloads the same
    ``kmax``/``kmin`` once per group member: at ``GROUP=4, NSUB=4`` that is 32 loads
    where 8 suffice, and it made selection 20x quest's cost (305 ms vs 15 ms at 128k)
    for only 4x the state -- the state traffic, not the arithmetic, is the whole cost
    of this kernel.
    """
    acc = tl.full((TILE_N,), float("-inf"), dtype=tl.float32)
    for u in tl.static_range(NSUB):
        base = (STATE + pid_b * stride_sb + pid_h * stride_sh
                + offs_n[:, None] * stride_sn + u * stride_su
                + offs_d[None, :] * stride_sd)
        kmax = tl.load(base + FMAX * stride_sf, mask=valid_n[:, None], other=0.0)
        kmin = tl.load(base + FMIN * stride_sf, mask=valid_n[:, None], other=0.0)
        for g in tl.static_range(GROUP):
            qg = _group_of(QS, g, GROUP)
            d = tl.sum(tl.maximum(qg[None, :] * kmax, qg[None, :] * kmin), axis=1)
            # `acc` folds group and sub-block together. Sub-blocks always reduce by
            # max, and when GRED is also max the two commute, so one accumulator is
            # enough. For sum/mean the group reduction must not double-count across
            # sub-blocks, so those keep a per-group best in `gbest`.
            if GRED == 0:
                acc = d if (u == 0 and g == 0) else tl.maximum(acc, d)
    if GRED != 0:
        # sum / mean over the group, each member's own best sub-block
        acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for g in tl.static_range(GROUP):
            qg = _group_of(QS, g, GROUP)
            gbest = tl.full((TILE_N,), float("-inf"), dtype=tl.float32)
            for u in tl.static_range(NSUB):
                base = (STATE + pid_b * stride_sb + pid_h * stride_sh
                        + offs_n[:, None] * stride_sn + u * stride_su
                        + offs_d[None, :] * stride_sd)
                kmax = tl.load(base + FMAX * stride_sf, mask=valid_n[:, None], other=0.0)
                kmin = tl.load(base + FMIN * stride_sf, mask=valid_n[:, None], other=0.0)
                d = tl.sum(tl.maximum(qg[None, :] * kmax, qg[None, :] * kmin), axis=1)
                gbest = d if u == 0 else tl.maximum(gbest, d)
            acc += gbest
        if GRED == 1:
            acc = acc / GROUP
    return acc


@triton.jit
def score_norm(
    STATE,
    stride_sb, stride_sh, stride_sn, stride_sf, stride_su, stride_sd,
    pid_b, pid_h, offs_n, offs_d, valid_n,
    FSLOT: tl.constexpr, TILE_N: tl.constexpr, NSUB: tl.constexpr,
):
    """L2 norm of state field ``FSLOT``, max'd over sub-blocks, per KV block."""
    best = tl.full((TILE_N,), float("-inf"), dtype=tl.float32)
    for u in tl.static_range(NSUB):
        st = tl.load(
            STATE + pid_b * stride_sb + pid_h * stride_sh
            + offs_n[:, None] * stride_sn + FSLOT * stride_sf + u * stride_su
            + offs_d[None, :] * stride_sd,
            mask=valid_n[:, None], other=0.0,
        )
        n = tl.sqrt(tl.sum(st * st, axis=1))
        best = n if u == 0 else tl.maximum(best, n)
    return best
