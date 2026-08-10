"""Transpose the pattern: q-major -> kv-major, on device, in Triton.

The forward needs "which KV blocks does query block m attend?" (q-major, what the
scorer naturally produces). ``dK``/``dV`` need the opposite: "which query blocks
selected KV block n?" — because a `dk` tile is a sum over the query blocks that used it.

Why this is a real problem and not a reshape: the relation is *ragged in the other
direction*. Each query block selects a fixed `cnt` blocks, but a KV block is selected by
an unpredictable number of query blocks.

**Segment order is load-bearing, not cosmetic.** dk/dv accumulate over a segment in fp32,
so a permuted segment changes the summation order and gradients differ in the last bits
(measured ~4e-5 relative at ``block_q=1``, where segments reach hundreds of entries).
Without a fixed order, identical inputs give different gradients and a regression cannot
be told apart from run-to-run noise.

Four kernels, all O(nnz), deterministic **by construction**:

    1. count       — per (kv block, query CHUNK) histogram
    2. scan        — exclusive prefix over kv blocks -> CSR offsets
    3. chunk_scan  — exclusive prefix over chunks *within* each kv block
    4. scatter     — each chunk fills its own reserved slice, ascending by query block

Step 3 is what removes the sort. A program owning a contiguous range of query blocks
emits its entries in increasing query order anyway; the chunk prefix tells it where that
range starts inside the segment. So the whole segment comes out ascending with **no
atomic on any position and no sort**.

Two earlier designs, both measured and rejected:

* an ``atomic_add`` cursor (race order) plus an O(len²) rank sort afterwards. Correct,
  but the sort cost **37 ms at block_q=1, seqlen 16k — 45% of the entire step**, because
  segment length scales as 1/block_q and the sort is quadratic in it. It had only been
  benchmarked at block_q=64, where segments are 64x shorter and it looked free (0.1 ms).
  That is the measurement mistake, not the algorithm choice: the cost was invisible at
  the shape it was validated at.
* a rank scan over the whole query axis, O(Mq²): 1.7e10 loads at block_q=1.

Flash-Sparse-Attention solves the same problem by scattering the query index into a dense
``[Nkv, Mq]`` grid and compacting, so position *is* the query index — elegant, and O(nnz)
in work, but the grid is O(T²) in memory (16 MB per head at 16k, ~8 GB at 128k), so the
chunked CSR below is used instead.

Result is CSR over KV blocks: ``t_offsets [B, Hkv, Nkv+1]`` and
``t_indices [B, Hkv, nnz]`` int32 of query-block ids, each segment strictly ascending.
``nnz = Mq * K`` is known from shapes, so no allocation depends on a device value.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..pattern import SparsePattern

#: query blocks per scatter program — the granularity of the chunk prefix.
BLOCK_M = 64


@triton.jit
def _count_kernel(
    IDX, CNT, COUNTS,
    stride_ib, stride_ih, stride_im, stride_ik,
    stride_cb, stride_ch, stride_cm,
    stride_ub, stride_uh, stride_un, stride_uc,
    num_q_blocks,
    MAX_SEL: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """``COUNTS[b,h,n,chunk] = |{(m,slot) in chunk : idx[m,slot] == n}|``.

    Per *chunk*, not just per kv block: that extra axis is what lets the scatter place
    entries without an atomic. The atomic here accumulates a **count**, not a position,
    so its result does not depend on arrival order.
    """
    pid_m = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_q_blocks
    cnt = tl.load(CNT + pid_b * stride_cb + pid_h * stride_ch + offs_m * stride_cm,
                  mask=m_mask, other=0)
    base = COUNTS + pid_b * stride_ub + pid_h * stride_uh + pid_m * stride_uc
    for s in range(0, MAX_SEL):
        kv = tl.load(
            IDX + pid_b * stride_ib + pid_h * stride_ih
            + offs_m * stride_im + s * stride_ik,
            mask=m_mask & (s < cnt), other=-1,
        )
        ok = m_mask & (s < cnt) & (kv >= 0)
        tl.atomic_add(base + kv * stride_un, 1, mask=ok)


@triton.jit
def _scan_kernel(
    COUNTS, OFFSETS,
    stride_ub, stride_uh, stride_un, stride_uc,
    stride_ob, stride_oh, stride_on,
    num_kv_blocks, num_chunks,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Sum chunks -> segment totals -> exclusive prefix over kv blocks -> CSR offsets.

    One program per (batch, kv-head). ``tl.cumsum`` over a single ``BLOCK_N`` tile: the
    kv-block count is ``seqlen/block_kv`` (~2048 at 128k), so one tile suffices and a
    multi-pass scan would only add launches.
    """
    pid_h = tl.program_id(0); pid_b = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)
    n_mask = offs_n < num_kv_blocks
    per = tl.load(
        COUNTS + pid_b * stride_ub + pid_h * stride_uh
        + offs_n[:, None] * stride_un + offs_c[None, :] * stride_uc,
        mask=n_mask[:, None] & (offs_c < num_chunks)[None, :], other=0,
    )
    totals = tl.sum(per, axis=1)
    incl = tl.cumsum(totals, axis=0)
    base = OFFSETS + pid_b * stride_ob + pid_h * stride_oh
    tl.store(base + offs_n * stride_on, incl - totals, mask=n_mask)
    tl.store(base + num_kv_blocks * stride_on, tl.sum(totals, axis=0))


@triton.jit
def _chunk_scan_kernel(
    COUNTS, CHUNK_OFF,
    stride_ub, stride_uh, stride_un, stride_uc,
    num_chunks,
    BLOCK_C: tl.constexpr,
):
    """Exclusive prefix over chunks *within* one kv block.

    ``CHUNK_OFF[b,h,n,c]`` = how many entries of kv block ``n`` come from chunks before
    ``c``. Added to the segment base, that is precisely where chunk ``c`` may write — so
    the scatter needs no atomic, and because chunks are ordered by query index the
    segment comes out ascending.
    """
    pid_n = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < num_chunks
    row = pid_b * stride_ub + pid_h * stride_uh + pid_n * stride_un + offs_c * stride_uc
    c = tl.load(COUNTS + row, mask=c_mask, other=0)
    tl.store(CHUNK_OFF + row, tl.cumsum(c, axis=0) - c, mask=c_mask)


@triton.jit
def _scatter_kernel(
    IDX, CNT, OFFSETS, CHUNK_OFF, T_IND,
    stride_ib, stride_ih, stride_im, stride_ik,
    stride_cb, stride_ch, stride_cm,
    stride_ob, stride_oh, stride_on,
    stride_ub, stride_uh, stride_un, stride_uc,
    stride_tb, stride_th, stride_tn,
    num_q_blocks,
    MAX_SEL: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Write each chunk's entries into its reserved slice, ascending by query block.

    No atomics. ``OFFSETS[n] + CHUNK_OFF[n, chunk]`` is this chunk's exclusive start in
    segment ``n``; within the chunk an entry's position is its rank under the total order
    (query block, slot). Both ranks are exclusive counts over the tile, so the segment
    ends up sorted by construction.
    """
    pid_m = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_q_blocks
    cnt = tl.load(CNT + pid_b * stride_cb + pid_h * stride_ch + offs_m * stride_cm,
                  mask=m_mask, other=0)

    idx_hb = IDX + pid_b * stride_ib + pid_h * stride_ih
    off_hb = OFFSETS + pid_b * stride_ob + pid_h * stride_oh
    cok_hb = CHUNK_OFF + pid_b * stride_ub + pid_h * stride_uh + pid_m * stride_uc
    out_hb = T_IND + pid_b * stride_tb + pid_h * stride_th
    lanes = tl.arange(0, BLOCK_M)

    for s in range(0, MAX_SEL):
        kv = tl.load(idx_hb + offs_m * stride_im + s * stride_ik,
                     mask=m_mask & (s < cnt), other=-1)
        ok = m_mask & (s < cnt) & (kv >= 0)

        # Position within this chunk's slice = how many entries of the same kv block
        # come from a STRICTLY EARLIER QUERY BLOCK in this tile.
        #
        # `validate()` forbids a row from naming the same kv block twice, so at most one
        # slot per lane can match a given kv -- which means counting *rows* is a total
        # order and no slot tie-break is needed. Ranking per-slot instead (an earlier
        # attempt) collided whenever two rows selected the same block at different
        # slots: both got position 0 and the segment came out [3, 2] instead of [2, 3].
        rank = tl.zeros((BLOCK_M,), dtype=tl.int32)
        for t in range(0, MAX_SEL):
            kv_t = tl.load(idx_hb + offs_m * stride_im + t * stride_ik,
                           mask=m_mask & (t < cnt), other=-1)
            ok_t = m_mask & (t < cnt) & (kv_t >= 0)
            hit = (kv[:, None] == kv_t[None, :]) & ok_t[None, :]
            rank += tl.sum(
                tl.where(hit & (lanes[None, :] < lanes[:, None]), 1, 0).to(tl.int32),
                axis=1)

        seg_base = tl.load(off_hb + kv * stride_on, mask=ok, other=0)
        chunk_base = tl.load(cok_hb + kv * stride_un, mask=ok, other=0)
        tl.store(out_hb + (seg_base + chunk_base + rank) * stride_tn,
                 offs_m.to(T_IND.dtype.element_ty), mask=ok)


def transpose_pattern(pattern: SparsePattern) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(t_offsets, t_indices)`` — CSR of the pattern over KV blocks.

    ``t_offsets[b, h, n] : t_offsets[b, h, n+1]`` slices ``t_indices`` to the query blocks
    that selected KV block ``n``, ascending.

    Four launches, no sort, no host sync; every allocation is shape-derived, so the path
    stays cudagraph-safe.
    """
    b, hkv, m = pattern.cnt.shape
    n_kv = pattern.num_kv_blocks
    dev = pattern.idx.device
    nnz = m * pattern.max_selected              # static bound from shapes
    n_chunks = triton.cdiv(m, BLOCK_M)

    counts = torch.zeros((b, hkv, n_kv, n_chunks), dtype=torch.int32, device=dev)
    chunk_off = torch.empty_like(counts)
    t_offsets = torch.empty((b, hkv, n_kv + 1), dtype=torch.int32, device=dev)
    # -1-filled, not `empty`: nothing reads past `t_offsets[..., -1]`, but allocator
    # garbage in the tail would make the returned tensor differ run to run, defeating a
    # bitwise-reproducibility check on the artifact itself.
    t_indices = torch.full((b, hkv, nnz), -1, dtype=torch.int32, device=dev)

    grid_m = (n_chunks, hkv, b)
    _count_kernel[grid_m](
        pattern.idx, pattern.cnt, counts,
        *pattern.idx.stride(), *pattern.cnt.stride(), *counts.stride(),
        m, MAX_SEL=pattern.max_selected, BLOCK_M=BLOCK_M,
    )
    _scan_kernel[(hkv, b)](
        counts, t_offsets,
        *counts.stride(), *t_offsets.stride(),
        n_kv, n_chunks,
        BLOCK_N=triton.next_power_of_2(n_kv),
        BLOCK_C=triton.next_power_of_2(n_chunks),
    )
    _chunk_scan_kernel[(n_kv, hkv, b)](
        counts, chunk_off, *counts.stride(),
        n_chunks, BLOCK_C=triton.next_power_of_2(n_chunks),
    )
    # NOTE: t_offsets is passed whole (its own strides), NOT a [..., :-1] copy. A sliced
    # .contiguous() copy has a different last-dim extent and therefore a different stride
    # than the tensor the dk/dv kernel reads, which silently mis-addresses the bases.
    _scatter_kernel[grid_m](
        pattern.idx, pattern.cnt, t_offsets, chunk_off, t_indices,
        *pattern.idx.stride(), *pattern.cnt.stride(),
        *t_offsets.stride(), *chunk_off.stride(), *t_indices.stride(),
        m, MAX_SEL=pattern.max_selected, BLOCK_M=BLOCK_M,
    )
    return t_offsets, t_indices


def verify_transpose(pattern: SparsePattern, t_offsets, t_indices) -> None:
    """Assert the transpose is exactly the inverse relation. **Test-only.**

    The highest-value assertion in the project: a wrong transpose *silently drops
    gradient*. Loss still goes down, so nothing looks broken — it just trains to a
    different objective. Uses host loops deliberately: being obviously correct matters
    more than speed in a checker.
    """
    b, hkv, _ = pattern.cnt.shape
    ar = torch.arange(pattern.max_selected, device=pattern.idx.device, dtype=torch.int32)
    valid = ar[None, None, None, :] < pattern.cnt[..., None]

    for bi in range(b):
        for hi in range(hkv):
            assert int(t_offsets[bi, hi, -1]) == int(valid[bi, hi].sum()), (
                f"(b={bi},h={hi}) CSR total != number of valid selections"
            )
            for n in range(pattern.num_kv_blocks):
                lo, hi_ = int(t_offsets[bi, hi, n]), int(t_offsets[bi, hi, n + 1])
                seg = t_indices[bi, hi, lo:hi_]
                got = set(seg.tolist())
                want = set(
                    torch.nonzero((pattern.idx[bi, hi] == n) & valid[bi, hi])[:, 0].tolist()
                )
                assert got == want, (
                    f"kv block {n}: transpose {sorted(got)} != pattern {sorted(want)}"
                )
                if seg.numel() > 1:
                    assert bool((seg.diff() > 0).all()), (
                        f"kv block {n} segment is not ascending: {seg[:12].tolist()}"
                    )
