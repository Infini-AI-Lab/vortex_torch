"""INT4 staging arena: hold in-flight blocks in bf16, quantize once on completion.

The problem this exists to solve
-------------------------------
A block's scale is a *reduction over the block*: K's is an absmax over the block's tokens, V's an
absmax over channels. So a block that is still being appended to has **no scale yet**, and cannot
be stored quantized -- quantizing with a partial scale and re-quantizing when the block grows would
compound error on every token, and would also mean re-reading and re-writing the whole block per
decode step.

The answer is **two TENSORS, not two dtypes**. ``flow.py`` assigns a dtype per FIELD, so a bf16
staging area, the uint8 payload and the fp32 scales coexist in one cache dict with no notion of a
mixed-dtype tensor anywhere. A block lives in bf16 while it is in flight, is quantized exactly once
from never-quantized values the moment it completes, and lives packed thereafter.

Sized by CONCURRENCY, not by context
------------------------------------
There is at most one in-flight block per (request, kv head), so the arena is
``capture_ceiling x kv_heads`` slots and its footprint is **constant in context length** (measured
identical at 4 blocks and at 4000). The alternative -- a per-block bf16 mirror -- was implemented
and measured at **21632 B/block against bf16's own 16896, a 1.28x REGRESSION**: it reintroduces
exactly the bytes INT4 removes. That is the single most important thing not to redo here.

Two sizings were also tried and both gave **0% accuracy**: from ``max_running_requests`` and from
``req_to_token_pool.size``. Both undersize the arena, and starvation is silent -- a token whose
claim finds no slot is simply dropped. Hence :data:`CAPTURE_CEILING` being a hardcoded constant
with the cudagraph reason attached, and hence ``STAT_DECLINED``.

Keys are BLOCK IDS, not request slots
-------------------------------------
Indexing by request slot is the obvious design and it does not work: ``set_kv_buffer`` is the only
hook where incoming K/V is visible and it is passed no request id. ``req_to_token`` maps
``(req_slot, pos) -> kv_index``, so ``loc`` holds that map's *values*; recovering the slot means
searching the map plus a host sync, which cudagraph capture forbids outright. Block ids are already
globally unique from sglang's page pool, so they are the key.

Why the fused write path is DECODE-ONLY
---------------------------------------
Migration reads the whole staged block, so it may only run in a program that can *see* every
token of that block. Across a kernel boundary that is free; within one launch it is not, and adding
a fence would not help because the writers are in different programs.

* **Decode** contributes at most one token per block per launch, so every other token of the block
  was written by an *earlier launch* and is visible. Migration fuses into the stage kernel:
  measured **2.04 -> 1.29 ms/step (1.6x)**, which is almost all launch overhead -- the floor is
  0.0116 ms against 0.022 per kernel, so half the write was launches.
* **Prefill** writes many tokens of one block in a single launch, so it must stage first and
  migrate in a second kernel.

This is passed in as ``fused=`` rather than inferred. Inferring it from shapes was tried
(``fused = n_tok <= num_kv_heads``) and silently disabled the fusion for every realistic batch --
a performance bug that no test could see because both paths are correct.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from ...cache.triton_kernels.int4_kv import INT4_BIAS, INT4_QMAX
from .request_domain import RequestCacheDomain

logger = logging.getLogger(__name__)

_QMAX = tl.constexpr(INT4_QMAX)
_BIAS = tl.constexpr(INT4_BIAS)

#: Concurrent block-table rows the cudagraph capture is sized for. Hardcoded rather than derived:
#: the arena must be allocated before capture (a first-use allocation inside capture is *captured
#: into the graph*), and the two derivations that were tried both undersized it into silent
#: starvation. At 32x128 bf16 per slot x kv_heads this is ~1.2 GB on a 36-layer model -- a real
#: cost, not waste, and small against the KV it makes room for.
CAPTURE_CEILING = 256

#: ``stats`` slots. Only slow paths are counted, so this costs nothing per token.
STAT_CLAIM = 0        # a block acquired a slot
STAT_DECLINED = 1     # NO FREE SLOT -- the token was DROPPED. Must be zero; see the module docstring
STAT_MIGRATED = 2     # a block completed and was quantized into the page cache
STAT_SPUN = 3         # waited for another program's in-progress claim (normal at prefill width)
STAT_SPIN_EXHAUSTED = 4   # gave up waiting -- token DROPPED. Must be zero; see :data:`MAX_SPIN`
N_STATS = 5

#: Spin bound for a program waiting on another program's in-progress claim. The wait is provably
#: short -- the claimant is already resident (it executed the winning ``atomic_cas``) and its probe
#: loop depends on nothing a waiter does, so it always publishes -- and this bound exists only so
#: that a bug elsewhere degrades into a counted drop instead of a hung kernel. Measured worst case
#: at prefill width (8192 programs, 512 slots) was under 200 iterations.
MAX_SPIN = 1 << 16

#: Kernel-readable copies of the stat indices. A @jit'ed function cannot read a plain module
#: global, and the ANNOTATED form (``x: tl.constexpr = 0``) is explicitly unsupported -- it must be
#: the ``tl.constexpr(...)`` CALL form. Same trap as ``_QMAX`` in int4_kv.py.
_S_CLAIM = tl.constexpr(STAT_CLAIM)
_S_DECLINED = tl.constexpr(STAT_DECLINED)
_S_MIGRATED = tl.constexpr(STAT_MIGRATED)
_S_SPUN = tl.constexpr(STAT_SPUN)
_S_EXHAUST = tl.constexpr(STAT_SPIN_EXHAUSTED)

#: ``slot_of`` sentinels. ``-2`` = a claim is in progress; see :func:`_claim`.
_FREE = tl.constexpr(-1)
_CLAIMING = tl.constexpr(-2)
_MAX_SPIN = tl.constexpr(MAX_SPIN)


def arena_slots(num_kv_heads: int) -> int:
    """Slots to allocate: one in-flight block per (row, kv head) at the capture ceiling."""
    return CAPTURE_CEILING * max(1, num_kv_heads)


@triton.jit
def _claim(SLOT_OF, OWNER_OF, STATS, blk, n_slots):
    """Return this block's staging slot, claiming one if it has none. ``-1`` = arena full.

    Arbitration is on the BLOCK, not on the slot pool
    -------------------------------------------------
    Exactly one program per block wins ``atomic_cas(slot_of + blk, -1, -2)`` and is the only one
    that probes for a slot; everyone else waits for it to publish. The obvious alternative -- let
    every program grab a slot optimistically and give it back on losing the race for
    ``slot_of[blk]`` -- is correct at decode width and **breaks at prefill width**, which is how it
    was found rather than reasoned:

        64 programs, 8 slots:  claims 2, lost_races 11, DECLINED 51, migrated 0

    Every program transiently held a slot, so the pool looked full to everyone still probing and
    51 tokens were **dropped**. The failure is not a shortage -- 2 slots were needed -- it is that
    optimistic claiming makes transient demand equal to the program count. Here at most one program
    per block ever holds a slot, so demand equals the number of distinct in-flight blocks, which is
    what the arena is sized for.

    The spin is safe **only because it never waits on a slot**. A "wait for a slot to free up" loop
    would deadlock: the program that would free one may be behind this one in the launch order, and
    there is no preemption. Waiting on a claim in progress has no such cycle -- the claimant is
    already resident and its probe depends on nothing a waiter does.

    ``num_warps=1`` is mandatory in any kernel calling this. Triton's scalar atomics are
    single-laned per WARP, so with more warps each warp of one program runs the loop independently
    and two of them win two different slots for the same block.

    Probing starts at ``blk % n_slots`` rather than 0 so claims for different blocks do not all
    contend on slot 0.
    """
    # Both sentinels and the map are int32. Triton unifies a variable's type across a branch, so a
    # plain ``mine = -1`` (int32) later assigned an int64 is a compile error -- the good outcome;
    # the bad one is a silent widening that then disagrees with the int32 map on a cas.
    s = tl.atomic_cas(SLOT_OF + blk, _FREE, _CLAIMING)
    if s == _FREE:
        # This program owns the claim for ``blk``. Free slot == ``owner == -1`` AND ``mask == 0``;
        # the release path maintains that pairing, which is why claiming never touches the mask.
        mine = tl.full((), -1, tl.int32)
        start = (blk % n_slots).to(tl.int32)
        for j in tl.range(0, n_slots, num_stages=1):
            if mine < 0:
                p = (start + j.to(tl.int32)) % n_slots.to(tl.int32)
                if tl.atomic_cas(OWNER_OF + p, -1, blk) == -1:
                    mine = p
        # Publish unconditionally, including the failure: leaving ``_CLAIMING`` behind would spin
        # every waiter for this block to exhaustion.
        tl.atomic_xchg(SLOT_OF + blk, mine)
        if mine < 0:
            tl.atomic_add(STATS + _S_DECLINED, 1)
        else:
            tl.atomic_add(STATS + _S_CLAIM, 1)
        s = mine
    elif s == _CLAIMING:
        tl.atomic_add(STATS + _S_SPUN, 1)
        for _ in tl.range(0, _MAX_SPIN, num_stages=1):
            if s == _CLAIMING:
                # atomic, not tl.load: a plain load can be hoisted or cached out of the loop, and
                # the spin would then never observe the publish.
                s = tl.atomic_add(SLOT_OF + blk, 0)
        if s == _CLAIMING:
            tl.atomic_add(STATS + _S_EXHAUST, 1)
            s = tl.full((), -1, tl.int32)
    return s


@triton.jit
def _migrate(STAGE_K, STAGE_V, PK, PV, SCALE_K, SCALE_V, SLOT_OF, OWNER_OF, MASK_OF, STATS,
             slot, blk,
             HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
             BLOCK_T: tl.constexpr, N_TOK: tl.constexpr):
    """Quantize one complete staged block into the page cache and free its slot.

    The load is one contiguous ``[T, HEAD_DIM]`` read and the pack uses ``tl.split`` -- the exact
    mirror of the read path's ``tl.join``, and for the same reason: the alternative is two stride-2
    accesses in which every transaction carries half-useful bytes.

    Ordering needs no fence. Every consumer of the packed bytes or of ``slot_of`` runs in a *later
    kernel*, so all of this program's stores are complete before any of them can observe anything.
    Within this launch the only way to observe the half-released state would be a second write to a
    block that has already completed, which sglang does not do: a page position is written exactly
    once per allocation (prefix reuse *references* a page, it does not rewrite it).
    """
    t = tl.arange(0, BLOCK_T)
    d = tl.arange(0, HEAD_DIM)
    dh = tl.arange(0, HALF_DIM)
    tmask = t < N_TOK
    m = tmask[:, None]

    # int64 offsets. ``blk`` is int32 to match the maps, but a large host-KV pool times the block
    # numel overflows int32 (at 32x128 that is 1.05M blocks, which a multi-hundred-GB host cache
    # reaches), and the wrap would address a valid-looking wrong block.
    slot = slot.to(tl.int64)
    blk64 = blk.to(tl.int64)
    src = slot * N_TOK * HEAD_DIM + t[:, None] * HEAD_DIM + d[None, :]
    xk = tl.load(STAGE_K + src, mask=m, other=0.0).to(tl.float32)
    xv = tl.load(STAGE_V + src, mask=m, other=0.0).to(tl.float32)

    # K per-CHANNEL (max over the block's tokens), V per-TOKEN (max over channels). The axes differ
    # between the two tensors and that asymmetry is measured, not stylistic -- see int4_kv.py: one
    # axis for both is the obvious implementation and costs the most (attn-out err 0.510 vs 0.147).
    ks = tl.maximum(tl.max(tl.abs(xk), axis=0) / _QMAX, 1e-8)          # [HEAD_DIM]
    vs = tl.maximum(tl.max(tl.abs(xv), axis=1) / _QMAX, 1e-8)          # [BLOCK_T]

    qk = tl.extra.cuda.libdevice.round(xk / ks[None, :])
    qv = tl.extra.cuda.libdevice.round(xv / vs[:, None])
    qk = tl.minimum(tl.maximum(qk, -_QMAX), _QMAX).to(tl.int32) + _BIAS
    qv = tl.minimum(tl.maximum(qv, -_QMAX), _QMAX).to(tl.int32) + _BIAS

    klo, khi = tl.split(tl.reshape(qk, (BLOCK_T, HALF_DIM, 2)))
    vlo, vhi = tl.split(tl.reshape(qv, (BLOCK_T, HALF_DIM, 2)))
    pdst = blk64 * N_TOK * HALF_DIM + t[:, None] * HALF_DIM + dh[None, :]
    tl.store(PK + pdst, (klo | (khi << 4)).to(tl.uint8), mask=m)
    tl.store(PV + pdst, (vlo | (vhi << 4)).to(tl.uint8), mask=m)
    tl.store(SCALE_K + blk64 * HEAD_DIM + d, ks)
    tl.store(SCALE_V + blk64 * N_TOK + t, vs, mask=tmask)

    # Release. Zero the mask BEFORE dropping the owner, so a slot with ``owner == -1`` always has
    # ``mask == 0`` -- the invariant _claim relies on to skip touching the mask.
    tl.store(SLOT_OF + blk, -1)
    tl.store(MASK_OF + slot, 0)
    tl.store(OWNER_OF + slot, -1)
    tl.atomic_add(STATS + _S_MIGRATED, 1)


@triton.jit
def _stage_kernel(
    STAGE_K, STAGE_V, NEW_K, NEW_V, LOC,
    PK, PV, SCALE_K, SCALE_V, SLOT_OF, OWNER_OF, MASK_OF, STATS,
    n_tok, n_slots,
    NUM_KV_HEAD: tl.constexpr, HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
    FULL_MASK: tl.constexpr, FUSED_MIGRATE: tl.constexpr,
):
    """Write incoming bf16 K/V into the staging arena; optionally migrate on completion.

    One program per (token, kv head), mirroring ``set_kv.py``'s grid and reusing its
    block-interleaved position mapping verbatim -- the arena is keyed on the block id that mapping
    produces, so any divergence between the two would stage under a key nothing ever reads.
    """
    token_id = tl.program_id(0)
    if token_id >= n_tok:
        return
    head_id = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)

    pos = tl.load(LOC + token_id)
    trans = (pos // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + pos % PAGE_SIZE
    # int32 to match the maps' dtype (see _claim). The narrowing is safe for the block id -- an
    # int32 block index is 2G blocks -- while the byte offsets inside _migrate widen back to int64.
    blk = (trans // BLOCK_SIZE).to(tl.int32)
    row = (trans % BLOCK_SIZE).to(tl.int32)

    slot = _claim(SLOT_OF, OWNER_OF, STATS, blk, n_slots)
    if slot < 0:
        return                                  # arena full: counted in STAT_DECLINED

    src = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + d
    dst = (slot.to(tl.int64) * BLOCK_SIZE * HEAD_DIM
           + row.to(tl.int64) * HEAD_DIM + d)
    tl.store(STAGE_K + dst, tl.load(NEW_K + src))
    tl.store(STAGE_V + dst, tl.load(NEW_V + src))

    # One bit per row, so completion is observed by exactly one program: the one whose atomic_or
    # takes the mask from not-full to full. A count would double-fire if a row were ever written
    # twice, and a "did I write the last row" test assumes in-order arrival.
    bit = tl.full((), 1, tl.int64) << row.to(tl.int64)
    old = tl.atomic_or(MASK_OF + slot, bit)
    if FUSED_MIGRATE:
        if ((old | bit) == FULL_MASK) & (old != FULL_MASK):
            _migrate(STAGE_K, STAGE_V, PK, PV, SCALE_K, SCALE_V, SLOT_OF, OWNER_OF, MASK_OF, STATS,
                     slot, blk,
                     HEAD_DIM=HEAD_DIM, HALF_DIM=HALF_DIM,
                     BLOCK_T=BLOCK_T, N_TOK=BLOCK_SIZE)


@triton.jit
def _migrate_scan_kernel(
    STAGE_K, STAGE_V, PK, PV, SCALE_K, SCALE_V, SLOT_OF, OWNER_OF, MASK_OF, STATS,
    n_slots,
    HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr, FULL_MASK: tl.constexpr,
):
    """Migrate every complete block in the arena. The prefill counterpart of the fused path.

    Scans all slots unconditionally. Bounding the scan to the slots a step could have touched was
    implemented and measured: **flat ~0.023 ms from 256 to 2048 slots**, i.e. entirely launch-bound,
    so the bookkeeping bought nothing and only added a way to get the bound wrong.
    """
    slot = tl.program_id(0)
    if slot >= n_slots:
        return
    if tl.load(MASK_OF + slot) != FULL_MASK:
        return
    blk = tl.load(OWNER_OF + slot)
    if blk < 0:
        return
    _migrate(STAGE_K, STAGE_V, PK, PV, SCALE_K, SCALE_V, SLOT_OF, OWNER_OF, MASK_OF, STATS,
             slot, blk,
             HEAD_DIM=HEAD_DIM, HALF_DIM=HALF_DIM, BLOCK_T=BLOCK_T, N_TOK=BLOCK_SIZE)


class Int4Arena:
    """Per-layer staging arena for in-flight (partially written) INT4 blocks.

    A thin specialisation of :class:`~vortex_torch.engine.sgl.request_domain.RequestCacheDomain`:
    the domain owns the slot allocation and the key->slot map (which the compiler also addresses
    ``FORMAT.SLOTTED`` fields through), and this class adds only what is specific to *quantizing on
    completion* -- the completion bitmask, the counters, and the two kernels. Sharing the map rather
    than keeping a parallel copy is the point: two maps for one domain can silently disagree, and
    the SLOTTED codegen reads whichever one it was handed.

    The bf16 staging payload is NOT owned here; it lives in the cache dict as an ordinary
    request-domain field. That separation is deliberate -- an earlier version held ``arena.k`` /
    ``arena.v`` itself, and every consumer then had to know about the arena to find the newest
    tokens, so the read path broke the moment staging became a pool field.

    Policy is ``complete`` + ``drop``, and ``drop`` is not a default worth changing: evicting an
    incomplete block means quantizing one that was only partly written, which is silent data loss,
    whereas a drop is a counted loss of one token.
    """

    def __init__(self, num_blocks: int, n_slots: int, block_size: int, head_dim: int,
                 device: str = "cuda"):
        if block_size > 64:
            raise ValueError(
                f"INT4 staging tracks block completion as a bitmask in one int64, so "
                f"block_size must be <= 64, got {block_size}"
            )
        if head_dim % 2:
            raise ValueError(f"INT4 needs an even head_dim, got {head_dim}")
        self.domain = RequestCacheDomain(
            num_blocks, n_slots, retention="complete", overflow="drop", device=device,
        )
        self.n_slots = self.domain.n_slots
        self.block_size = int(block_size)
        self.head_dim = int(head_dim)
        self.full_mask = (1 << self.block_size) - 1
        # Every buffer is allocated up front, here and in the domain. A lazily-grown map would be
        # allocated during the first captured step and the allocation becomes part of the graph.
        self.mask_of = torch.zeros((self.n_slots,), dtype=torch.int64, device=device)
        self.stats = torch.zeros((N_STATS,), dtype=torch.int32, device=device)

    @property
    def slot_of(self) -> torch.Tensor:
        """The domain's key->slot map. Also what a SLOTTED field is addressed through, so a flow
        reading the staged block in a compiled path passes exactly this tensor."""
        return self.domain.slot_of

    @property
    def owner_of(self) -> torch.Tensor:
        return self.domain.owner_of

    def nbytes(self) -> int:
        return (self.domain.nbytes_maps()
                + sum(t.element_size() * t.numel() for t in (self.mask_of, self.stats)))

    def stage(self, stage_k, stage_v, new_k, new_v, loc, cache, page_size: int,
              *, fused: bool, k_scale_name: str, v_scale_name: str) -> None:
        """Stage a step's K/V, and (``fused=True``, decode only) migrate blocks that complete.

        ``fused`` is the caller's forward mode, not something inferred here: it is only sound when
        the launch writes at most one token per block, which is what decode guarantees and prefill
        does not.
        """
        n_tok = loc.shape[0]
        if n_tok == 0:
            return
        num_kv_head = new_k.shape[1]
        _stage_kernel[(n_tok, num_kv_head)](
            stage_k, stage_v, new_k, new_v, loc,
            cache["k"], cache["v"], cache[k_scale_name], cache[v_scale_name],
            self.slot_of, self.owner_of, self.mask_of, self.stats,
            n_tok, self.n_slots,
            NUM_KV_HEAD=num_kv_head, HEAD_DIM=self.head_dim, HALF_DIM=self.head_dim // 2,
            PAGE_SIZE=page_size, BLOCK_SIZE=self.block_size,
            BLOCK_T=triton.next_power_of_2(self.block_size),
            FULL_MASK=self.full_mask, FUSED_MIGRATE=fused,
            # num_warps=1 is a CORRECTNESS requirement in every kernel with a scalar claim loop:
            # Triton's scalar atomics are single-laned per WARP, so with more warps each warp runs
            # the loop independently and two of them win two different slots for one block. The
            # measured signature is a payload torn at warp granularity -- see host_kv.py, where the
            # same bug cost 8/60 corrupt trials at num_warps=4.
            num_warps=1,
        )

    def migrate_complete(self, stage_k, stage_v, cache,
                         *, k_scale_name: str, v_scale_name: str) -> None:
        """Quantize every complete staged block. The prefill path's second kernel."""
        _migrate_scan_kernel[(self.n_slots,)](
            stage_k, stage_v, cache["k"], cache["v"],
            cache[k_scale_name], cache[v_scale_name],
            self.slot_of, self.owner_of, self.mask_of, self.stats,
            self.n_slots,
            HEAD_DIM=self.head_dim, HALF_DIM=self.head_dim // 2,
            BLOCK_SIZE=self.block_size, BLOCK_T=triton.next_power_of_2(self.block_size),
            FULL_MASK=self.full_mask,
            num_warps=1,
        )

    def counters(self) -> dict:
        """Host-side read of the counters. Never call this on a captured path."""
        c = self.stats.tolist()
        return {"claims": c[STAT_CLAIM], "declined": c[STAT_DECLINED],
                "migrated": c[STAT_MIGRATED], "spun": c[STAT_SPUN],
                "spin_exhausted": c[STAT_SPIN_EXHAUSTED]}

    def occupancy(self) -> int:
        """Slots currently holding an in-flight block. Diagnostic only (syncs)."""
        return self.domain.occupancy()


__all__ = ["CAPTURE_CEILING", "MAX_SPIN", "N_STATS", "STAT_CLAIM", "STAT_DECLINED",
           "STAT_MIGRATED", "STAT_SPUN", "STAT_SPIN_EXHAUSTED", "arena_slots", "Int4Arena",
           "int4_request_domain"]


def int4_request_domain(num_blocks: int, num_kv_heads: int, **kwargs) -> RequestCacheDomain:
    """The domain INT4 staging wants, sized by the capture ceiling. See :func:`arena_slots`."""
    return RequestCacheDomain(num_blocks, arena_slots(num_kv_heads),
                              retention="complete", overflow="drop", **kwargs)
