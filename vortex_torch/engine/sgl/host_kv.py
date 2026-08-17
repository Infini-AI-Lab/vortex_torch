"""Host-resident KV cache with a GPU-initiated, GPU-side block cache.

The KV blocks live in **pinned host memory**; the vortex auxiliary cache
(centroids / envelopes / Save state) stays on the GPU. Sparse decoding only ever
reads the *selected* blocks, so instead of holding the whole KV on device we
fetch those blocks over PCIe on demand, from inside a Triton kernel, and hand
the attention backend a small device-resident staging pool plus a rewritten
block table.

Why in-kernel and not ``cudaMemcpyAsync``
-----------------------------------------
Measured on a B200 (PCIe Gen5 x16, ~64 GB/s theoretical), copying selected
32-token blocks:

===================  ==========  ==========  ==========
pages                       256        1024        4096
in-kernel gather      42.3 GB/s   46.2 GB/s   47.6 GB/s
host gather + DMA      8.4 GB/s    9.9 GB/s    5.2 GB/s
===================  ==========  ==========  ==========

The in-kernel path runs at 66-74% of the theoretical link and is 4.6-9x faster,
because the alternative has to *materialise a contiguous staging image on the
host first* — a CPU-side gather over scattered blocks, which is both slow and
serialised against the decode loop. Issuing one ``memcpy`` per block instead
would trade that for thousands of tiny transfers and per-copy launch overhead.
Reading straight from pinned host memory inside the kernel needs neither: the
addresses are computed on device from the block table that is already there.

Why a persistent cache and not per-step staging
-----------------------------------------------
Per-step staging is simpler but moves every selected block across PCIe on every
step. At batch 32 that is multiple GB per decode step — a few hundred
milliseconds of pure copy, which no amount of kernel tuning fixes. Sparse
selections are strongly correlated step to step (attention sinks and the local
window are re-selected essentially always, which is why the underlying
algorithms work at all), so the pool here is **persistent**: a block already
resident is not re-fetched. Only genuine misses cross the bus.

Correctness under concurrency
-----------------------------
One program per block-table entry, so the same block is usually requested by
many programs at once, and a block resident in a slot may be concurrently
considered for eviction. The protocol below is race-free rather than
merely unlikely to race:

* ``pin_gen[slot]`` is claimed with ``atomic_xchg(.., gen)``. A program may use
  or evict a slot **only if the exchange returns something other than ``gen``**,
  i.e. only if it is the first to touch that slot this step. A reader that wins
  the exchange has pinned the slot, so no evictor can subsequently take it; a
  reader that loses treats its block as a miss, which is conservative but always
  safe.
* ``claim_gen[block]`` (same ``atomic_xchg`` idiom) elects exactly **one**
  fetching program per distinct block per step. That is the dedup: 2048 requests
  over 256 distinct blocks copy 256 blocks, not 2048 (measured 8.0x less data;
  1.8-8.0x across a range of selection patterns).
* Eviction publishes ownership with ``atomic_xchg(owner_of + slot, block)`` and
  retires the previous tenant with ``atomic_cas(slot_of + old, slot, -1)`` — the
  compare-and-swap so a tenant that has *already* been re-homed elsewhere is not
  wrongly marked absent.
* The remap runs as a **separate launch**. The kernel boundary is the barrier
  that makes every ``slot_of`` publication visible, which is what lets a program
  that lost a race still end up pointing at the right slot. Spinning inside one
  kernel instead would risk deadlock, since the producer may not be resident.

A consequence worth stating: because the remap resolves every entry through
``slot_of``, a block that was both hit and (redundantly) re-fetched still
resolves consistently — duplicate work, never divergent data.

The generation counter is bumped **on device** (:func:`tick`) so the whole path
is capturable: a host-side integer would be frozen into the captured kernel
arguments and every replay would reuse one generation. The victim scan is a
bounded loop over ``capacity`` probes; it cannot fail while fewer than
``capacity`` slots are pinned, and when it does fail the entry is published as
``-1`` and counted, never left resolving to another block's slot (see
:meth:`HostKVCache.required_capacity` for why the pool is normally *not* sized to
the worst case, and what that costs).

Two mistakes worth keeping written down, because both produce silently wrong
attention rather than an error:

* **Claim with ``atomic_xchg``, not ``atomic_cas``.**
  ``atomic_cas(gen_of + p, gen - 1, gen)`` looks equivalent and is wrong: it only
  succeeds for blocks whose stored generation is exactly the previous step's, so
  any block not selected on the immediately preceding step is never fetched while
  its table entry still points at another block's data. It passes any test that
  reuses one selection set and breaks as soon as the selection changes.
* **Do not remap the caller's block table in place.** It is shared by every layer
  (the planner writes the BOS/EOS slots once per step; the topk kernel refills
  only the middle per layer), so an in-place remap makes layer 1 remap
  layer 0's output. Measured RULER 0/20 versus 100% correct. The translated ids go
  into a pool-owned table instead — allocated once at full size, because
  reallocating it per batch size frees a tensor an earlier cuda graph captured
  (``cudaErrorIllegalAddress`` mid-decode).

Validated on Qwen3-4B (MHA, 9 flows x 2 indexer backends) and GLM-4.7-Flash (MLA)
through ``examples/ruler/sweep_{mha,mla}.sh``, with cuda graphs and the radix cache
on, including the ``Save``-using ``running_avg_block_sparse`` flow. Unit tests:
``examples/misc/test_host_kv_cache.py``.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .cache_policy import (
    AGE_RESERVED,
    POLICIES,
    POLICY_BLOCK_LRU,
    POLICY_NONE,
    WAYS,
    n_sets_for,
    policy_code,
    set_of,
    touch_way,
    victim_way,
)

logger = logging.getLogger(__name__)

#: Elements per program per copy iteration. The payload of one block is
#: ``block_size * head_dim`` contiguous elements (vortex stores KV as
#: ``[num_blocks, block_size, head_dim]``), so 2048 covers a 16x128 block in one
#: pass and a 32x128 block in two.
_COPY_TILE = 2048

#: Shards for the REQUESTS counter. 32 spreads the hit path's atomic over a full warp's
#: worth of distinct addresses, which is enough to remove the serialisation without making
#: the host-side sum meaningful work (stats() already synchronises).
_REQ_SHARDS = 32


@triton.jit
def _tick_kernel(GEN):
    """Advance the step counter on device (see module docstring)."""
    if tl.program_id(0) == 0:
        tl.store(GEN, tl.load(GEN) + 1)


@triton.jit
def _tick_many_kernel(GEN_PTRS, n, BLOCK: tl.constexpr):
    """Advance EVERY layer's step counter in one launch.

    ``tick()`` is per-cache and there is one cache per layer, so a 64-layer model paid 64
    kernel launches per step to increment 64 integers. Measured on A100: 0.0117 ms per
    launch, i.e. **0.75 ms/step** of pure launch overhead — for comparison the entire
    all-hit fetch path (the one 95% of requests take) is 0.071 ms. The work is nil; the
    launches were the cost.

    ``GEN_PTRS`` is an int64 tensor of device addresses, one per layer, so a single program
    can chase them. Kept as a separate kernel rather than changing ``tick()``'s signature
    because ``tick()`` is also called on its own (single-layer tests, fetch_prefix paths).
    """
    offs = tl.arange(0, BLOCK)
    m = offs < n
    ptrs = tl.load(GEN_PTRS + offs, mask=m, other=0).to(tl.pointer_type(tl.int32))
    cur = tl.load(ptrs, mask=m, other=0)
    tl.store(ptrs, cur + 1, mask=m)


@triton.jit
def _fetch_kernel(
    HK, HV, DK, DV,
    TBL, ROWLEN, INDPTR,
    SLOT_OF, OWNER_OF, PIN_GEN, CLAIM_GEN,
    AGE, INS, INS_CTR, GEN, MISSES, FETCHES, REQUESTS,
    n_sets, row_stride, max_per_row, num_blocks,
    BLOCK_NUMEL: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    TILE: tl.constexpr,
    IS_CSR: tl.constexpr,
    POLICY: tl.constexpr,
    WAYS_C: tl.constexpr,
    N_SHARD: tl.constexpr,
):
    """Resolve one block-table entry: reuse a resident block, or fetch it.

    Grid is ``(max_per_row, num_rows)``. Entries past a row's real length are
    skipped — they hold last step's values, and treating those as block ids
    would both waste bandwidth and inflate demand past the pool's capacity
    bound.
    """
    j = tl.program_id(0)
    r = tl.program_id(1)

    if IS_CSR:
        lo = tl.load(INDPTR + r)
        n = tl.load(INDPTR + r + 1) - lo
        pos = lo + j
    else:
        # ROWLEN is in TOKENS (trtllm's ``sparse_seqlens``), so convert to a block
        # count. Passing it through as if it were blocks would over-read the row
        # by ``block_size``x, faulting or fetching junk block ids.
        n = (tl.load(ROWLEN + r) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        pos = r * row_stride + j
    if j >= n:
        return

    p = tl.load(TBL + pos)
    # Reject out-of-range ids as well as negatives. ``slot_of`` / ``claim_gen`` are
    # sized ``num_blocks``, so an id past the end indexes them out of bounds — an
    # illegal access, or a silent write into whatever allocation follows. A planner
    # can legitimately emit an id in its trailing guard page, so this is a real
    # boundary rather than a paranoid check (it was hit in practice).
    if (p < 0) | (p >= num_blocks):
        return

    gen = tl.load(GEN)

    # Elect one representative per distinct block BEFORE testing residency.
    # Order matters: the residency test pins the slot, and a pin can only be won
    # once per step, so if every requesting program tested residency then all but
    # the first would lose the pin and declare a false miss — re-fetching a block
    # that is already resident. Measured: an identical selection replayed warm
    # grew residency 69 -> 106 blocks, i.e. it re-copied over PCIe exactly what
    # the cache existed to avoid. Deduping first means one pin attempt per block;
    # everyone else is resolved by the remap through ``slot_of``.
    if tl.atomic_xchg(CLAIM_GEN + p, gen) == gen:
        # Not the representative: the block is already being handled this step, so this
        # entry must NOT fetch or pin (that is the dedup's whole purpose -- see above).
        #
        # ``block_lru`` still records the reference. That is the difference from ``lru``:
        # recency belongs to the BLOCK, and a block selected by many rows is hotter than
        # one selected by a single row, so every reference promotes it. Reserved BOS/sink
        # blocks are the motivating case -- every row selects them each step, and under
        # slot-LRU they are promoted exactly once, same as a block one row touched.
        # The promotion MUST be gated on the slot being pinned to this step. My first version
        # promoted whenever ``owner_of[s] == p`` and served 6 wrong blocks in
        # examples/misc/test_host_kv_policy.py: an unpinned slot can be concurrently claimed
        # as a victim by another program, and bumping its age mid-eviction makes the *stale*
        # tenant look most-recently-used. That both protects the wrong block and lets the
        # entry resolve through ``slot_of`` to a slot whose payload is being overwritten.
        #
        # ``pin_gen[s] == gen`` means the representative (or an earlier reference) already
        # pinned this slot for this step, so no evictor can take it and the promotion is
        # safe. Reading the pin instead of claiming it is deliberate: claiming would steal
        # the pin from the representative and turn its hit into a miss.
        if POLICY == POLICY_BLOCK_LRU:
            s_dup = tl.load(SLOT_OF + p)
            if s_dup >= 0:
                if tl.load(PIN_GEN + s_dup) == gen:
                    if tl.load(OWNER_OF + s_dup) == p:
                        touch_way(AGE, (s_dup // WAYS_C) * WAYS_C, s_dup % WAYS_C,
                                  INS_CTR, WAYS_C, POLICY)
        return

    # Count the DEMAND, once per distinct block per step -- deliberately after the dedup so
    # it is the same denominator FETCHES is a numerator of: hit_rate = 1 - fetches/requests.
    # Counting before the dedup would instead measure table entries, inflating the denominator
    # by however many rows happen to select the same block and making the hit rate look better
    # the more the batch shares blocks. `overflow` (the kernel's MISSES) is NOT this: it counts
    # staging *failures* when no way in the set is free, which is a pool-sizing pathology, not
    # a cache miss.
    # Sharded: one address per program-block rather than one global counter. This is the
    # only global atomic on the HIT path, which every program of every step reaches, so a
    # single address serialises the whole step's ~1500 programs. The shard index is derived
    # from the block id (not program_id) so it stays stable under cuda-graph replay, and the
    # host sums the shards in stats(). MISSES/FETCHES/INS_CTR are deliberately NOT sharded:
    # they only fire on a miss, which at a 95% hit rate is 1-in-20 traffic.
    tl.atomic_add(REQUESTS + (p % N_SHARD), 1)

    # --- fast path: already resident, and its slot is ours for this step ---
    #
    # ``none`` skips this deliberately: it disables *reuse* while keeping every other
    # mechanism, so it is the control for "what is the cache buying". Constexpr, so
    # the branch costs nothing in the caching policies.
    s = tl.load(SLOT_OF + p)
    if POLICY == POLICY_NONE:
        s = -1
    if s >= 0:
        if tl.atomic_xchg(PIN_GEN + s, gen) != gen:
            if tl.load(OWNER_OF + s) == p:
                # Hit. Pinned, so no evictor can take it this step. Record the use
                # so the policy's recency ordering reflects it — this is the only
                # thing that makes lru differ from fifo, and it is deliberately the
                # single extra cost on the hot (hit) path.
                touch_way(AGE, (s // WAYS_C) * WAYS_C, s % WAYS_C,
                          INS_CTR, WAYS_C, POLICY)
                return

    # --- miss: pick a victim inside this block's SET ---
    #
    # Set-associative, following OneFlow's one_embedding LRU cache: a block id maps
    # to one set of WAYS ways and only that set is examined. The previous version
    # rotated a global cursor over the whole pool, which cost O(capacity) atomics
    # per miss once every slot was pinned — measured 3.24 ms/step vs 0.10 ms at
    # twice the capacity, a 32x cliff that could only be dodged by over-provisioning
    # HBM. Bounded to one set, the miss cost no longer depends on pool size, so the
    # pool can be sized for hit rate instead.
    #
    # Which way to displace is the POLICY's decision (lru / fifo / full); see
    # cache_policy.victim_way. It is passed as a constexpr so each policy compiles
    # to its own kernel with no per-entry branching.
    set_id = set_of(p, n_sets)
    set_base = set_id * WAYS_C
    # Claim a way, retrying within the set. Two entries of the same step that map to
    # one set get the same "best" way from the policy, so without a retry the loser
    # of the pin race is dropped — measured as exactly half of a 256-block demand
    # failing on a pool 16x large enough, because ids i and i+n_sets share a set.
    #
    # OneFlow serialises this with a per-set warp mutex. A lock would not be
    # cuda-graph capturable here (and this kernel is scalar per entry, not
    # warp-cooperative), so instead: re-ask the policy after a lost race. The winner
    # now reads as pinned, so ``victim_way`` returns a different way and the set
    # fills one entry per attempt. Bounded by WAYS_C attempts, after which the set
    # genuinely has no free way for this step.
    slot = -1
    tries = 0
    while (slot < 0) & (tries < WAYS_C):
        way = victim_way(AGE, set_base, gen, PIN_GEN, INS, POLICY, WAYS_C)
        if way < 0:
            tries = WAYS_C                      # set fully pinned: stop, report below
        else:
            cand = set_base + way
            if tl.atomic_xchg(PIN_GEN + cand, gen) != gen:
                slot = cand                     # won the way
            tries += 1
    if slot < 0:
        # No way in this set is available to this step (or, under ``full``, the pool
        # is mis-sized). Publishing -1 is REQUIRED: the remap resolves every entry
        # through ``slot_of``, so a block that was never staged would otherwise
        # resolve to whatever slot it last occupied — now owned by another block —
        # and attention would read the wrong KV with no error anywhere.
        tl.atomic_add(MISSES, 1)
        tl.store(SLOT_OF + p, -1)
        return

    tl.atomic_add(FETCHES, 1)                   # one PCIe block copy
    old = tl.atomic_xchg(OWNER_OF + slot, p)
    if old >= 0:
        # Only retire the old tenant if it still believes it lives here: it may
        # already have been re-homed elsewhere by a concurrent claim.
        tl.atomic_cas(SLOT_OF + old, slot, -1)
    tl.store(SLOT_OF + p, slot)
    # Policy bookkeeping for the freshly filled way. ``INS`` is a monotonic insert
    # stamp (fifo's ordering key); ``AGE`` becomes most-recently-used so this way is
    # the LAST thing lru will evict, which is what stops a cold fill from
    # immediately displacing itself.
    # ``INS`` is fifo's key (insert order); ``AGE`` is lru's (last-use order). Both
    # come from the same monotone counter, so a freshly filled way is the most
    # recent by either measure and will not immediately evict itself.
    stamp = tl.atomic_add(INS_CTR, 1) & 0x3FFFFFFF
    tl.store(INS + slot, stamp)
    tl.store(AGE + slot, stamp)

    src = p * BLOCK_NUMEL
    dst = slot * BLOCK_NUMEL
    for off in tl.range(0, BLOCK_NUMEL, TILE, num_stages=1):
        idx = off + tl.arange(0, TILE)
        m = idx < BLOCK_NUMEL
        tl.store(DK + dst + idx, tl.load(HK + src + idx, mask=m), mask=m)
        tl.store(DV + dst + idx, tl.load(HV + src + idx, mask=m), mask=m)


@triton.jit
def _remap_kernel(
    SRC_TBL, DST_TBL, ROWLEN, INDPTR, SLOT_OF,
    row_stride, max_per_row, num_blocks, zero_slot,
    BLOCK_TOKENS: tl.constexpr,
    IS_CSR: tl.constexpr,
):
    """Write host block ids, translated to staging slots, into ``DST_TBL``.

    ``DST_TBL`` is a **separate** tensor from the one the indexer wrote, and that
    separation is load-bearing rather than tidiness. Remapping in place looks
    natural — the attention wrapper holds that tensor's address from capture time,
    so the output must land somewhere it reads — but it corrupts the table:

    * the block table is shared by every layer, and the planner fills the BOS/EOS
      slots **once per step** while the topk kernel refills only the middle slots
      **per layer**. An in-place remap replaces those per-step BOS/EOS entries with
      staging slots on layer 0, so layer 1 remaps an already-remapped value —
      indexing ``slot_of`` by a slot id. Measured: RULER 0/20 (vs 20/20 dense),
      i.e. attention silently reading unrelated blocks.

    So the pool keeps its own destination table and the attention call is pointed
    at that instead, leaving the indexer's table pristine for the next layer.
    """
    j = tl.program_id(0)
    r = tl.program_id(1)
    if IS_CSR:
        lo = tl.load(INDPTR + r)
        n = tl.load(INDPTR + r + 1) - lo
        pos = lo + j
    else:
        n = (tl.load(ROWLEN + r) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        pos = r * row_stride + j
    if j >= n:
        return
    p = tl.load(SRC_TBL + pos)
    if (p >= 0) & (p < num_blocks):
        s = tl.load(SLOT_OF + p)
        # ``s < 0`` means the cache could not stage this block for this step (every
        # way of its set is pinned by live blocks, or ``full`` ran out of room).
        # Point the entry at the RESERVED ZERO SLOT — the last way of the pool, kept
        # permanently zero-filled and never a fetch target.
        #
        # The obvious alternatives are both wrong. Leaving the host block id is an
        # out-of-bounds index into the staging pool. Substituting slot 0 (what this
        # did originally) hands the attention kernel a real but *unrelated* block, so
        # a refusal silently corrupts the score instead of dropping a contribution —
        # measured as wrong data served on an over-subscribed pool. A zero K/V block
        # contributes a constant logit and a zero value, i.e. the entry degrades to
        # "no information" rather than "wrong information", which is the honest
        # behaviour for a cache that cannot place a block. Every occurrence is
        # counted in ``overflow`` so the pool can be resized.
        tl.store(DST_TBL + pos, tl.where(s >= 0, s, zero_slot))
    else:
        tl.store(DST_TBL + pos, p)


@triton.jit
def _invalidate_locs_kernel(
    LOC, SLOT_OF, OWNER_OF, AGE, n,
    NUM_KV_HEAD: tl.constexpr, PAGE_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Evict the blocks covering the token positions in ``LOC``.

    Mirrors the address arithmetic in
    ``cache/triton_kernels/set_kv.py::set_kv_buffer_kernel`` exactly — vortex
    stores KV block-interleaved by head, so one *token* touches one block per KV
    head, at scattered block ids. Recomputing the mapping here (rather than
    passing block ids in) keeps the two in step: if the layout changes, both must
    change together and the duplication is visible.
    """
    i = tl.program_id(0)
    h = tl.program_id(1)
    if i >= n:
        return
    pos = tl.load(LOC + i)
    trans = (pos // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + h * PAGE_SIZE + pos % PAGE_SIZE
    blk = (trans // BLOCK_SIZE).to(tl.int32)
    s = tl.load(SLOT_OF + blk)
    if s >= 0:
        tl.store(SLOT_OF + blk, -1)
        # Only disown the slot if this block still holds it: the block may have
        # been evicted and the slot re-let to someone else between the two loads.
        if tl.atomic_cas(OWNER_OF + s, blk, -1) == blk:
            # RELEASE the way, not just the ownership. ``full`` treats ``age != 0``
            # as "permanently taken" (it never evicts), so a way whose tenant was
            # invalidated but whose age was left set is dead: it owns nothing and can
            # never be filled again. Decode invalidates the newest block every step
            # and layer, so those dead ways accumulate until sets fill and ``full``
            # starts refusing blocks — measured as RULER drifting to 14-80% while
            # ``lru``/``fifo``, which pick victims by key order rather than by
            # emptiness, were unaffected and stayed at 100%.
            tl.store(AGE + s, 0)


@triton.jit
def _stage_dense_kernel(HK, HV, DK, DV, SRC_IDX, DST_IDX, n,
                        BLOCK_NUMEL: tl.constexpr, TILE: tl.constexpr):
    """Stage the block named by ``SRC_IDX[i]`` into slot ``i``; set ``DST_IDX[i]=i``.

    Used by the dense prefix path. The identity mapping means no cache lookup, no
    atomics and no dedup: a radix prefix names each block once, so there is
    nothing to dedup, and it is read once, so nothing to cache.

    Source and destination are **separate tensors on purpose**. The obvious
    version rewrites the index array in place, which is wrong here: the prefill
    index array is shared by every layer, and once layer 0 has replaced the block
    ids with ``0..n-1`` layer 1 would stage host block ``i`` for entry ``i``
    instead of the block actually wanted. Reading from an immutable snapshot makes
    each layer's staging identical and the destination write idempotent.
    """
    i = tl.program_id(0)
    if i >= n:
        return
    p = tl.load(SRC_IDX + i)
    tl.store(DST_IDX + i, i)
    if p < 0:
        return
    src = p * BLOCK_NUMEL
    dst = i * BLOCK_NUMEL
    for off in tl.range(0, BLOCK_NUMEL, TILE, num_stages=1):
        idx = off + tl.arange(0, TILE)
        m = idx < BLOCK_NUMEL
        tl.store(DK + dst + idx, tl.load(HK + src + idx, mask=m), mask=m)
        tl.store(DV + dst + idx, tl.load(HV + src + idx, mask=m), mask=m)


@triton.jit
def _set_latent_kernel(BUF, LOC, NOPE, ROPE, n,
                       LATENT: tl.constexpr, RANK: tl.constexpr,
                       ROPE_DIM: tl.constexpr):
    """Write ``[kv_c | k_pe]`` into a host-resident MLA latent buffer.

    Exists because sglang's own MLA writer is a JIT'd CUDA kernel that asserts its
    destination is on ``cuda:0``
    (``jit_kernel/csrc/elementwise/set_mla_kv_buffer.cuh``), so it rejects the
    pinned host buffer outright — unlike vortex's Triton ``set_kv`` launcher, which
    only dereferences the pointer it is handed. Semantics mirror upstream's:
    ``buf[loc, 0, :rank] = kv_c`` and ``buf[loc, 0, rank:] = k_pe``.
    """
    i = tl.program_id(0)
    if i >= n:
        return
    pos = tl.load(LOC + i)
    dst = BUF + pos * LATENT
    r = tl.arange(0, RANK)
    tl.store(dst + r, tl.load(NOPE + i * RANK + r))
    q = tl.arange(0, ROPE_DIM)
    tl.store(dst + RANK + q, tl.load(ROPE + i * ROPE_DIM + q))


@triton.jit
def _gather_tokens_kernel(SRC, IDX, DST, n,
                          DIM: tl.constexpr, REAL_DIM: tl.constexpr):
    """``DST[i, :] = SRC[IDX[i], :]`` with SRC in pinned host memory."""
    i = tl.program_id(0)
    if i >= n:
        return
    p = tl.load(IDX + i)
    d = tl.arange(0, DIM)
    m = d < REAL_DIM
    tl.store(DST + i * REAL_DIM + d,
             tl.load(SRC + p * REAL_DIM + d, mask=m), mask=m)


class HostKVCache:
    """A GPU staging pool caching blocks of a host-resident KV buffer.

    One instance per layer. ``host_k`` / ``host_v`` are pinned host tensors
    shaped ``[num_blocks, block_size, head_dim]``; the pool holds ``capacity``
    of those blocks on device.

    All mutable state is device-resident and every buffer is allocated once, so
    :meth:`fetch` is safe to capture in a cuda graph and to replay with a
    different selection each time.
    """

    def __init__(
        self,
        host_k: torch.Tensor,
        host_v: torch.Tensor,
        capacity: int,
        device: str | torch.device,
        *,
        fused: bool = False,
        policy: str = "lru",
    ):
        # ``fused``: K and V are the SAME tensor (MLA's single latent field). The
        # V staging buffer is then a duplicate of K, so it is aliased rather than
        # allocated — otherwise every MLA layer would waste a full staging pool of
        # HBM holding a second copy of the same blocks.
        if fused:
            assert host_k.data_ptr() == host_v.data_ptr(), (
                "fused=True means K and V are one tensor (MLA latent)"
            )
        assert host_k.shape == host_v.shape, "host K and V must have equal shape"
        assert host_k.is_pinned() and host_v.is_pinned(), (
            "host KV must be pinned; a pageable host tensor cannot be read from a "
            "kernel and would fault"
        )
        self.host_k = host_k
        self.host_v = host_v
        self.num_blocks = host_k.shape[0]
        self.block_numel = host_k.shape[1] * host_k.shape[2]
        #: Tokens per block — needed to convert trtllm's token-valued
        #: ``sparse_seqlens`` into a per-row block count.
        self.block_tokens = host_k.shape[1]
        # Capacity is rounded UP to whole sets, mirroring OneFlow's
        # ``n_set * kWarpSize``: a partial set would leave ways that no block id
        # maps to, i.e. HBM that can never be used.
        self.policy = policy
        self.policy_code = policy_code(policy)
        self.n_sets = n_sets_for(capacity)
        self.capacity = self.n_sets * WAYS
        #: Last slot of the pool, held permanently zero and never used as a fetch
        #: target, so an entry the cache cannot place resolves to a zero K/V block
        #: instead of to another block's data (see ``_remap_kernel``). Costs one
        #: block of HBM; ``usable_slots`` is what eviction may touch.
        self.zero_slot = self.capacity - 1
        self.usable_slots = self.capacity - 1
        self.device = device

        # ``full`` promises never to evict, so every host block must have a way it can
        # occupy permanently. Total capacity is NOT the binding constraint: blocks map
        # to sets by ``id % n_sets``, so a set must accommodate every block congruent
        # to it — ``ceil(num_blocks / n_sets)`` of them — within its WAYS ways. A pool
        # that merely satisfies ``capacity >= num_blocks`` can still have an
        # over-subscribed set, and ``full`` then refuses those blocks and serves the
        # reserved zero block instead: measured as RULER 14% where ``lru``/``fifo``
        # held 100%. Silent degradation is the wrong answer for a mis-set knob, so
        # this is rejected at construction.
        if policy == "full":
            per_set = -(-self.num_blocks // self.n_sets)      # ceil
            # One way of one set is the reserved zero slot, so that set has WAYS-1.
            if per_set > WAYS - 1:
                need_sets = -(-self.num_blocks // (WAYS - 1))
                raise ValueError(
                    f"vortex_host_kv_policy='full' never evicts, so every one of the "
                    f"{self.num_blocks} host blocks needs a permanent way. Blocks map "
                    f"to sets by id % n_sets, so with {self.n_sets} sets a set would "
                    f"hold up to {per_set} blocks but only has {WAYS - 1} usable ways "
                    f"(one is the reserved zero slot). Raise "
                    f"vortex_host_kv_pool_blocks to >= {need_sets * WAYS}, lower "
                    f"vortex_host_kv_gb, or use 'lru' / 'fifo', which evict instead "
                    f"of refusing blocks."
                )

        self.fused = fused
        shape = (self.capacity,) + tuple(host_k.shape[1:])
        self.dev_k = torch.zeros(shape, dtype=host_k.dtype, device=device)
        self.dev_v = self.dev_k if fused else torch.zeros(
            shape, dtype=host_v.dtype, device=device)

        i32 = torch.int32
        # -1 = "not resident anywhere" / "owns nothing".
        self.slot_of = torch.full((self.num_blocks,), -1, dtype=i32, device=device)
        self.owner_of = torch.full((self.capacity,), -1, dtype=i32, device=device)
        # Generations start below the first step's value so nothing looks claimed.
        self.pin_gen = torch.full((self.capacity,), -1, dtype=i32, device=device)
        self.claim_gen = torch.full((self.num_blocks,), -1, dtype=i32, device=device)
        # Per-way policy state, laid out set-major so one set is contiguous.
        #: LRU rank: 0 = empty, 1 = least-recently-used, WAYS = most-recent. Ages
        #: are a permutation over the occupied ways (OneFlow's scheme), so eviction
        #: is "the way with rank 1" rather than a search for a minimum timestamp.
        self.age = torch.zeros((self.capacity,), dtype=i32, device=device)
        #: Monotonic insert stamp per way — fifo's ordering key. Separate from
        #: ``age`` because fifo must NOT be perturbed by reads.
        self.ins = torch.zeros((self.capacity,), dtype=i32, device=device)
        self.ins_ctr = torch.zeros((1,), dtype=i32, device=device)
        self.gen = torch.zeros((1,), dtype=i32, device=device)
        # Mark the reserved zero slot so no policy can select it. This is an ``age``
        # sentinel, not a pin: pinning is tested as ``pin == gen``, so a far-future
        # pin is never "currently pinned" and left the slot looking like the LRU
        # rank-1 way — the first victim. Real blocks were staged into it, and
        # refusals then read that block instead of zeros.
        self.age[self.zero_slot] = AGE_RESERVED
        self.dev_k[self.zero_slot].zero_()
        if not fused:
            self.dev_v[self.zero_slot].zero_()
        #: Entries the pool could not serve. Non-zero means it is under-sized;
        #: read it off the host between steps, never inside one (a sync would
        #: break capture).
        self.overflow = torch.zeros((1,), dtype=i32, device=device)
        #: Cumulative count of blocks actually copied over PCIe. Compare against
        #: the number of table entries to see the hit rate the cache is buying.
        self.fetches = torch.zeros((1,), dtype=i32, device=device)
        #: Distinct blocks the indexer asked for, counted once per block per step (after the
        #: dedup in ``_fetch_kernel``). Pairs with ``fetches`` -- which counts the PCIe copies
        #: an actual miss triggers -- to give the cache hit rate. Without it there is no
        #: denominator: ``fetches`` alone cannot distinguish "few misses" from "few requests".
        #: Sharded across ``_REQ_SHARDS`` addresses so the hit path's atomic_add does not
        #: serialise a whole step on one location; ``stats()`` sums them.
        self.requests = torch.zeros((_REQ_SHARDS,), dtype=i32, device=device)
        #: Dense prefix staging (prefill-with-prefix only), grown on demand. See
        #: :meth:`fetch_prefix` for why allocating here is safe.
        self._prefix_k: Optional[torch.Tensor] = None
        self._prefix_v: Optional[torch.Tensor] = None
        #: Destination for the remapped block table (see _remap_kernel).
        self._remap_out: Optional[torch.Tensor] = None

    # -- sizing ---------------------------------------------------------------
    @staticmethod
    def required_capacity(num_rows: int, blocks_per_row: int) -> int:
        """Capacity at which a step's whole demand can be resident at once.

        ``num_rows * blocks_per_row`` is the worst-case distinct demand of one step;
        each such block pins a way, so this is the point beyond which a step cannot
        overflow. It is a *hit-rate* target now, not a correctness requirement: with
        set-associative eviction a smaller pool simply misses more, and per-miss cost
        no longer grows with capacity.

        The old contract was different and worth recording, because it drove a
        sizing rule that is now obsolete: victims used to be found by rotating a
        global cursor, so a pool sized exactly to demand had every slot pinned and
        each miss probed O(capacity) slots — measured 3.24 ms/step against 0.10 ms
        at 2x capacity. That forced a 2x margin purely to dodge the scan. Set
        associativity bounds the probe to one set (:data:`cache_policy.WAYS` ways),
        so the margin is no longer bought with HBM and this returns the honest
        figure.

        Note the residual sensitivity: demand is spread over sets by
        ``block_id % n_sets``, so a *set* can fill even when the pool as a whole has
        room. That is the standard associativity trade-off, and it is why overflow
        is counted rather than assumed impossible.
        """
        return max(1, num_rows * blocks_per_row)

    @staticmethod
    def affordable_capacity(
        block_numel: int, elt_size: int, layers: int, budget_bytes: int, *, fused: bool = False
    ) -> int:
        """Blocks per layer that fit ``budget_bytes`` of HBM across all layers.

        ``fused`` (MLA) stages one buffer instead of two, so it fits twice as many.
        """
        per_block = block_numel * elt_size * (1 if fused else 2)
        return max(1, int(budget_bytes) // max(1, per_block * max(1, layers)))

    def nbytes_device(self) -> int:
        bufs = [self.dev_k, self.slot_of, self.owner_of, self.pin_gen,
                self.claim_gen, self.age, self.ins]
        if not self.fused:                       # fused aliases dev_k; don't double-count
            bufs.append(self.dev_v)
        return sum(t.element_size() * t.numel() for t in bufs)

    # -- runtime --------------------------------------------------------------
    def tick(self) -> None:
        """Start a new step. Call **once per step**, before the first fetch.

        Bumped on device so a captured graph advances it on every replay.

        For a multi-layer pool prefer :func:`tick_all`, which does every layer in one
        launch — see :func:`_tick_many_kernel` for why that matters.
        """
        _tick_kernel[(1,)](self.gen)

    def fetch(
        self,
        table: torch.Tensor,
        *,
        row_lens: Optional[torch.Tensor] = None,
        indptr: Optional[torch.Tensor] = None,
        num_rows: int,
        max_per_row: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make every block named by ``table`` resident, and rewrite it in place.

        Pass ``row_lens`` for a 2D block table (trtllm) or ``indptr`` for CSR
        indices (flashinfer) — the two backends name their selections
        differently but the work is identical. ``row_lens`` is in **tokens**
        (trtllm's ``sparse_seqlens``) and converted to blocks on device;
        ``indptr`` is already a block-count prefix sum.

        Returns ``(dev_k, dev_v, remapped_table)``. ``table`` itself is left
        UNCHANGED — the translated ids go into a pool-owned table, because the
        caller's table is shared across layers and re-remapping it would corrupt
        the per-step BOS/EOS slots (see :func:`_remap_kernel`). Pass the returned
        table to the attention call.
        """
        is_csr = indptr is not None
        if is_csr == (row_lens is not None):
            raise ValueError("pass exactly one of row_lens (trtllm) / indptr (flashinfer)")

        row_stride = 0 if is_csr else table.stride(0)
        grid = (max_per_row, num_rows)
        out = self._remap_table(table)
        _fetch_kernel[grid](
            self.host_k, self.host_v, self.dev_k, self.dev_v,
            table, row_lens, indptr,
            self.slot_of, self.owner_of, self.pin_gen, self.claim_gen,
            self.age, self.ins, self.ins_ctr,
            self.gen, self.overflow, self.fetches, self.requests,
            self.n_sets, row_stride, max_per_row, self.num_blocks,
            BLOCK_NUMEL=self.block_numel, BLOCK_TOKENS=self.block_tokens,
            TILE=_COPY_TILE, IS_CSR=is_csr,
            POLICY=self.policy_code, WAYS_C=WAYS, N_SHARD=_REQ_SHARDS,
        )
        _remap_kernel[grid](
            table, out, row_lens, indptr, self.slot_of,
            row_stride, max_per_row, self.num_blocks, self.zero_slot,
            BLOCK_TOKENS=self.block_tokens, IS_CSR=is_csr,
        )
        return self.dev_k, self.dev_v, out

    def reserve_remap(self, table: torch.Tensor) -> None:
        """Pre-allocate the remap buffer for ``table``'s full shape.

        Call from ``init_forward_metadata`` / the planner — i.e. OUTSIDE any
        captured region — so :meth:`fetch` never has to allocate. A lazy first-use
        allocation inside a cuda graph is captured as part of the graph, which is
        both illegal and (worse) silent: the pool's later replays then read a buffer
        the graph itself created.
        """
        self._remap_table(table)

    def _remap_table(self, table: torch.Tensor) -> torch.Tensor:
        """A pool-owned tensor shaped like ``table`` to receive the translated ids.

        **The allocation must happen exactly once, and the address must never
        change.** sglang captures one cuda graph per batch size, so this is called
        with a differently-shaped table for each. Reallocating on a shape change
        frees the tensor an *earlier* graph already captured, leaving that graph
        replaying against freed memory — observed as
        ``cudaErrorIllegalAddress`` during decode, long after capture.

        So callers pass the **full** fixed-size block table and a row count; the
        buffer is allocated to that full shape on first use and every batch size
        gets a view of it. Growth is still handled (a larger table would need a
        bigger buffer) but cannot happen after capture, because the full table's
        shape does not depend on the batch.
        """
        cached = self._remap_out
        if (cached is None or cached.dtype != table.dtype
                or cached.numel() < table.numel()
                or cached.shape[1:] != table.shape[1:]):
            if cached is not None:
                # Reallocation, not growth per se: a different row width or dtype
                # also lands here. Either way the old tensor is freed, so any cuda
                # graph captured against it now holds a dangling pointer. In
                # serving this must not happen after capture — callers pass the
                # full, batch-independent table for exactly that reason.
                logger.warning(
                    "host KV remap table reallocated %s -> %s; if this happens "
                    "after cuda-graph capture, previously captured graphs now hold "
                    "a freed pointer",
                    tuple(cached.shape), tuple(table.shape),
                )
            cached = torch.empty_like(table)
            self._remap_out = cached
        # The remap kernel addresses both buffers with the CALLER's row stride, so a
        # returned view must share it. A row-count slice does (same width), but
        # assert rather than rely on that holding if the shapes ever diverge.
        assert cached.stride(0) == table.stride(0) or cached.shape[0] == table.shape[0], (
            f"remap buffer row stride {cached.stride(0)} != table {table.stride(0)}"
        )
        return cached[: table.shape[0]] if cached.shape[0] != table.shape[0] else cached

    def fetch_prefix(
        self,
        indices: torch.Tensor,
        src_indices: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Promote a **dense** prefix to device and rewrite ``indices`` in place.

        This serves the prefill-with-prefix path, which runs on a radix-cache hit
        and reads *every* block of the cached prefix rather than a sparse
        selection. That makes it fundamentally different from :meth:`fetch`:

        * demand is the whole prefix, so it is not bounded by the sparse budget
          and cannot use the eviction pool without thrashing (each block would be
          evicted before the kernel reads it);
        * it is a **prefill** path, and cuda-graph capture is decode-only
          (``init_forward_metadata_capture_cuda_graph`` asserts
          ``is_decode_or_idle``), so allocating here is allowed — the constraint
          that forces preallocation elsewhere does not apply.

        So this uses its own contiguous buffer, grown geometrically and reused
        across calls, and stages the prefix compactly: destination slot ``i`` is
        entry ``i``, no cache lookup and no atomics. Prefix blocks are also not
        worth caching — a radix hit reads each one once, then decode proceeds
        through the sparse path.

        ``src_indices`` is the per-step snapshot of the original block ids taken
        by :meth:`snapshot_prefix`; ``indices`` is the live buffer the attention
        wrapper reads, rewritten here to ``0..n-1``. Staging from an immutable
        snapshot is what makes this safe to call once per layer: without it, layer
        0's rewrite would become layer 1's input and every later layer would stage
        the wrong blocks.
        """
        if n == 0:
            return self.dev_k, self.dev_v
        if self._prefix_k is None or self._prefix_k.shape[0] < n:
            cap = max(n, 2 * (0 if self._prefix_k is None else self._prefix_k.shape[0]))
            shape = (cap,) + tuple(self.host_k.shape[1:])
            self._prefix_k = torch.empty(shape, dtype=self.host_k.dtype, device=self.device)
            self._prefix_v = torch.empty(shape, dtype=self.host_v.dtype, device=self.device)
            logger.debug("host KV prefix staging grown to %d blocks", cap)
        _stage_dense_kernel[(n,)](
            self.host_k, self.host_v, self._prefix_k, self._prefix_v,
            src_indices, indices, n,
            BLOCK_NUMEL=self.block_numel, TILE=_COPY_TILE,
        )
        return self._prefix_k, self._prefix_v

    def invalidate_locs(self, loc: torch.Tensor, page_size: int, num_kv_heads: int) -> None:
        """Evict the blocks covering token positions ``loc``.

        Called after new K/V is written for those tokens: a resident copy in the
        staging pool predates the write, so serving it would attend over stale
        (usually zero) K/V for the newest tokens — precisely the ones the local
        window always selects. Targeted rather than a full
        :meth:`invalidate` because this runs on every decode step, and dropping
        the whole pool each step would make the cache useless.

        Cudagraph-safe: fixed grid from ``loc``'s static shape, no host sync.
        """
        n = loc.numel()
        if n == 0:
            return
        block_size = self.block_tokens
        _invalidate_locs_kernel[(n, num_kv_heads)](
            loc, self.slot_of, self.owner_of, self.age, n,
            NUM_KV_HEAD=num_kv_heads, PAGE_SIZE=page_size, BLOCK_SIZE=block_size,
        )

    def invalidate(self) -> None:
        """Drop every cached block.

        Needed whenever host blocks are rewritten behind the pool's back — a
        radix-cache page move relocates content between block ids, so a slot's
        recorded owner no longer describes its contents.
        """
        self.slot_of.fill_(-1)
        self.owner_of.fill_(-1)
        self.pin_gen.fill_(-1)
        # Ages back to 0 (= empty) so the next fill is a cold one rather than
        # inheriting a recency order for blocks that are no longer resident.
        self.age.zero_()
        self.ins.zero_()
        # Re-establish the reserved zero slot: the wipes above would otherwise make
        # it look empty, i.e. a legal fill target.
        self.age[self.zero_slot] = AGE_RESERVED
        self.owner_of[self.zero_slot] = -1

    def stats(self) -> dict:
        """Host-visible counters. Synchronises — debug/reporting only."""
        return {
            "policy": self.policy,
            "n_sets": self.n_sets,
            "ways": WAYS,
            "capacity": self.capacity,
            "usable_slots": self.usable_slots,
            "num_host_blocks": self.num_blocks,
            "generation": int(self.gen.item()),
            "overflow": int(self.overflow.item()),
            "fetches": int(self.fetches.item()),
            "requests": int(self.requests.sum().item()),
            "hit_rate": self.hit_rate(),
            "resident": int((self.owner_of >= 0).sum().item()),
        }

    def hit_rate(self) -> float:
        """Fraction of requested blocks genuinely served from the GPU pool.

        ``(requests - fetches - overflow) / requests``.

        The ``overflow`` term is essential and was missing in the first version, which
        computed ``1 - fetches/requests``. An overflow is a **refusal**: no way in the
        block's set was free, so ``_fetch_kernel`` publishes slot ``-1`` and the entry reads
        the reserved zero block. A refusal copies nothing, so it is not a fetch -- and the
        naive formula therefore scored it as a *hit*. That inverted the metric: measured on a
        48x31 decode trace, an undersized pool (256 blocks, 0.17x demand) reported a "hit
        rate" of 0.95 while 81% of its requests were refusals; the corrected rate is 0.14, and
        it now rises with capacity (0.14 -> 0.60 from 256 -> 1024 blocks) as it must.

        A pool that overflows is also serving WRONG data to those entries -- zeros instead of
        KV -- so a non-trivial ``overflow`` is a correctness alarm, not just a perf note.
        Report it alongside the rate.

        Returns 0.0 before any request, so a cold cache needs no special-casing at call sites.
        """
        req = int(self.requests.sum().item())
        if req <= 0:
            return 0.0
        served = req - int(self.fetches.item()) - int(self.overflow.item())
        return max(0.0, served / req)



def tick_all(caches) -> None:
    """Advance the step counter of every cache in ``caches`` with ONE kernel launch.

    Replaces a per-layer ``for c in caches: c.tick()`` loop. On a 64-layer model that loop
    cost 64 launches x 0.0117 ms = ~0.75 ms/step to increment 64 int32s, which is >10x the
    entire all-hit fetch path (0.071 ms). The pointer array is cached on first use and
    reused, so the steady-state cost is one launch.

    Safe under cuda-graph capture for the same reason ``tick`` is: the increment happens on
    device, so a replay advances the generation exactly as a fresh launch would. Allocating
    the pointer array lazily inside a captured region would be a bug (the allocation becomes
    part of the graph), which is why it is built on the FIRST call -- callers already invoke
    tick outside capture, from the per-step planner.
    """
    if not caches:
        return
    n = len(caches)
    if n == 1:
        caches[0].tick()
        return
    holder = caches[0]
    ptrs = getattr(holder, "_tick_ptrs", None)
    if ptrs is None or ptrs.numel() != n:
        ptrs = torch.tensor([c.gen.data_ptr() for c in caches],
                            dtype=torch.int64, device=caches[0].gen.device)
        holder._tick_ptrs = ptrs
    block = triton.next_power_of_2(n)
    _tick_many_kernel[(1,)](ptrs, n, BLOCK=block)


def set_host_latent(
    buf: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    """Write the fused MLA latent into a **host-resident** ``kv_buffer``.

    Drop-in for sglang's ``set_mla_kv_buffer_triton`` for the host-KV case; see
    :func:`_set_latent_kernel` for why upstream's cannot be used.
    """
    n = loc.numel()
    if n == 0:
        return
    rank = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    _set_latent_kernel[(n,)](
        buf, loc, cache_k_nope.contiguous(), cache_k_rope.contiguous(), n,
        LATENT=buf.shape[-1], RANK=rank, ROPE_DIM=rope_dim,
    )



def gather_host_tokens(
    host_buf: torch.Tensor,
    indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather ``indices`` *tokens* (not blocks) from a host buffer onto the device.

    For the MLA prefix reconstruction, which rebuilds per-head K/V from the latent
    with a token-granular ``index_select`` over the whole cached prefix. That is a
    different shape of request from :meth:`HostKVCache.fetch_prefix` (blocks, and
    routed through the staging pool), and torch's ``index_select`` cannot serve it
    directly: the index lives on the device and the source on the host, which is
    exactly the ``wrapper_CUDA__index_select`` device-mismatch error.

    Gathering on the host instead would be correct but slow and synchronous, so
    this does the indexed copy in a kernel reading pinned memory, like every other
    path here. Prefill only, and prefill is not cuda-graph captured, so allocating
    the destination per call is acceptable; pass ``out`` to reuse a buffer.
    """
    n = indices.numel()
    dim = host_buf.shape[-1]
    if out is None:
        out = torch.empty((n, dim), dtype=host_buf.dtype, device=indices.device)
    if n == 0:
        return out
    _gather_tokens_kernel[(n,)](
        host_buf, indices, out, n, DIM=triton.next_power_of_2(dim), REAL_DIM=dim,
    )
    return out


__all__ = ["HostKVCache", "set_host_latent", "gather_host_tokens"]
