"""Cache policies for the GPU-as-a-cache-over-host-KV tier.

The GPU staging pool in :mod:`vortex_torch.engine.sgl.host_kv` is a **cache** over a
larger host-resident KV store: the host buffer holds the whole context, the device
holds whichever blocks recent steps touched. Which block gets displaced when the
device is full is a policy choice, and different serving shapes want different
answers — so the policy lives here, behind one small device-side contract, rather
than being welded into the fetch kernel.

Design follows OneFlow's ``one_embedding`` cache (Apache-2.0;
``oneflow/core/embedding/{cache.h,lru_cache.cu,full_cache.cu}``), which solves the
same two-tier problem for embedding tables. Two ideas are taken from it directly:

**1. Set-associative, not global.** OneFlow's LRU cache hashes a key to a *set* of
32 ways (one per lane of a warp) and only ever searches or evicts inside that set.
That bounds the work per miss to one set, instead of a scan over the whole pool.
This matters here for a measured reason: the first version of this code picked
victims with a global rotating cursor, which cost **O(capacity)** atomics per miss
once the pool was full — 3.24 ms/step against 0.10 ms when capacity was doubled,
a 32x cliff whose only fix was to over-provision HBM. A set-associative pool is
flat in capacity instead, so the pool can be sized for hit rate rather than to
dodge a scan.

**2. Recency in a small per-way field, compared only within a set.** OneFlow keeps
``age`` as a rank permutation (``0`` empty, ``1`` LRU, ``ways`` MRU) and renumbers the
set on every hit. That renumbering needs the set's mutex, which a warp-cooperative
kernel has and this scalar, lock-free kernel does not — done unsynchronised it races
and stops being a permutation, which measurably *inverted* the policy. So vortex keeps
the same ordering semantics (smaller = evict first) but stores a monotone stamp
written with one ``atomic_max``, so a hit needs no coordination with the other ways.

Vortex differences from OneFlow, all forced by this being a *serving attention*
cache rather than a training embedding cache:

* **Identity hash.** OneFlow hashes arbitrary int64 embedding keys. Our keys are
  dense block ids in ``[0, num_blocks)``, so ``block_id % n_sets`` is already
  uniform, and using it means neighbouring blocks of one request spread across
  sets instead of colliding. No hash function, no load factor.
* **No per-set mutex.** OneFlow takes a warp mutex per set because a training step
  writes values into the cache. Here the device copy is read-only between steps
  (the host tier is authoritative; writes go to the host buffer and invalidate),
  so a claim is one ``atomic_cas`` per way and a losing claimant just treats its
  block as a miss — conservative, never wrong. That also keeps the whole path
  cuda-graph capturable, which a spin lock would not be.
* **Pinning instead of eviction lists.** OneFlow returns evicted keys so the caller
  can write them back. Nothing here is dirty, so eviction needs no write-back; what
  it *does* need is a guarantee that a block in use by the current step is not
  displaced by a later miss in the same step. That is the ``gen`` pin, and it is why
  a policy must consult it before choosing a victim.

Policies
--------
``lru``
    Set-associative LRU, the OneFlow design above. Best when the working set is
    larger than the pool but has temporal locality — the normal case, since sparse
    attention re-selects sinks and the local window nearly every step.
``block_lru``
    Same sets and the same ordering key as ``lru``, but recency is keyed on the
    **block** and promoted on *every* reference rather than once per step. This is
    OneFlow's granularity: its cache promotes a key each time it is looked up, whereas
    our ``lru`` sees each block at most once per step because ``_fetch_kernel``'s
    CLAIM_GEN dedup returns early for the non-representative entries. The difference is
    the batch-sharing signal — a block selected by 40 rows and a block selected by 1 get
    the same stamp under ``lru``, while ``block_lru`` ranks the shared one hotter.
    Reserved BOS/sink blocks are the motivating case, since every row selects them.
    Costs one extra ``atomic_max`` per duplicate reference and nothing on the miss path.
``fifo``
    Same sets, but a per-set insert counter picks the victim, so a way's residency
    does not depend on how often it is read. Cheaper than LRU (no rank shuffle on
    hit) and immune to a scan-heavy step evicting everything useful. Useful when
    selections churn.
``full``
    No eviction: the pool is asserted large enough for every block, so a miss is a
    first-touch fill. This is OneFlow's ``kFull`` policy. It turns the cache into a
    pure prefetch buffer and is the right choice when ``host_kv_gb`` is only
    modestly larger than what fits in HBM.
``none``
    **Caching disabled.** The device pool is used purely as per-step staging: the
    residency fast path is skipped, so every selected block is re-copied from the
    host every step even if it is already resident. Slower by construction — it is
    the control that shows what the cache is worth, and the fallback if a caching
    bug is ever suspected in production. Everything else (set-associative slot
    allocation, pinning, the reserved zero block) is unchanged, so it isolates
    *reuse* rather than switching to a different code path.

Adding one means adding a branch in :func:`victim_way` and a name here; the fetch
kernel does not change.
"""
from __future__ import annotations

import triton
import triton.language as tl

#: Ways per set. 32 = one per lane of a warp, matching OneFlow's ``kWarpSize``
#: choice: a warp can then examine an entire set with one ``ballot``-style pass and
#: no cross-lane loop. Our kernel is scalar per block-table entry rather than
#: warp-cooperative, so the value is not load-bearing for us in the same way —
#: but it keeps the associativity high enough that a set rarely fills while the
#: per-miss probe stays bounded at 32.
WAYS: int = 32

#: Policy names accepted by ``vortex_host_kv_policy``.
POLICIES = ("lru", "block_lru", "fifo", "full", "none")

#: ``age`` value marking a way as RESERVED — never a fetch target, never evicted.
#: One way of the pool carries it: the zero-filled slot that entries the cache
#: cannot place resolve to (see ``host_kv._remap_kernel``).
#:
#: A distinct sentinel rather than a large ``pin_gen``: pinning is tested as
#: ``pin == gen_now``, so a far-future pin value is never equal to the current
#: generation and therefore does NOT make a way untouchable. Overloading the pin
#: that way left the reserved slot looking like an ordinary rank-1 LRU way — i.e.
#: the first victim — and real blocks were staged into it, so refusals then read
#: that block's data instead of zeros.
AGE_RESERVED = 0x7FFFFFFF
_AGE_RESERVED_C = tl.constexpr(0x7FFFFFFF)

#: Integer codes, so the policy can be a ``tl.constexpr`` and specialise the
#: kernel at compile time rather than branching per entry at runtime.
#:
#: These must be ``tl.constexpr(...)`` in **call** form, not plain ints and not
#: annotated (``x: tl.constexpr = 0``): a @jit'ed function cannot read a module
#: global unless it was instantiated that way, and the annotation form is
#: explicitly unsupported. Triton says so, but only once the kernel that reads
#: them is compiled, so the failure surfaces far from the definition.
POLICY_LRU = tl.constexpr(0)
POLICY_FIFO = tl.constexpr(1)
POLICY_FULL = tl.constexpr(2)
POLICY_NONE = tl.constexpr(3)
#: ``block_lru`` — OneFlow's granularity. ``lru`` above keys recency on the SLOT and is
#: promoted at most once per step, because ``_fetch_kernel``'s CLAIM_GEN dedup returns
#: early for every non-representative entry. ``block_lru`` instead promotes on **every
#: reference**, so a block selected by many rows in one step outranks a block selected by
#: one — the sharing signal the dedup otherwise discards. Reserved BOS/sink blocks are the
#: clearest case: every row selects them, so they should be the last thing evicted.
POLICY_BLOCK_LRU = tl.constexpr(4)

#: Plain ints for the launch site (the constexprs above are for kernel code).
_CODES = {"lru": 0, "fifo": 1, "full": 2, "none": 3, "block_lru": 4}


def policy_code(name: str) -> int:
    """Map a policy name to its kernel constant, rejecting typos loudly."""
    try:
        return _CODES[name]
    except KeyError:
        raise ValueError(
            f"unknown host-KV cache policy {name!r}; expected one of {POLICIES}"
        ) from None


def n_sets_for(capacity: int) -> int:
    """Sets needed for ``capacity`` blocks, at :data:`WAYS` ways each.

    Rounded up, so the realised capacity is ``n_sets * WAYS >= capacity`` — the
    same convention as OneFlow (``n_set = ceil(capacity / kWarpSize)``, with the
    cache then reporting ``n_set * kWarpSize`` as its capacity).
    """
    return max(1, (int(capacity) + WAYS - 1) // WAYS)


@triton.jit
def set_of(block_id, n_sets):
    """Which set owns ``block_id``.

    Identity-modulo rather than a hash: block ids are already dense and uniform,
    and taking them mod the set count spreads *consecutive* blocks — the local
    window of one request — across different sets instead of piling them into one.
    A multiplicative hash would scatter them equally well but buys nothing here and
    costs a multiply per probe.
    """
    return block_id % n_sets


@triton.jit
def victim_way(
    AGE, set_base, gen_now, PIN_GEN,
    INS, POLICY: tl.constexpr, WAYS_C: tl.constexpr,
):
    """Choose a way in ``[0, WAYS_C)`` to displace, or -1 if the set is unusable.

    Every policy obeys two invariants, and they are the reason this is one function
    rather than three kernels:

    1. **An empty way always wins.** A way with ``age == 0`` has never been filled,
       so taking it evicts nothing. Checking this first makes a cold cache fill
       without any policy-specific work.
    2. **A way pinned this step is untouchable.** ``PIN_GEN[way] == gen_now`` means
       some entry of the *current* step already resolved to that way, so displacing
       it would pull the block out from under a read that is about to happen. Such a
       way is skipped even if the policy would prefer it. If every way in the set is
       pinned, the caller is told (-1) and reports an overflow rather than
       corrupting a live block.

    Within those, the policies differ only in the ordering key:

    * ``lru``  — the way with the smallest non-zero age (rank 1 when the set is
      full), i.e. least recently *used*.
    * ``fifo`` — the way with the smallest insertion stamp, i.e. least recently
      *inserted*; reads never change it.
    * ``full`` — never evicts. Reaching this code with a full set means the pool was
      mis-sized for the ``full`` policy, so it returns -1 and the miss is counted.
    """
    if POLICY == POLICY_FULL:
        # Cold fill only: never evict a live block. A way is available iff it has
        # never been filled (``age == 0``); ``AGE_RESERVED`` and any live stamp are
        # both non-zero, so neither is taken.
        #
        # Pinned-ness still has to be honoured even here. A way filled EARLIER IN
        # THIS SAME STEP has a live stamp, so it is already excluded by ``age != 0``
        # — but a way whose block was evicted by an earlier ``lru`` run and then
        # reused keeps its stamp too, which is why "age == 0" is the only safe test
        # for "free" and why a genuinely full set must be reported rather than
        # guessed at. Returning -1 makes the caller count an overflow, which is the
        # signal that the pool was mis-sized for a policy that cannot evict.
        best = -1
        w = 0
        while (best < 0) & (w < WAYS_C):
            age = tl.load(AGE + set_base + w)
            pin = tl.load(PIN_GEN + set_base + w)
            if (age == 0) & (pin != gen_now):
                best = w
            w += 1
        return best

    # --- lru / fifo: prefer an empty way, else the minimum ordering key ---
    #
    # One pass, one ordering key per way, so there is no ``not`` on a Triton value:
    # Python's ``not`` does NOT lower to elementwise negation of a device value (it
    # coerces to a host bool), so ``if not pinned`` silently tested the wrong thing
    # and free ways were skipped while pinned ones were chosen. Measured as heavy
    # overflow plus wrong data even with a pool 18x the demand. Compare with ``== 0``
    # instead, and fold the pin into the key so unpinned-empty < unpinned-live <
    # pinned without any boolean algebra:
    #   pinned      -> KEY_PINNED (never chosen unless nothing else exists)
    #   empty       -> 0          (always preferred; evicts nothing)
    #   occupied    -> age (lru) or insert stamp (fifo)
    best = -1
    best_key = 0x7FFFFFFF
    w = 0
    while w < WAYS_C:
        age = tl.load(AGE + set_base + w)
        pin = tl.load(PIN_GEN + set_base + w)
        if POLICY == POLICY_FIFO:
            live_key = tl.load(INS + set_base + w)
        else:
            live_key = age
        # ``age == 0`` is empty; +1 keeps a live key strictly above an empty one even
        # when its own key is 0 (a way inserted at stamp 0).
        key = tl.where(age == 0, 0, live_key + 1)
        key = tl.where(pin == gen_now, 0x7FFFFFFE, key)
        # Reserved way: strictly worse than any pinned way, so it is only ever
        # "chosen" when the set is entirely unusable — and the caller rejects that.
        key = tl.where(age == _AGE_RESERVED_C, 0x7FFFFFFF, key)
        if key < best_key:
            best = w
            best_key = key
        w += 1
    # Every way pinned this step -> unusable; the caller counts an overflow.
    if best_key >= 0x7FFFFFFE:
        return -1
    return best


@triton.jit
def touch_way(AGE, set_base, way, TICK, WAYS_C: tl.constexpr, POLICY: tl.constexpr):
    """Record a use of ``way`` for the recency ordering.

    LRU only. Promotes ``way`` to the most-recently-used rank and decrements every
    way that outranked it, which keeps the ages a permutation of ``1..k`` over the
    ``k`` occupied ways — OneFlow's scheme, and the reason eviction is "find rank 1"
    instead of "find the minimum of k timestamps".

    ``fifo`` and ``full`` deliberately do nothing: for them a read must not change
    residency, which is what makes ``fifo`` immune to one scan-heavy step flushing
    the whole pool.
    """
    if (POLICY == POLICY_LRU) or (POLICY == POLICY_BLOCK_LRU):
        # Promote with a single monotone stamp instead of OneFlow's rank shuffle.
        #
        # OneFlow can renumber a set's ranks (decrement everything above the hit way,
        # set the hit way to ``ways``) because a warp holds the set's mutex and owns
        # all 32 ways in registers. This kernel is scalar per block-table entry, with
        # many programs touching one set concurrently and no lock, so the shuffle's
        # read-modify-write over the other ways races: two hits in the same set
        # interleave and leave ranks that are no longer a permutation. Measured, the
        # corruption inverted the policy — LRU re-fetched MORE than FIFO (160 vs 112
        # copies) because promoted ways were being scored as least-recently-used.
        #
        # A monotone counter is race-free and orders the same way: bigger = more
        # recent, so ``victim_way``'s "smallest key" is still the least recently used.
        # It needs no coordination between ways, at the cost of ages no longer being
        # a compact 1..k permutation — which nothing depends on.
        tl.atomic_max(AGE + set_base + way, tl.atomic_add(TICK, 1) & 0x3FFFFFFF)


__all__ = [
    "WAYS", "POLICIES", "AGE_RESERVED", "POLICY_LRU", "POLICY_BLOCK_LRU",
    "POLICY_FIFO", "POLICY_FULL",
    "policy_code", "n_sets_for", "set_of", "victim_way", "touch_way",
]
