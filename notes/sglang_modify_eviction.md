# Modifying sglang's KV-cache eviction policy

This is a recipe for changing **how / when sglang evicts KV cache**. All
file paths are under
`third_party/sglang/v0.5.16/sglang/python/sglang/`.

> **Written against the vendored sglang 0.4.9 tree**, which is no longer in the
> repo (0.5.16 is the only vendored release as of the `v0.9` branch). The four
> knobs below still exist, but some have moved file — notably the KV-pool and
> pool-sizing code, which 0.5.16 split into `mem_cache/kv_cache_configurator.py`
> and `model_executor/pool_configurator.py`. Search by symbol name, not path.

There are **four independent knobs** you can turn. Pick the one that
matches the change you want — most users only need #1 (swap the
selection rule) or #2 (change the trigger).

| You want to change… | Edit |
| --- | --- |
| 1. Which leaf gets evicted (LRU → LFU / size / age / cost…) | `TreeNode.__lt__` or `RadixCache.evict` in `srt/mem_cache/radix_cache.py` |
| 2. *When* eviction is triggered (admission threshold, retract threshold) | `PrefillAdder.rem_total_tokens` in `srt/managers/schedule_policy.py`, `Scheduler.update_running_batch` in `srt/managers/scheduler.py` |
| 3. What is *protected* from eviction (lock-ref scope) | `RadixCache.inc_lock_ref` / `dec_lock_ref`, plus call sites in `PrefillAdder.add_one_req` |
| 4. The whole cache class (radix → custom tree, LRU list, none) | Subclass `BasePrefixCache`, wire it in `Scheduler.__init__` (`srt/managers/scheduler.py:599`) |

---

## Anatomy: the eviction call graph (for context)

Eviction is **pull-based** — there is no background reaper. It only
runs from two call sites:

```
PrefillAdder.add_one_req          ← admission needs more slots
ScheduleBatch.retract_decode      ← decode tick ran out of slots
        │
        ▼
RadixCache.evict(num_tokens)            ← srt/mem_cache/radix_cache.py:267
        ├── _collect_leaves()           ← gather candidate set
        ├── heapq.heapify(leaves)       ← order by TreeNode.__lt__
        └── while num_evicted < num_tokens:
                x = heapq.heappop(leaves)
                if x.lock_ref > 0: continue          # skip in-use
                allocator.free(x.value)              # release KV slots
                _delete_leaf(x)
                if x.parent.children == {}:
                    heapq.heappush(leaves, x.parent) # parent is now a leaf
```

Three pluggable points are visible in this loop: the **comparator**
(`__lt__`), the **predicate** (`lock_ref > 0`), and the **trigger**
(who calls `evict`).

---

## Knob 1 — Change *which* leaf is evicted (selection rule)

The current rule is **LRU on `last_access_time`**, encoded in one line:

```python
# srt/mem_cache/radix_cache.py:72
def __lt__(self, other: "TreeNode"):
    return self.last_access_time < other.last_access_time
```

The min-heap pops the smallest `__lt__`, so whatever you put here
becomes the eviction priority. Examples:

### 1a. LFU (least-frequently-used)

Add a `hit_count` field on `TreeNode` and bump it inside
`match_prefix` (line ~341, where `last_access_time` is already
updated). Then:

```python
def __lt__(self, other):
    return self.hit_count < other.hit_count
```

### 1b. Size-weighted LRU (free big leaves first when ties)

```python
def __lt__(self, other):
    return (self.last_access_time, -len(self.value)) < (other.last_access_time, -len(other.value))
```

### 1c. Cost-aware (cheap-to-recompute leaves first)

If you can stash the original prefill cost on the node
(`self.prefill_cost = ...` in the insert path), the comparator becomes
`self.prefill_cost / max(age, 1) < other.prefill_cost / max(age, 1)`.

### 1d. Replacing the loop entirely

If your policy doesn't fit a heap (e.g. you want to scan
non-leaves, or use a global LRU list across the whole tree), override
`evict()` itself rather than `__lt__`. Keep the contract:

- Free at least `num_tokens` of slots if possible.
- Never free a node with `lock_ref > 0` (in-flight requests rely on
  this — see knob 3 for changing it).
- Call `self.token_to_kv_pool_allocator.free(x.value)` and
  `self._delete_leaf(x)` for every reclaimed node so the radix tree
  invariants hold.
- Update `evictable_size_` correctly (`_delete_leaf` does this).

The cleanest way to ship this is to subclass — see knob 4.

---

## Knob 2 — Change *when* eviction triggers (admission / retraction thresholds)

Eviction has **two upstream triggers**. Make admission stricter or
retraction earlier and you evict more often (and earlier); make them
looser and you evict less often (but risk OOM and retracts).

### 2a. Admission threshold — `PrefillAdder.rem_total_tokens`

`srt/managers/schedule_policy.py:314`:

```python
@property
def rem_total_tokens(self):
    return (
        self.token_to_kv_pool_allocator.available_size()
        + self.tree_cache.evictable_size()
        - self.rem_total_token_offset
    )
```

This is the budget `add_one_req` checks against. Two useful tweaks:

- **Reserve a safety margin** (evict more eagerly to avoid retracts):
  ```python
  - self.rem_total_token_offset
  - self.safety_reserve   # e.g. running_batch.batch_size * 64
  ```
- **Refuse to count evictable cache** (evict only on real free space —
  more retracts, more cache reuse):
  ```python
  return self.token_to_kv_pool_allocator.available_size() - self.rem_total_token_offset
  ```

### 2b. Retract threshold — `Scheduler.update_running_batch`

`srt/managers/scheduler.py:1677` (approx):

```python
if self.token_to_kv_pool_allocator.available_size() < required_tokens or TEST_RETRACT:
    retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
```

`required_tokens` ≈ `len(running_batch)` per decode step. Multiplying
it makes retract fire earlier (so eviction runs earlier inside
`retract_decode`); shrinking it pushes everything to the brink of OOM.

### 2c. Add a periodic background evict (anti-pattern, mention only)

You *could* call `self.tree_cache.evict(K)` from the idle branch of
`event_loop_normal`, but it usually hurts: you free prefixes that the
next request would have hit. If you do this, gate it on a watermark
(`evictable_size_ > X`).

---

## Knob 3 — Change *what is protected* (lock-ref semantics)

`inc_lock_ref` / `dec_lock_ref` (`srt/mem_cache/radix_cache.py:292`,
`306`) walk node→root incrementing a counter. `evict()` skips any node
with `lock_ref > 0`. The lock is taken once per active request:
`PrefillAdder._lock_node` (line 377 in `schedule_policy.py`) and
`add_one_req` (line 502).

You can change protection in three ways:

- **Time-bounded protection** — let `evict()` ignore the lock if the
  node hasn't been touched in T seconds. Useful if you suspect a
  request leaked a lock.
  ```python
  if x.lock_ref > 0 and (now - x.last_access_time) < T:
      continue
  ```
- **Per-request priority** — store the request's priority on the
  `TreeNode` when the lock is acquired and let high-priority evict
  low-priority *locked* nodes (force-retract the holder; you'll have
  to push the holder back to `waiting_queue`, mirroring
  `retract_decode`).
- **Disable lock entirely for opportunistic prefixes** — useful if you
  want a "soft" cache layer that never blocks admission. Add a
  `node.soft = True` flag at insert time and have `inc_lock_ref` skip
  it.

---

## Knob 4 — Plug in a whole new cache class

The cleanest, lowest-blast-radius approach: subclass `RadixCache` (or
`BasePrefixCache` for a from-scratch impl), then swap it in.

### Step 1 — Subclass

```python
# new file: srt/mem_cache/my_cache.py
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

class MyEvictionRadixCache(RadixCache):
    """LRU + size-weighted tiebreaker, with a 5% headroom reserve."""

    def evict(self, num_tokens: int):
        # Always evict 5% more than asked, to amortize calls.
        super().evict(int(num_tokens * 1.05))
```

Or override `_collect_leaves` / the heap comparator on a subclass of
`TreeNode` — but watch out: `TreeNode` is constructed in many places
inside `radix_cache.py`, so you'd also need to patch those construction
sites. Overriding `evict()` is usually less invasive.

### Step 2 — Wire it into the scheduler

`srt/managers/scheduler.py:599`:

```python
self.tree_cache = RadixCache(
    req_to_token_pool=self.req_to_token_pool,
    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
    page_size=self.page_size,
    disable=server_args.disable_radix_cache,
    enable_kv_cache_events=self.enable_kv_cache_events,
)
```

Replace with:

```python
from sglang.srt.mem_cache.my_cache import MyEvictionRadixCache
self.tree_cache = MyEvictionRadixCache(
    req_to_token_pool=self.req_to_token_pool,
    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
    page_size=self.page_size,
    disable=server_args.disable_radix_cache,
    enable_kv_cache_events=self.enable_kv_cache_events,
)
```

For a config-driven swap, gate it on a new server arg (e.g.
`server_args.cache_eviction_policy == "size_lru"`) so you can A/B
test without forking.

### Step 3 — Honour the `BasePrefixCache` contract

If you go further and replace `RadixCache` with a from-scratch
implementation, your class **must** provide every method in
`srt/mem_cache/base_prefix_cache.py` (lines 31–97):

- `reset`, `match_prefix`, `cache_finished_req`, `cache_unfinished_req`,
  `evict`, `inc_lock_ref`, `dec_lock_ref`, `evictable_size`,
  `protected_size`, `total_size`.

`PrefillAdder` reads `evictable_size()` and uses
`inc_lock_ref` / `dec_lock_ref`, so getting those right is what
keeps admission and retraction honest.

---

## Quick cookbook

| Goal | Minimal change |
| --- | --- |
| LFU instead of LRU | Add `hit_count` to `TreeNode`, change `__lt__` in `radix_cache.py:72` |
| Evict big leaves first on a tie | Edit `__lt__` to return `(time, -len(value))` tuple |
| Keep more headroom (avoid retracts) | Subtract a margin in `PrefillAdder.rem_total_tokens` (`schedule_policy.py:314`) |
| Stop counting evictable cache as "free" | Drop `+ tree_cache.evictable_size()` from `rem_total_tokens` |
| Allow eviction of locked nodes after T sec | Add age check in `RadixCache.evict` skip clause |
| Pluggable policy via flag | New subclass + new server arg + `if/elif` at `scheduler.py:599` |
| Disable eviction entirely | `--disable-radix-cache` on the CLI; scheduler then uses `ChunkCache` (`scheduler.py:574`), which frees on request finish but never evicts to share |

---

## Things to test after any change

1. **Memory leak detection** — `Scheduler.check_memory()` runs every
   idle iteration and crashes if `evictable_size + protected_size +
   allocator.used != max_total_num_tokens`. Eviction bugs surface
   here first.
2. **Retract-decode behaviour** — set
   `SGLANG_TEST_RETRACT=1` (env var read by `update_running_batch` via
   `TEST_RETRACT`) to force retraction every step. A correct policy
   still makes forward progress.
3. **Prefix-hit rate** — `Scheduler` logs cache hit rate periodically.
   A "more aggressive" policy that drops the hit rate below baseline
   is usually a regression even if throughput looks fine on
   short-prompt benchmarks.
