# How sglang schedules requests, evicts KV cache, and retracts decode

A code-pointer guide for `third_party/sglang/v0.5.16/sglang/python/sglang/`. Every
file path below is relative to that root.

> **Line numbers (and some paths) are from the vendored sglang 0.4.9 tree**,
> which is no longer in the repo — 0.5.16 is the only vendored release as of the
> `v0.9` branch. Use the symbol names, which are stable, as the anchor; the
> scheduler grew substantially between the two releases.

There are four behaviours to understand, and they are all driven by a
single actor — the `Scheduler` in `srt/managers/scheduler.py`:

```
                              ┌────────────────────────┐
   recv_requests ─────────►   │  waiting_queue (List)  │
                              └──────────┬─────────────┘
                                         │  PrefillAdder.add_one_req
                                         ▼
   ┌──────────────────┐  merge   ┌───────────────────────┐
   │ new prefill batch│ ───────► │  running_batch        │ ──► forward
   └──────────────────┘          │  (ScheduleBatch)      │
                                 └────────┬──────────────┘
                                          │ retract_decode (OOM)
                                          ▼
                                back to waiting_queue
```

KV memory lives in `TokenToKVPoolAllocator` (raw slots) plus
`RadixCache` (a prefix tree that owns evictable / locked tokens). The
admission test, the eviction loop, and the retraction loop all read the
same accounting numbers, so the four sections below are tightly coupled
even though they live in different files.

---

## 1. The scheduling main loop

**File:** `srt/managers/scheduler.py`

The scheduler is a single long-lived process per TP rank. After
`Scheduler.__init__` finishes wiring the worker, memory pools and tree
cache, it calls one of three event loops:

| Method | Line | When it runs |
| --- | --- | --- |
| `Scheduler.event_loop_normal()` | ~752 | default, no overlap |
| `Scheduler.event_loop_overlap()` | ~773 | `--enable-overlap-schedule`, CPU/GPU overlap |
| `Scheduler.event_loop_pp()` | ~818 | pipeline-parallel |

All three share the same skeleton — only the placement of the
forward call vs. the next batch's preparation differs. Every iteration:

```
recv_requests()                     # pull new ReqInput from the IPC queue
process_input_requests(recv_reqs)   # tokenise + push onto self.waiting_queue
get_next_batch_to_run()             # decide: prefill, decode, or idle
run_batch(batch)                    # tp_worker.forward_batch_generation
process_batch_result(batch, result) # sample, detokenise, send to client
check_memory()                      # leak detection / logging when idle
```

`get_next_batch_to_run()` (~line 1454) is the dispatch point. It first
calls `get_new_batch_prefill()` to try to admit something from the
waiting queue. If a prefill batch comes back, that batch is run (or
*chunk-prefill*-merged into the running batch in mixed mode). Otherwise
the existing `running_batch` is decoded one step via
`update_running_batch()` (~line 1677).

`run_batch()` (~line 1706) is a thin wrapper over
`tp_worker.forward_batch_generation()` and produces a
`GenerationBatchResult`.

---

## 2. Admitting a new request into the batch

**Files:** `srt/managers/scheduler.py`, `srt/managers/schedule_policy.py`

There are two layers: the *scheduler* loop that drains the queue, and
the *PrefillAdder* that owns the per-request admission predicate.

### 2.1 Draining the waiting queue

`Scheduler.get_new_batch_prefill()` (~line 1519):

1. Bails immediately if `running_batch.batch_is_full` or the
   waiting queue is empty (~line 1525).
2. Builds a `PrefillAdder` (~line 1547) seeded with the current free
   token budget, the chunked-prefill chunk size, and the running batch.
3. Sorts the queue by the configured policy (LPM / DFS / FCFS / LOF) via
   `SchedulePolicy.calc_priority()` (~line 97 in `schedule_policy.py`).
4. Walks the sorted queue calling `adder.add_one_req(req)` (~line 1591).
   Stops at the first `AddReqResult.NO_TOKEN` or when the running-batch
   cap is hit.
5. Materialises a new `ScheduleBatch` from `adder.can_run_list`, or
   returns `None` if nothing fit.

### 2.2 The admission predicate

`PrefillAdder.add_one_req()` in `srt/managers/schedule_policy.py`
(~line 462). The check is:

```python
total_tokens = req.extend_input_len + min(
    req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
)
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN          # KV pool would overflow
if real_input_tokens >= self.rem_input_tokens and self.can_run_list:
    return AddReqResult.OTHER             # prefill chunk budget exhausted
```

The two budgets are:

- `rem_total_tokens` (~line 315) =
  `allocator.available_size() + tree_cache.evictable_size() − offset`.
  Note that **evictable cache is counted as free** — admission can
  succeed by *implicitly* evicting LRU prefix-cache nodes (see §3).
- `rem_input_tokens` is bounded by `max_prefill_tokens` (per-pass
  prefill cap) and by `chunked_prefill_size` (per-chunk cap).

If the request's input is longer than `rem_chunk_tokens` (~line 517), it
is truncated and recorded as `new_chunked_req`, becoming the chunked
prefill that the next iteration continues.

### 2.3 The hard caps that gate this

Set in `Scheduler.__init__` (~line 366):

| Field | Source |
| --- | --- |
| `max_running_requests` | `tp_worker.get_worker_info()` / `--max-running-requests` |
| `max_prefill_tokens`   | `--max-prefill-tokens` |
| `chunked_prefill_size` | `--chunked-prefill-size` |
| `max_total_num_tokens` | derived from KV pool size |

---

## 3. Evicting KV cache

**File:** `srt/mem_cache/radix_cache.py` (and `base_prefix_cache.py`)

`RadixCache` is the prefix tree whose leaves own contiguous KV slots.
Eviction is **leaf-only LRU with lock-ref protection**.

### 3.1 The evict loop

`RadixCache.evict(num_tokens)` (~line 267):

```python
leaves = self._collect_leaves()
heapq.heapify(leaves)                # min-heap by last_access_time
while num_evicted < num_tokens and leaves:
    x = heapq.heappop(leaves)
    if x.lock_ref > 0:               # in use by an active req → skip
        continue
    self.token_to_kv_pool_allocator.free(x.value)
    num_evicted += len(x.value)
    self._delete_leaf(x)
    if x.parent.children == {}:      # parent is now a leaf → re-push
        heapq.heappush(leaves, x.parent)
```

So eviction is **strictly LRU among unlocked leaves**, and walks
upward as it peels nodes. There is no time-based / size-based
threshold — eviction is pull-based, triggered by the admission and
retraction paths when they need slots.

### 3.2 Lock-ref accounting (what protects an in-flight request)

- `inc_lock_ref(node)` (~line 292): walks `node → root`,
  `node.lock_ref += 1`. The first lock on a node moves its bytes from
  `evictable_size_` to `protected_size_`.
- `dec_lock_ref(node)` (~line 306): reverse direction; releasing the
  last lock returns the bytes to `evictable_size_`.

`PrefillAdder` calls `inc_lock_ref(req.last_node)` (~line 502 in
`schedule_policy.py`) when it admits a request, and the scheduler calls
`dec_lock_ref` when the request finishes or is retracted. That is what
keeps `evict()` from yanking the prefix out from under a running req.

### 3.3 Where eviction is actually called

Eviction is **never called by a free-running background thread**. It is
called from exactly two places:

1. `PrefillAdder.add_one_req()` — implicitly, because
   `rem_total_tokens` includes `evictable_size()` and the scheduler
   then calls `tree_cache.evict()` while assembling the prefill batch.
2. `ScheduleBatch.retract_decode()` — explicitly, after a request is
   ejected (see §4), to top up the freelist for the survivors.

### 3.4 Caching a partial decode

`RadixCache.cache_unfinished_req(req)` (~line 217) is the bridge between
decode state and the tree:

- Inserts the partial output into the radix tree (`self.insert`).
- Frees any token-to-KV indices that didn't end up in the tree.
- `dec_lock_ref(old_last_node)` then `inc_lock_ref(new_last_node)` so
  the lock follows the request as it grows.

`cache_finished_req(req)` does the same on the success path.

---

## 4. Retracting (cancelling) a running request

**File:** `srt/managers/schedule_batch.py` (with the trigger in
`scheduler.py`)

Retraction is sglang's escape hatch when decode runs the KV pool dry
mid-step. The retracted request is **rolled back to its prompt** and
pushed onto the waiting queue, so the next `get_new_batch_prefill()`
will re-prefill it.

### 4.1 Trigger

`Scheduler.update_running_batch()` (~line 1677) checks memory after each
decode step:

```python
if (self.token_to_kv_pool_allocator.available_size()
        < required_tokens
        or TEST_RETRACT):
    retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
    self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
```

`required_tokens` is roughly `len(running_batch) * 1` for one decode
step (plus a margin), so this fires the moment a decode tick can't be
guaranteed to land somewhere.

### 4.2 The retract loop

`ScheduleBatch.retract_decode(server_args)` (~line 1375):

```python
sorted_indices = sort by (output_ids_len, -input_ids_len)  # evict longest-decoded first
while available_size() < required_tokens:
    req = sorted_indices.pop()
    retracted_reqs.append(req)
    if using RadixCache:
        tree_cache.dec_lock_ref(req.last_node)   # unlock cache
        tree_cache.evict(residual_tokens)        # then LRU-evict to top up
    else:                                        # ChunkCache (no prefix sharing)
        token_to_kv_pool_allocator.free(req.kv_indices)
    req.reset_for_retract()                      # output_ids = [], state ← waiting
self.filter_batch(keep_indices=sorted_indices)
```

Sort order — *most output, least input* first — is deliberate: it
maximises tokens reclaimed per request retracted while preferring to
keep the requests that have invested the least decode work.

### 4.3 What "re-prefill later" looks like

`req.reset_for_retract()` clears `output_ids` and resets the request to
the post-tokenise / pre-prefill state. The scheduler then calls
`_extend_requests_to_queue(retracted_reqs, is_retracted=True)` (~line
1692 in `scheduler.py`), which puts the request back on
`self.waiting_queue`. On the next iteration, `get_new_batch_prefill()`
admits it like any other waiting request and re-runs prefill from the
prompt — the prefix-cache hit (if any) makes that re-prefill cheap, but
the original output tokens are gone and will be regenerated.

There is no separate "cancel" path for *successful* completion or for
client-side cancellation — completion goes through
`process_batch_result` → `cache_finished_req` → `dec_lock_ref`, and a
client-side abort surfaces as a `BatchTokenIDOut` with the abort flag,
processed in `process_input_requests`.

---

## File / symbol cheat-sheet

| Topic | File | Key symbols (line) |
| --- | --- | --- |
| Event loops | `srt/managers/scheduler.py` | `Scheduler.event_loop_normal` (~752), `event_loop_overlap` (~773), `event_loop_pp` (~818) |
| Dispatch | `srt/managers/scheduler.py` | `Scheduler.get_next_batch_to_run` (~1454), `run_batch` (~1706) |
| Admit from queue | `srt/managers/scheduler.py` | `Scheduler.get_new_batch_prefill` (~1519) |
| Admission test | `srt/managers/schedule_policy.py` | `PrefillAdder.add_one_req` (~462), `rem_total_tokens` accounting (~315) |
| Queue priority | `srt/managers/schedule_policy.py` | `SchedulePolicy.calc_priority` (~97), `CacheAwarePolicy`, `CacheAgnosticPolicy` |
| KV eviction | `srt/mem_cache/radix_cache.py` | `RadixCache.evict` (~267), `inc_lock_ref` (~292), `dec_lock_ref` (~306), `cache_unfinished_req` (~217) |
| Retract trigger | `srt/managers/scheduler.py` | `Scheduler.update_running_batch` (~1677), `_extend_requests_to_queue` (~1692) |
| Retract loop | `srt/managers/schedule_batch.py` | `ScheduleBatch.retract_decode` (~1375), `Req.reset_for_retract` |
| KV pools | `srt/mem_cache/memory_pool.py`, `srt/mem_cache/allocator.py` | `ReqToTokenPool`, `TokenToKVPoolAllocator` |

### Reading order if you want to trace a request end-to-end

1. `srt/managers/tokenizer_manager.py` — request enters the system.
2. `Scheduler.recv_requests` → `process_input_requests` —
   it joins `waiting_queue`.
3. `get_new_batch_prefill` → `PrefillAdder.add_one_req` —
   admission and KV reservation.
4. `run_batch` → `tp_worker.forward_batch_generation` — first prefill.
5. `update_running_batch` — decode tick; possibly retracts.
6. `process_batch_result` → `cache_finished_req` /
   `cache_unfinished_req` — KV either persists in the radix tree
   for prefix reuse or is freed.
