"""The request-bound cache domain: sizing, policy, and INT4 riding on it.

The domain's whole reason to exist is that its footprint is decided by CONCURRENCY while the page
domain's is decided by CONTEXT. So section 1 is not a formality -- it is the claim, and the
alternative it replaces (a per-block bf16 mirror) measured a 1.28x regression against plain bf16.

What each section guards:

1. **Footprint is constant in context length.** Payload bytes must not mention the key count. If
   they do, the domain has silently become a second page domain.
2. **Policy validation refuses the unimplemented options.** ``residual_n`` and ``evict_lru`` are
   declared to document the design space; accepting them would run the nearest implemented policy
   instead, and for ``evict_lru`` that means quantizing a partly written block.
3. **The claim/release cycle maintains its invariant**: a slot whose owner is free is never still
   reachable through the map. Violating it hands one row to a claimant and a stale reader at once.
4. **INT4 rides ON the domain, it does not copy it.** ``arena.slot_of`` must BE the domain's tensor,
   because that same tensor is what a compiled SLOTTED read is addressed through -- two maps for one
   domain can disagree, and the codegen reads whichever it was handed.
5. **Keys from different requests do not alias**, since keys are block ids from a shared pool.
6. **A domain-backed arena still captures under cudagraph.** The whole point of allocating up front.

Run: python examples/misc/test_request_domain.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_torch.engine.sgl.int4_arena import (  # noqa: E402
    CAPTURE_CEILING, Int4Arena, arena_slots, int4_request_domain,
)
from vortex_torch.engine.sgl.int4_store import (  # noqa: E402
    K_SCALE, STAGE_K, STAGE_V, V_SCALE, int4_cache_meta, int4_request_cache_meta,
)
from vortex_torch.engine.sgl.request_domain import (  # noqa: E402
    SLOT_FREE, RequestCacheDomain, request_domain_for,
)
from vortex_torch.engine.sgl.request_policy import resolve  # noqa: E402

fails = 0
BT, D, NKV, PAGE = 32, 128, 2, 32

print("1. footprint is decided by CONCURRENCY, not by context length")
meta = int4_request_cache_meta(BT, D)
rows = []
for num_keys in (4, 256, 4096, 1 << 20):
    dom = RequestCacheDomain(num_keys, arena_slots(NKV))
    rows.append((num_keys, dom.payload_bytes(meta), dom.nbytes_maps()))
    del dom
payloads = {p for _, p, _ in rows}
ok = len(payloads) == 1
for num_keys, payload, maps in rows:
    print(f"   {num_keys:>8d} keys: payload {payload / 2**20:>7.1f} MiB   maps "
          f"{maps / 2**10:>8.1f} KiB")
print(f"   payload identical across a 262144x range in context: "
      f"{'ok' if ok else '<-- FAIL: the domain scales with context'}")
if not ok:
    fails += 1
# The maps DO scale (slot_of is one int32 per key) and that is fine -- 4 B/block against bf16's
# 16896 B/block. State the ratio so nobody "optimizes" it into a hash table.
_, _, big_maps = rows[-1]
print(f"   maps scale, at 4 B/key = {4 / (2 * BT * D * 2) * 100:.3f}% of a bf16 block: fine")

print("\n2. policy validation")
ok = resolve("complete", "drop") == (0, 0)
print(f"   complete + drop resolves: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
for retention, overflow, why in (
        ("residual_n", "drop", "residual_n retention"),
        ("complete", "evict_lru", "evict_lru overflow")):
    try:
        resolve(retention, overflow)
        print(f"   {why} accepted <-- FAIL (would run a different policy silently)")
        fails += 1
    except NotImplementedError as e:
        has_reason = "silent" in str(e) or "starves" in str(e)
        print(f"   {why} refused WITH a reason: {'ok' if has_reason else '<-- FAIL (no reason)'}")
        if not has_reason:
            fails += 1
for bad in (("nonsense", "drop"), ("complete", "nonsense")):
    try:
        resolve(*bad)
        print(f"   typo {bad} accepted <-- FAIL")
        fails += 1
    except ValueError:
        print(f"   typo {bad} rejected: ok")

print("\n3. claim / release maintains the map<->owner invariant")
dom = RequestCacheDomain(64, 4)
dom.slot_of[7] = 2
dom.owner_of[2] = 7
ok = dom.slot_for(7) == 2 and dom.occupancy() == 1
print(f"   occupied: slot_for(7)={dom.slot_for(7)} occupancy={dom.occupancy()} "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
dom.release(7)
ok = dom.slot_for(7) == SLOT_FREE and int(dom.owner_of[2]) == -1 and dom.occupancy() == 0
print(f"   after release: map cleared AND owner freed (never one without the other): "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
dom.release(7)   # idempotent; a double release must not free someone else's slot
dom.slot_of[9] = 2
dom.owner_of[2] = 9
dom.release(7)
ok = dom.slot_for(9) == 2
print(f"   releasing an already-free key does not steal another key's slot: "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = False
try:
    RequestCacheDomain(64, 0)
except ValueError:
    ok = True
print(f"   a zero-slot domain is rejected (it would decline every token): "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n4. INT4 rides ON the domain -- one map, not a copy")
arena = Int4Arena(64, 8, BT, D)
ok = arena.slot_of.data_ptr() == arena.domain.slot_of.data_ptr()
print(f"   arena.slot_of IS domain.slot_of (same storage): {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
    print("      a second map means a compiled SLOTTED read can be addressed through the "
          "stale one")
ok = arena.owner_of.data_ptr() == arena.domain.owner_of.data_ptr()
print(f"   arena.owner_of IS domain.owner_of: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = (arena.domain.retention, arena.domain.overflow) == ("complete", "drop")
print(f"   policy is complete+drop (drop is required, not a default): "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
d2 = int4_request_domain(1024, NKV)
ok = d2.n_slots == CAPTURE_CEILING * NKV
print(f"   int4_request_domain sizes from the capture ceiling: {d2.n_slots} slots "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n5. the staged payload allocates at n_slots, and stages through the domain's map")
cache_meta = int4_cache_meta(BT, D)
cache = {n: torch.zeros((64, s[0], s[1]), dtype=dt, device="cuda")
         for n, (s, dt) in cache_meta.items()}
payload = arena.domain.allocate_payload(int4_request_cache_meta(BT, D))
ok = all(tuple(t.shape) == (arena.n_slots, BT, D) for t in payload.values())
print(f"   payload shapes {[tuple(t.shape) for t in payload.values()]} lead with n_slots="
      f"{arena.n_slots} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

torch.manual_seed(0)
kv = ((torch.randn(BT, NKV, D, device="cuda") * 0.5).to(torch.bfloat16),
      (torch.randn(BT, NKV, D, device="cuda") * 0.5).to(torch.bfloat16))
for t in range(BT):
    loc = torch.tensor([t], dtype=torch.int64, device="cuda")
    arena.stage(payload[STAGE_K], payload[STAGE_V], kv[0][t:t + 1], kv[1][t:t + 1],
                loc, cache, PAGE, fused=True, k_scale_name=K_SCALE, v_scale_name=V_SCALE)
torch.cuda.synchronize()
c = arena.counters()
ok = c["migrated"] == NKV and c["declined"] == 0 and arena.domain.occupancy() == 0
print(f"   {BT} tokens staged through the domain: {c} occupancy={arena.domain.occupancy()} "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n6. keys from different requests do not alias")
# Two far-apart page regions, as two concurrent requests get from sglang's pool.
arena2 = Int4Arena(4096, 16, BT, D)
cache2 = {n: torch.zeros((4096, s[0], s[1]), dtype=dt, device="cuda")
          for n, (s, dt) in cache_meta.items()}
pay2 = arena2.domain.allocate_payload(int4_request_cache_meta(BT, D))
for base in (0, 2048):
    for t in range(BT):
        loc = torch.tensor([base + t], dtype=torch.int64, device="cuda")
        arena2.stage(pay2[STAGE_K], pay2[STAGE_V], kv[0][t:t + 1], kv[1][t:t + 1],
                     loc, cache2, PAGE, fused=True, k_scale_name=K_SCALE, v_scale_name=V_SCALE)
torch.cuda.synchronize()
c = arena2.counters()
ok = c["migrated"] == 2 * NKV and c["declined"] == 0
print(f"   two disjoint key ranges: {c} (want migrated={2 * NKV}) {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n7. a domain-backed arena still captures under cudagraph")
arena3 = Int4Arena(64, 8, BT, D)
cache3 = {n: torch.zeros((64, s[0], s[1]), dtype=dt, device="cuda")
          for n, (s, dt) in cache_meta.items()}
pay3 = arena3.domain.allocate_payload(int4_request_cache_meta(BT, D))
loc = torch.zeros(1, dtype=torch.int64, device="cuda")
nk = torch.zeros(1, NKV, D, dtype=torch.bfloat16, device="cuda")
nv = torch.zeros_like(nk)


def run():
    arena3.stage(pay3[STAGE_K], pay3[STAGE_V], nk, nv, loc, cache3, PAGE,
                 fused=True, k_scale_name=K_SCALE, v_scale_name=V_SCALE)


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        run()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        run()
    arena3.domain.reset()
    arena3.mask_of.zero_()
    arena3.stats.zero_()
    for t in range(BT):
        loc.fill_(t)
        nk.copy_(kv[0][t:t + 1]); nv.copy_(kv[1][t:t + 1])
        g.replay()
    torch.cuda.synchronize()
    c = arena3.counters()
    ok = c["migrated"] == NKV and c["declined"] == 0
    print(f"   captured and replayed {BT}x: {c} {'ok' if ok else '<-- FAIL'}")
    if not ok:
        fails += 1
except Exception as e:
    print(f"   <-- FAIL: capture raised {type(e).__name__}: {e}")
    fails += 1

print("\n8. request_domain_for sizes by (concurrency x kv heads)")
d = request_domain_for(1024, 64, 4)
ok = d.n_slots == 256
print(f"   64 rows x 4 heads -> {d.n_slots} slots {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("REQUEST DOMAIN TESTS PASS")
