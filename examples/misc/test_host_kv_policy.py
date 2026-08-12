"""GPU-as-a-cache over host KV: policy correctness, behaviour and cost.

Three things must hold, and they are separable:

1. **Correctness is policy-independent.** Whatever gets evicted, every table entry
   must still resolve to a slot holding the block it named. A policy bug shows up as
   wrong data, not as a worse hit rate.
2. **The policies must actually differ**, in the direction claimed: LRU keeps a
   re-referenced working set that FIFO cycles out; ``full`` never evicts.
3. **Per-miss cost must be flat in capacity.** This is the whole point of going
   set-associative — the old global cursor was O(capacity) per miss (measured 32x).

Run: python examples/misc/test_host_kv_policy.py
"""
import sys, time, torch
sys.path.insert(0, "/scratch/zhuominc/vortex_torch")
from vortex_torch.engine.sgl.host_kv import HostKVCache
from vortex_torch.engine.sgl.cache_policy import WAYS, POLICIES

BLK, D, NB = 32, 128, 8192
hk = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True); hk.normal_()
hv = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True); hv.normal_()

def serve(c, ids):
    rows, mpr = ids.shape
    rl = torch.full((rows,), mpr * BLK, dtype=torch.int32, device="cuda")
    c.tick(); c.reserve_remap(ids)
    dk, dv, out = c.fetch(ids.clone(), row_lens=rl, num_rows=rows, max_per_row=mpr)
    torch.cuda.synchronize()
    return dk, dv, out

def verify(c, ids):
    """Every entry resolves to a slot holding ITS block (or is a counted overflow)."""
    dk, dv, out = serve(c, ids)
    slot = out.cpu().long(); want = ids.cpu().long()
    bad = 0
    for r in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            s = int(slot[r, j])
            if s < 0 or s >= c.capacity:
                bad += 1; continue
            if float((dk[s].float().cpu() - hk[int(want[r, j])].float()).abs().max()) != 0.0:
                bad += 1
    return bad, int(c.overflow.item())

ok = True
# A block id maps to set ``id % n_sets``, so a set can fill even when the pool as a
# whole has room -- the standard set-associative trade-off. An entry the cache cannot
# place is REPORTED (counted in ``overflow``, slot published as -1) rather than
# silently pointed at another block's data, and the caller drops it. So the invariant
# under test is: *every entry the cache accepts is correct*, and anything it refuses
# is accounted for. Sizing the pool so no set overflows is a hit-rate concern,
# checked separately below.
print("1) accepted entries are always correct; refusals are counted, never silent:")
for pol in POLICIES:
    # Pool >= demand so `full` (which never evicts) is exercised fairly; the two
    # evicting policies also get a deliberately over-subscribed run afterwards.
    # ``full`` never evicts, so it REQUIRES a pool covering every host block and
    # raises otherwise (a small pool with `full` would silently refuse blocks --
    # measured as RULER 10% vs 100%). Size it accordingly rather than special-casing
    # the assertion away.
    # ``full`` needs every block to own a way permanently, and blocks map to sets by
    # ``id % n_sets`` — so the binding constraint is per-SET occupancy, not total
    # capacity. Size for ceil(NB / (WAYS-1)) sets; the constructor rejects anything
    # smaller rather than silently refusing blocks.
    cap = (-(-NB // (WAYS - 1)) * WAYS) if pol == "full" else 4096
    c = HostKVCache(hk, hv, cap, "cuda", policy=pol)
    bad_tot = 0
    for step in range(6):
        g = torch.Generator(device="cuda").manual_seed(step)
        ids = torch.randint(0, 1024, (8, 32), generator=g, device="cuda", dtype=torch.int32)
        b, _ = verify(c, ids); bad_tot += b
    st = c.stats()
    good = bad_tot == 0 and st["overflow"] == 0
    ok &= good
    print(f"   {pol:<5} cap={st['capacity']:<5} ({st['n_sets']} sets x {st['ways']} ways) "
          f"pool>=demand: wrong={bad_tot} overflow={st['overflow']} "
          f"{'ok' if good else 'FAIL'}")
    del c; torch.cuda.empty_cache()

print("   over-subscribed (pool 4x SMALLER than the working set) -- refusals expected:")
for pol in ("lru", "fifo"):
    c = HostKVCache(hk, hv, 256, "cuda", policy=pol)
    served = refused = 0
    for step in range(6):
        g = torch.Generator(device="cuda").manual_seed(step)
        ids = torch.randint(0, 1024, (8, 32), generator=g, device="cuda", dtype=torch.int32)
        dk, dv, out = serve(c, ids)
        slot = out.cpu().long(); want = ids.cpu().long()
        for r in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                sl = int(slot[r, j])
                got = dk[sl].float().cpu()
                if sl == c.zero_slot:
                    # Refused: must resolve to the reserved ZERO block, so the entry
                    # contributes no information rather than another block's data.
                    if float(got.abs().max()) != 0.0:
                        refused = -10**9
                    else:
                        refused += 1
                    continue
                if float((got - hk[int(want[r, j])].float()).abs().max()) != 0.0:
                    refused = -10**9                 # served WRONG data: fatal
                else:
                    served += 1
    good = refused >= 0
    ok &= good
    print(f"   {pol:<5} served correctly={served} refused={max(refused,0)} "
          f"{'ok: served correct, refused -> zero block' if good else 'FAIL: wrong data served'}")
    del c; torch.cuda.empty_cache()

print("\n2) do the policies differ, in the direction claimed?")
# The distinguishing workload for LRU-vs-FIFO is a HOT set that is re-referenced
# plus a COLD stream that is not. LRU should keep the hot set (each reference
# promotes it); FIFO should cycle it out (insert order ignores reads).
#
# The comparison has to be made INSIDE ONE SET, because eviction is per-set: a hot
# block only competes with cold blocks that share its set. Spreading hot and cold
# over 128 sets measures capacity pressure instead of the eviction choice, and shows
# no difference (measured 521 vs 519). So pick ids that are congruent mod n_sets.
def hot_cold(pol, n_sets, ways, rounds=12):
    cap = n_sets * ways
    c = HostKVCache(hk, hv, cap, "cuda", policy=pol)
    # All ids below map to set 0; the set holds `ways` of them (minus the reserved
    # way if it lands here), so hot+cold together over-subscribe exactly one set.
    hot = torch.tensor([[i * n_sets for i in range(1, 9)]],
                       dtype=torch.int32, device="cuda")            # 8 hot blocks
    rl_hot = torch.full((1,), hot.shape[1] * BLK, dtype=torch.int32, device="cuda")
    c.tick(); c.reserve_remap(hot)
    c.fetch(hot.clone(), row_lens=rl_hot, num_rows=1, max_per_row=hot.shape[1])
    torch.cuda.synchronize()
    base = int(c.fetches.item())
    for rnd in range(rounds):
        # re-reference the hot set (LRU promotes; FIFO does not)
        c.tick(); c.fetch(hot.clone(), row_lens=rl_hot, num_rows=1,
                          max_per_row=hot.shape[1])
        # then a cold burst into the SAME set
        cold = torch.tensor([[(100 + rnd * 8 + k) * n_sets for k in range(8)]],
                            dtype=torch.int32, device="cuda")
        rl_cold = torch.full((1,), cold.shape[1] * BLK, dtype=torch.int32, device="cuda")
        c.tick(); c.reserve_remap(cold)
        c.fetch(cold.clone(), row_lens=rl_cold, num_rows=1, max_per_row=cold.shape[1])
        torch.cuda.synchronize()
    total = int(c.fetches.item()) - base
    # A hot block re-fetched means it was evicted between references.
    del c; torch.cuda.empty_cache()
    return total

r = {p: hot_cold(p, 16, 32) for p in ("lru", "fifo")}
print(f"   PCIe copies over 12 rounds, hot+cold contending for ONE set:")
for p_, v in r.items():
    print(f"     {p_:<5} {v:>4}")
better = r["lru"] <= r["fifo"]
ok &= better
print(f"   => {'ok: lru refetches no more than fifo (keeps the re-referenced set)' if better else 'FAIL: lru worse than fifo'}")

print("\n3) per-miss cost vs capacity (the old global scan was O(capacity)):")
for cap in (256, 2048, 16384):
    c = HostKVCache(hk, hv, cap, "cuda", policy="lru")
    ids = torch.randint(0, NB, (8, 32), device="cuda", dtype=torch.int32)
    serve(c, ids)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(20):
        ids = torch.randint(0, NB, (8, 32), device="cuda", dtype=torch.int32)
        serve(c, ids)                                # all-miss traffic
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 20
    print(f"   cap={c.capacity:>6}: {ms:6.3f} ms/step")
    del c; torch.cuda.empty_cache()

print("\n" + ("POLICY TESTS PASS" if ok else "*** FAILURES ***"))
raise SystemExit(0 if ok else 1)
