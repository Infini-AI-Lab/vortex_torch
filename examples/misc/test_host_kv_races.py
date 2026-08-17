"""Host-KV claim protocol: the two race bugs, as regression tests.

Both bugs this file guards were **silent** — no exception, no assert, just attention reading
another block's KV — and both were invisible to the existing tests, which check aggregate
counters or a single end-to-end match. What made them findable was per-element provenance:
fill host block ``b`` with the constant value ``b``, prefill the destination with a sentinel,
then histogram each slot. Every element then names the program that wrote it, so a slot with
two writers is unambiguous and the element POSITIONS reveal the granularity of the tear.

R1  **Scalar atomics are per-warp.** Triton single-lanes a scalar atomic per *warp*, not per
    CTA, so at ``num_warps>1`` every warp ran the claim loop independently: one won
    ``atomic_xchg(pin_gen + cand, gen)``, the others read back ``== gen``, concluded they had
    lost, and claimed *different* ways. ``slot`` then differed between warps of one program
    and the copy loop tore a block's payload across several ways, while every ``owner_of``
    still claimed a whole block. Measured before the fix: 0/60 corrupt trials at 1 warp, 8/60
    at 4, 57/60 at 8. The fix is ``num_warps=1`` at the launch site; this test asserts the
    shipped path is clean under heavy set pressure, and (as a canary) that the diagnostic
    still detects tearing when several warps are forced.

R2  **tick_all keyed its pointer cache on the list LENGTH.** Passing a same-length list of
    different cache objects reused the stale device-pointer array, so it ticked the OLD
    caches' generations and left the new ones at 0 forever. A cache whose ``gen`` never
    advances sees every pin as still held from a previous step, so ``victim_way`` refuses
    every way and the pool degenerates into permanent refusals. Now keyed on identity, with a
    strong reference so ``id()`` cannot be recycled behind the key.

Run: python examples/misc/test_host_kv_races.py
"""
import collections
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import vortex_torch.engine.sgl.host_kv as HK  # noqa: E402
from vortex_torch.engine.sgl.cache_policy import WAYS  # noqa: E402
from vortex_torch.engine.sgl.host_kv import HostKVCache, tick_all  # noqa: E402

BLK, D, NB = 32, 128, 256
SENT = -1.0
fails = 0

# Host block b is filled with the constant b, so every element of dev_k names its writer.
hk = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True)
hv = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True)
for _b in range(NB):
    hk[_b] = float(_b)
    hv[_b] = float(_b)


def audit(c, dk):
    """(bad, torn) over resident slots: bad = payload != owner, torn = two writers."""
    bad = torn = 0
    owner = c.owner_of.cpu()
    for s in (owner >= 0).nonzero().flatten().tolist():
        b = int(owner[s])
        vals = collections.Counter(dk[s].reshape(-1).float().cpu().tolist())
        if len(vals) == 1 and float(b) in vals:
            continue
        bad += 1
        if len([v for v in vals if v != SENT]) > 1:
            torn += 1
    return bad, torn


def run(policy, cap, rows, per, steps, num_warps=None):
    """Serve `steps` random selections; return (bad, torn). num_warps=None => shipped path."""
    c = HostKVCache(hk, hv, cap, "cuda", policy=policy)
    c.dev_k.fill_(SENT)
    c.dev_v.fill_(SENT)
    torch.cuda.synchronize()
    bad = torn = 0
    for step in range(steps):
        g = torch.Generator(device="cuda").manual_seed(step)
        ids = torch.randint(0, NB, (rows, per), generator=g, device="cuda", dtype=torch.int32)
        rl = torch.full((rows,), per * BLK, dtype=torch.int32, device="cuda")
        c.tick()
        c.reserve_remap(ids)
        if num_warps is None:
            dk, _dv, _out = c.fetch(ids.clone(), row_lens=rl, num_rows=rows, max_per_row=per)
        else:
            tbl = ids.clone()
            out = c._remap_table(tbl)
            HK._fetch_kernel[(per, rows)](
                c.host_k, c.host_v, c.dev_k, c.dev_v, tbl, rl, None,
                c.slot_of, c.owner_of, c.pin_gen, c.claim_gen,
                c.age, c.ins, c.ins_ctr, c.gen, c.overflow, c.fetches, c.requests,
                c.n_sets, tbl.stride(0), per, c.num_blocks,
                BLOCK_NUMEL=c.block_numel, BLOCK_TOKENS=c.block_tokens,
                TILE=HK._COPY_TILE, IS_CSR=False,
                POLICY=c.policy_code, WAYS_C=WAYS, N_SHARD=HK._REQ_SHARDS,
                num_warps=num_warps)
            HK._remap_kernel[(per, rows)](
                tbl, out, rl, None, c.slot_of, tbl.stride(0), per,
                c.num_blocks, c.zero_slot, BLOCK_TOKENS=c.block_tokens, IS_CSR=False)
            dk = c.dev_k
        torch.cuda.synchronize()
        b, t = audit(c, dk)
        bad += b
        torn += t
    del c
    torch.cuda.empty_cache()
    return bad, torn


print("R1  scalar-atomic claim loop: payload provenance under set pressure")
print(f"    {'policy':>10s} {'cap':>6s} {'shape':>10s} {'bad':>5s} {'torn':>5s}")
for pol in ("lru", "block_lru", "fifo", "none"):
    # cap well below demand => every set evicts every step, i.e. maximum contention
    for cap, rows, per in ((2048, 48, 31), (128, 48, 31)):
        bad, torn = run(pol, cap, rows, per, steps=10)
        fails += bad
        print(f"    {pol:>10s} {cap:>6d} {rows:>4d} x {per:<3d} {bad:>5d} {torn:>5d}"
              f"{'' if bad == 0 else '   <-- FAIL'}")

# Canary: the diagnostic must still be able to SEE tearing, otherwise a future change that
# reintroduces multiple warps would pass this file by accident.
bad_nw, torn_nw = run("lru", 128, 48, 31, steps=10, num_warps=4)
print(f"    canary, forced num_warps=4: bad={bad_nw} torn={torn_nw} "
      f"({'ok: detector works' if bad_nw > 0 else 'FAIL: detector is blind'})")
if bad_nw == 0:
    fails += 1

print("\nR2  tick_all pointer cache must key on identity, not length")


def mk(n):
    out = []
    for _ in range(n):
        a = torch.zeros(64, BLK, D, dtype=torch.bfloat16, pin_memory=True)
        b = torch.zeros(64, BLK, D, dtype=torch.bfloat16, pin_memory=True)
        out.append(HostKVCache(a, b, 256, "cuda", policy="lru"))
    return out


def gens(cs):
    return [int(c.gen.item()) for c in cs]


# same length, different objects, holder reused: the length key's blind spot
a = mk(4)
tick_all(a)
b = a[:1] + mk(3)
tick_all(b)
ok = gens(b[1:]) == [1, 1, 1] and gens(a[1:]) == [1, 1, 1]
print(f"    same-length swap: new={gens(b[1:])} old={gens(a[1:])} (want [1,1,1] / [1,1,1])"
      f" {'ok' if ok else '<-- FAIL'}")
fails += 0 if ok else 1

# id reuse: drop the old caches so CPython can recycle their ids
h = mk(4)
tick_all(h)
holder = h[0]
del h
gc.collect()
new = [holder] + mk(3)
tick_all(new)
ok = gens(new) == [2, 1, 1, 1]
print(f"    after gc/id-reuse: {gens(new)} (want [2,1,1,1]) {'ok' if ok else '<-- FAIL'}")
fails += 0 if ok else 1

# the fused tick must agree with the per-layer loop it replaced (MLA still loops)
d, e = mk(5), mk(5)
for _ in range(7):
    tick_all(d)
for _ in range(7):
    for x in e:
        x.tick()
ok = gens(d) == gens(e) == [7] * 5
print(f"    tick_all vs loop: {gens(d)} vs {gens(e)} {'ok' if ok else '<-- FAIL'}")
fails += 0 if ok else 1

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("HOST-KV RACE REGRESSION TESTS PASS")
