"""INT4 staging arena: exactly-once quantization, no starvation, cudagraph-safe.

What each section is guarding, and what a failure means:

1. **A staged block quantizes exactly once, on completion, from never-quantized values.** If it
   fired early the scale would be a partial reduction; if it fired twice a slot would be handed to
   two blocks. ``migrated`` is therefore an exact count, not a bound.

2. **Slots are recycled.** More blocks than slots must still complete, and every one must land. If
   release is broken the arena fills and later tokens are silently DROPPED -- the failure that
   produced 0% accuracy twice, with every counter reading 0, because starvation is a lost claim,
   not a refusal.

3. **The two paths agree bit-for-bit.** Decode fuses migration into the stage kernel; prefill
   cannot (it writes many tokens of one block per launch) so it migrates in a second kernel. Both
   must produce identical packed bytes -- otherwise accuracy depends on the batch composition.

4. **The result matches the reference quantizer.** The fused pack uses ``tl.split`` and computes
   the scale inside the kernel; it must agree with ``quantize_block``, which computes the scale in
   torch. Disagreement here is a channel-order or axis bug, and the read side would happily
   dequantize the wrong thing.

5. **Footprint is constant in context length.** The whole point of staging over a per-block bf16
   mirror -- that mirror measured a 1.28x REGRESSION against plain bf16.

6. **The full cycle captures and replays under cudagraph.** Decode is captured, so a claim loop
   that needs a host sync, or any lazy allocation, is unusable no matter how correct it is.

Run: python examples/misc/test_int4_arena.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_torch.cache.triton_kernels.int4_kv import quantize_block  # noqa: E402
from vortex_torch.engine.sgl.int4_arena import (  # noqa: E402
    CAPTURE_CEILING, Int4Arena, arena_slots,
)
from vortex_torch.engine.sgl.int4_store import K_SCALE, V_SCALE, int4_cache_meta  # noqa: E402

fails = 0
BS, D, PAGE, NKV = 32, 128, 32, 2
NB = 256


def make(num_blocks=NB, n_slots=8):
    meta = int4_cache_meta(BS, D)
    cache = {n: torch.zeros((num_blocks, s[0], s[1]), dtype=dt, device="cuda")
             for n, (s, dt) in meta.items()}
    arena = Int4Arena(num_blocks, n_slots, BS, D)
    stage_k = torch.zeros((n_slots, BS, D), dtype=torch.bfloat16, device="cuda")
    stage_v = torch.zeros_like(stage_k)
    return cache, arena, stage_k, stage_v


def step(arena, cache, sk, sv, positions, kv, fused):
    """One set_kv_buffer-shaped call: positions -> [n_tok] int64, kv -> [n_tok, NKV, D]."""
    loc = torch.tensor(positions, dtype=torch.int64, device="cuda")
    arena.stage(sk, sv, kv[0], kv[1], loc, cache, PAGE,
                fused=fused, k_scale_name=K_SCALE, v_scale_name=V_SCALE)


def rand_kv(n):
    a = (torch.randn(n, NKV, D, device="cuda") * 0.5).to(torch.bfloat16)
    b = (torch.randn(n, NKV, D, device="cuda") * 0.5).to(torch.bfloat16)
    return a, b


print("1. one block completes exactly once, and only when its last token arrives")
cache, arena, sk, sv = make()
kv = rand_kv(BS)
for t in range(BS):                                  # one token per launch = decode
    step(arena, cache, sk, sv, [t], (kv[0][t:t + 1], kv[1][t:t + 1]), fused=True)
    c = arena.counters()
    expect_mig = NKV if t == BS - 1 else 0
    if c["migrated"] != expect_mig:
        print(f"   token {t}: migrated={c['migrated']} want {expect_mig} <-- FAIL")
        fails += 1
        break
torch.cuda.synchronize()
c = arena.counters()
ok = c["migrated"] == NKV and c["declined"] == 0 and arena.occupancy() == 0
print(f"   after {BS} tokens x {NKV} heads: {c} occupancy={arena.occupancy()} "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n2. the packed result matches the reference quantizer")
for head in range(NKV):
    blk = ((0 // PAGE) * (PAGE * NKV) + head * PAGE) // BS
    ref = kv[0][:, head, :]
    want_pk, want_ks = quantize_block(ref, per_channel=True)
    ok_k = torch.equal(cache["k"][blk], want_pk)
    # scale computed in-kernel vs in torch: fp32 reductions in a different order
    ok_ks = torch.allclose(cache[K_SCALE][blk], want_ks, rtol=1e-6, atol=1e-9)
    refv = kv[1][:, head, :]
    want_pv, want_vs = quantize_block(refv, per_channel=False)
    ok_v = torch.equal(cache["v"][blk], want_pv)
    ok_vs = torch.allclose(cache[V_SCALE][blk], want_vs, rtol=1e-6, atol=1e-9)
    ok = ok_k and ok_ks and ok_v and ok_vs
    print(f"   head {head} block {blk}: K bytes={ok_k} K scale={ok_ks} "
          f"V bytes={ok_v} V scale={ok_vs} {'ok' if ok else '<-- FAIL'}")
    if not ok:
        fails += 1

print("\n3. more blocks than slots: slots recycle, nothing is DROPPED")
n_slots = 4
cache, arena, sk, sv = make(n_slots=n_slots)
n_blocks_written = 6
tot = BS * n_blocks_written
kv = rand_kv(tot)
# Sequential fill: block b's tokens all arrive before block b+1's, so at most NKV slots are ever
# live and 6 blocks pass through 4 slots.
for t in range(tot):
    step(arena, cache, sk, sv, [t], (kv[0][t:t + 1], kv[1][t:t + 1]), fused=True)
torch.cuda.synchronize()
c = arena.counters()
want_mig = n_blocks_written * NKV
ok = c["migrated"] == want_mig and c["declined"] == 0
print(f"   {n_blocks_written} blocks x {NKV} heads through {n_slots} slots: {c} "
      f"(want migrated={want_mig}, declined=0) {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
    if c["declined"]:
        print("      DECLINED > 0: tokens were dropped -- release is broken, and this is silent")

print("\n3b. PREFILL WIDTH: many programs, few slots -- the claim must not thunder-herd")
# The regression that motivated arbitrating on the block rather than the pool. With optimistic
# claiming this printed declined=51 of 64 tokens against an arena that only needed 2 slots.
cache, arena, sk, sv = make(n_slots=8)
kv = rand_kv(BS)
step(arena, cache, sk, sv, list(range(BS)), kv, fused=False)
torch.cuda.synchronize()
c = arena.counters()
ok = c["declined"] == 0 and c["spin_exhausted"] == 0 and arena.occupancy() == NKV
print(f"   {BS} tokens x {NKV} heads = {BS * NKV} programs, 8 slots: {c} "
      f"occupancy={arena.occupancy()} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
    if c["declined"]:
        print(f"      DECLINED {c['declined']}: transient demand exceeded the pool -- tokens lost")
    if c["spin_exhausted"]:
        print(f"      SPIN EXHAUSTED {c['spin_exhausted']}: a claim never published")

print("\n4. interleaved heads and multi-token launches (prefill) match the decode path")
# Same data, two ways: token-at-a-time fused, vs all-at-once staged + scanned.
cache_a, ar_a, ska, sva = make()
cache_b, ar_b, skb, svb = make()
kv = rand_kv(BS)
for t in range(BS):
    step(ar_a, cache_a, ska, sva, [t], (kv[0][t:t + 1], kv[1][t:t + 1]), fused=True)
step(ar_b, cache_b, skb, svb, list(range(BS)), kv, fused=False)
ar_b.migrate_complete(skb, svb, cache_b, k_scale_name=K_SCALE, v_scale_name=V_SCALE)
torch.cuda.synchronize()
same = all(torch.equal(cache_a[n], cache_b[n]) for n in ("k", "v", K_SCALE, V_SCALE))
ca, cb = ar_a.counters(), ar_b.counters()
ok = same and ca["migrated"] == cb["migrated"] == NKV
print(f"   decode-fused vs prefill-scan identical: {same}  "
      f"migrated {ca['migrated']} / {cb['migrated']} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n5. an INCOMPLETE block is left alone (no partial scale is ever written)")
cache, arena, sk, sv = make()
kv = rand_kv(BS - 1)
step(arena, cache, sk, sv, list(range(BS - 1)), kv, fused=False)
arena.migrate_complete(sk, sv, cache, k_scale_name=K_SCALE, v_scale_name=V_SCALE)
torch.cuda.synchronize()
c = arena.counters()
untouched = bool((cache[K_SCALE] == 0).all()) and bool((cache["k"] == 0).all())
ok = c["migrated"] == 0 and untouched and arena.occupancy() == NKV
print(f"   {BS - 1}/{BS} tokens: migrated={c['migrated']} page cache untouched={untouched} "
      f"still resident={arena.occupancy()} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n6. footprint is constant in context length")
small = Int4Arena(4, arena_slots(NKV), BS, D)
large = Int4Arena(4096, arena_slots(NKV), BS, D)
stage_bytes = arena_slots(NKV) * BS * D * 2 * 2
ok = small.nbytes() - 4 * 4 == large.nbytes() - 4096 * 4   # only slot_of scales, and it is int32
print(f"   slots={arena_slots(NKV)} ({CAPTURE_CEILING} rows x {NKV} heads), "
      f"staging={stage_bytes / 2**20:.1f} MiB/layer regardless of context; "
      f"maps {small.nbytes()} B @4 blocks vs {large.nbytes()} B @4096 "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n7. the full claim/stage/migrate cycle CAPTURES and REPLAYS under cudagraph")
cache, arena, sk, sv = make()
loc = torch.zeros(1, dtype=torch.int64, device="cuda")
nk = torch.zeros(1, NKV, D, dtype=torch.bfloat16, device="cuda")
nv = torch.zeros_like(nk)


def captured():
    arena.stage(sk, sv, nk, nv, loc, cache, PAGE,
                fused=True, k_scale_name=K_SCALE, v_scale_name=V_SCALE)


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        captured()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        captured()
    # Replay the graph once per token of a fresh block, feeding new inputs through the same
    # tensors. If capture had baked in a host-side decision this diverges from the eager result.
    arena.slot_of.fill_(-1); arena.owner_of.fill_(-1); arena.mask_of.zero_()
    arena.stats.zero_()
    for n in ("k", "v", K_SCALE, V_SCALE):
        cache[n].zero_()
    kv = rand_kv(BS)
    for t in range(BS):
        loc.fill_(t)
        nk.copy_(kv[0][t:t + 1]); nv.copy_(kv[1][t:t + 1])
        g.replay()
    torch.cuda.synchronize()
    c = arena.counters()
    blk = (0 * NKV * PAGE) // BS
    want_pk, _ = quantize_block(kv[0][:, 0, :], per_channel=True)
    ok = c["migrated"] == NKV and c["declined"] == 0 and torch.equal(cache["k"][blk], want_pk)
    print(f"   captured and replayed {BS}x: {c} bytes match eager={torch.equal(cache['k'][blk], want_pk)} "
          f"{'ok' if ok else '<-- FAIL'}")
    if not ok:
        fails += 1
except Exception as e:
    print(f"   <-- FAIL: capture/replay raised {type(e).__name__}: {e}")
    fails += 1

print("\n8. geometry limits are rejected, not silently mis-tracked")
for bad, why in ((dict(block_size=128), "block_size > 64 cannot fit the completion bitmask"),
                 (dict(head_dim=127), "odd head_dim cannot pack two per byte")):
    kwargs = dict(num_blocks=16, n_slots=4, block_size=BS, head_dim=D)
    kwargs.update(bad)
    try:
        Int4Arena(**kwargs)
        print(f"   {why}: accepted <-- FAIL")
        fails += 1
    except ValueError:
        print(f"   {why}: rejected ok")

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("INT4 ARENA TESTS PASS")
