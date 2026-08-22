"""INT4 store: selection gather, layout metadata, and honest compression accounting.

The store is the piece all three placements share, so a bug here is a bug everywhere.

1. **The selection gather equals per-block dequant.** ``gather_unpack`` pulls an arbitrary set of
   block ids out of the packed cache in one kernel; it must agree element-for-element with
   dequantizing those blocks individually. Disagreement means the block-stride arithmetic is
   wrong, which shows up as attention reading a neighbouring block -- plausible wrong data, no
   error. This also pins the channel ORDER: the gather uses ``tl.join`` for coalescing, which must
   land in natural order, and a permutation here scored 0% end-to-end.

2. **Order and repetition are respected.** The selection is whatever the indexer produced: not
   sorted, possibly with the same block twice (different rows selecting it). Output row i must
   correspond to ``table[i]``, always.

3. **In-flight blocks come from staging, not from the packed bytes.** A block still being written
   has no scale, so its packed bytes are meaningless. Omitting this overlay cost ALL the accuracy
   when first wired (0/20), because the un-migrated block is always the newest.

4. **Compression is reported including scale overhead.** The nominal 4x is diluted by the fp32
   scales; a caller sizing a pool needs the true number.

Run: python examples/misc/test_int4_store.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_torch.cache.triton_kernels.int4_kv import (  # noqa: E402
    dequantize_block, quantize_block,
)
from vortex_torch.engine.sgl.int4_store import (  # noqa: E402
    K_SCALE, STAGE_K, STAGE_V, V_SCALE, bytes_per_block, compression_ratio,
    gather_unpack, int4_cache_meta, int4_request_cache_meta,
    native_int4_attention_supported, quantize_into,
)

fails = 0
BT, D, NB = 32, 128, 64

print("0. environment")
cap = torch.cuda.get_device_capability()
native = native_int4_attention_supported()
print(f"   capability sm_{cap[0]}{cap[1]}, native INT4 attention = {native}"
      f" -> {'native NVFP4 path' if native else 'dequant-to-bf16 path (pre-Blackwell)'}")

print("\n1. cache_meta_info shapes and dtypes")
meta = int4_cache_meta(BT, D)
want = {
    "k": ((BT, D // 2), torch.uint8),
    "v": ((BT, D // 2), torch.uint8),
    K_SCALE: ((1, D), torch.float32),
    V_SCALE: ((BT, 1), torch.float32),
}
for key, exp in want.items():
    got = meta.get(key)
    ok = (got == exp)
    print(f"   {key:<16s} {got} {'ok' if ok else f'<-- FAIL, want {exp}'}")
    if not ok:
        fails += 1
req = int4_request_cache_meta(BT, D)
ok = req == {STAGE_K: (BT, D), STAGE_V: (BT, D)}
print(f"   staging (request domain) {req} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
try:
    int4_cache_meta(BT, 127)
    print("   odd head_dim accepted <-- FAIL (cannot pack two per byte)")
    fails += 1
except ValueError:
    print("   odd head_dim rejected: ok")

print("\n2. compression accounting includes the scales")
bpb = bytes_per_block(BT, D)
bf16 = 2 * BT * D * 2
ratio = compression_ratio(BT, D)
exp_bpb = 2 * BT * (D // 2) + 4 * D + 4 * BT
ok = (bpb == exp_bpb) and (3.0 < ratio < 4.0)
print(f"   bf16 {bf16} B/block -> int4 {bpb} B/block = {ratio:.2f}x "
      f"(between 3 and 4, NOT 4.00) {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n3. build a packed cache, then gather selections out of it")
torch.manual_seed(0)
ref_k = (torch.randn(NB, BT, D, device="cuda") * 0.5).to(torch.bfloat16)
ref_v = (torch.randn(NB, BT, D, device="cuda") * 0.5).to(torch.bfloat16)
cg = torch.ones(D, device="cuda"); cg[::16] = 8.0
ref_k = (ref_k.float() * cg).to(torch.bfloat16)
tg = torch.ones(BT, 1, device="cuda"); tg[::8] = 8.0
ref_v = (ref_v.float() * tg).to(torch.bfloat16)

pk = torch.zeros(NB, BT, D // 2, dtype=torch.uint8, device="cuda")
pv = torch.zeros(NB, BT, D // 2, dtype=torch.uint8, device="cuda")
sk = torch.zeros(NB, 1, D, dtype=torch.float32, device="cuda")
sv = torch.zeros(NB, BT, 1, dtype=torch.float32, device="cuda")
for b in range(NB):
    quantize_into(pk[b], pv[b], sk[b], sv[b], ref_k[b], ref_v[b])

# no block in flight for this part
slot_of = torch.full((NB,), -1, dtype=torch.int32, device="cuda")
st_k = torch.zeros(8, BT, D, dtype=torch.bfloat16, device="cuda")
st_v = torch.zeros_like(st_k)

for name, sel in (("sorted", [0, 1, 2, 3]),
                  ("unsorted", [17, 3, 63, 0, 40]),
                  ("with repeats", [5, 5, 9, 5]),
                  ("single", [42]),
                  ("all", list(range(NB)))):
    ids = torch.tensor(sel, dtype=torch.int32, device="cuda")
    ot = torch.empty_like(ids)
    ok_ = torch.zeros(len(sel), BT, D, dtype=torch.bfloat16, device="cuda")
    ov_ = torch.zeros(len(sel), BT, D, dtype=torch.bfloat16, device="cuda")
    gather_unpack(pk, pv, sk, sv, ids, ot, st_k, st_v, slot_of, ok_, ov_)
    torch.cuda.synchronize()

    bad = 0
    for i, b in enumerate(sel):
        wk = dequantize_block(pk[b], sk[b], True, BT, D, out_dtype=torch.bfloat16)
        wv = dequantize_block(pv[b], sv[b], False, BT, D, out_dtype=torch.bfloat16)
        if not torch.equal(ok_[i], wk):
            bad += 1
        if not torch.equal(ov_[i], wv):
            bad += 1
    tbl_ok = torch.equal(ot, torch.arange(len(sel), dtype=torch.int32, device="cuda"))
    ok = (bad == 0) and tbl_ok
    print(f"   {name:<13s} n={len(sel):<3d} mismatched={bad} table=identity:{tbl_ok} "
          f"{'ok' if ok else '<-- FAIL'}")
    if not ok:
        fails += 1

print("\n4. an IN-FLIGHT block reads from staging, not from the packed bytes")
live_blk = 7
slot_of2 = torch.full((NB,), -1, dtype=torch.int32, device="cuda")
slot_of2[live_blk] = 3
exact = (torch.randn(BT, D, device="cuda") * 0.5).to(torch.bfloat16)
st_k[3] = exact
st_v[3] = exact
ids = torch.tensor([live_blk, 8], dtype=torch.int32, device="cuda")
ot = torch.empty_like(ids)
gk = torch.zeros(2, BT, D, dtype=torch.bfloat16, device="cuda")
gv = torch.zeros_like(gk)
gather_unpack(pk, pv, sk, sv, ids, ot, st_k, st_v, slot_of2, gk, gv)
torch.cuda.synchronize()
ok = torch.equal(gk[0], exact) and torch.equal(gv[0], exact)
print(f"   in-flight block returns the EXACT staged bf16: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
# and the neighbour, which is NOT in flight, still comes from the packed bytes
wk8 = dequantize_block(pk[8], sk[8], True, BT, D, out_dtype=torch.bfloat16)
ok = torch.equal(gk[1], wk8)
print(f"   a completed block still comes from packed bytes: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n5. round-trip error matches the offline prediction")
ids = torch.arange(NB, dtype=torch.int32, device="cuda")
ot = torch.empty_like(ids)
gk = torch.zeros(NB, BT, D, dtype=torch.bfloat16, device="cuda")
gv = torch.zeros(NB, BT, D, dtype=torch.bfloat16, device="cuda")
gather_unpack(pk, pv, sk, sv, ids, ot, st_k, st_v, slot_of, gk, gv)
torch.cuda.synchronize()


def rel(a, b):
    return (a.float() - b.float()).norm().item() / a.float().norm().item()


ek, ev = rel(ref_k, gk), rel(ref_v, gv)
ok = ek < 0.15 and ev < 0.20
print(f"   K err={ek:.4f} (screen: 0.067 on real K)   V err={ev:.4f} (screen: 0.145) "
      f"{'ok' if ok else '<-- FAIL: far worse than predicted'}")
if not ok:
    fails += 1

print("\n6. an empty selection is a no-op, not a crash")
try:
    gather_unpack(pk, pv, sk, sv, torch.zeros(0, dtype=torch.int32, device="cuda"),
                  torch.zeros(0, dtype=torch.int32, device="cuda"),
                  st_k, st_v, slot_of, gk, gv)
    torch.cuda.synchronize()
    print("   empty selection: ok")
except Exception as e:
    print(f"   empty selection: <-- FAIL ({type(e).__name__}: {e})")
    fails += 1

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("INT4 STORE TESTS PASS")
