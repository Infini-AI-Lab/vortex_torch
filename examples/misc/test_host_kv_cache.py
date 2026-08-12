"""Correctness of the persistent host-KV block cache under contention.

``HostKVCache`` keeps KV in pinned host memory and caches selected blocks in a
small GPU pool, fetched by a Triton kernel. Its whole risk surface is
concurrency: one program per block-table entry, so the same block is requested
by many programs at once while other programs may be evicting it.

The invariant that must hold: after ``fetch()``, every table entry points at a
staging slot whose contents equal the host block it originally named. This
checks that under

1. **eviction pressure** — a working set 8x the pool, so almost everything is a
   miss and the victim scan runs hot;
2. **reuse** — an identical selection replayed must cross PCIe *zero* times.
   Asserted on the fetch counter, not on residency: residency can look stable
   while blocks are silently re-copied, which is how the original
   pin-before-dedup ordering bug hid (it re-fetched resident blocks because all
   but one requester lost the pin and declared a false miss);
3. **partial overlap** — only genuinely new blocks may be copied;
4. **dedup** — 512 entries naming 4 distinct blocks must copy 4 blocks;
5. **ragged rows** — entries past a row's length hold last step's ids and must
   be ignored, including a zero-length row;
6. **cudagraph** — captured once, replayed with different selections, which is
   the real serving path and the reason the generation counter lives on device.

Run: python examples/misc/test_host_kv_cache.py
"""
import torch
from vortex_torch.engine.sgl.host_kv import HostKVCache

BLK, D, NB = 32, 128, 4096
hk = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True); hk.normal_()
hv = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True); hv.normal_()

def check(c, tbl, rowlens, nrows, mpr, label):
    """``rowlens`` is in TOKENS (trtllm's sparse_seqlens); blocks = ceil(tok/BLK)."""
    orig = tbl.clone()
    dk, dv, out = c.fetch(tbl, row_lens=rowlens, num_rows=nrows, max_per_row=mpr)
    torch.cuda.synchronize()
    ov = int(c.overflow.item())
    bad = 0
    for r in range(nrows):
        n = -(-int(rowlens[r]) // BLK)
        if n == 0: continue
        want_id = orig[r, :n].cpu().long()
        slot = out[r, :n].cpu().long()
        if (slot < 0).any() or (slot >= c.capacity).any():
            bad += 1; continue
        ek = (dk[slot].float().cpu() - hk[want_id].float()).abs().max()
        ev = (dv[slot].float().cpu() - hv[want_id].float()).abs().max()
        if float(ek) != 0.0 or float(ev) != 0.0: bad += 1
    # The caller's table must be untouched: it is shared across layers, and
    # remapping it in place corrupted the per-step BOS/EOS slots (RULER 0/20).
    if not bool((tbl == orig).all()):
        print(f"  {label:<34} FAIL: input table was mutated")
        return False
    print(f"  {label:<34} rows={nrows:<4} overflow={ov:<3} "
          f"{'ok' if bad==0 and ov==0 else f'FAIL({bad} rows)'}")
    return bad == 0 and ov == 0

# ---- 1. eviction pressure: pool far smaller than the working set ----------
print("eviction pressure (pool 256 blocks, working set up to 2000):")
allok = True
c = HostKVCache(hk, hv, capacity=256, device="cuda")
for step, nd in enumerate([50, 2000, 8, 900, 120, 2000, 3], start=1):
    nrows, mpr = 8, 16
    ids = torch.randint(0, nd, (nrows, mpr), dtype=torch.int32)
    rl = torch.full((nrows,), mpr * BLK, dtype=torch.int32, device="cuda")
    c.tick()
    allok &= check(c, ids.cuda(), rl, nrows, mpr, f"step{step} distinct<={nd}")

# ---- 2. reuse: same selection twice must not refetch ----------------------
print("\nreuse (identical selection repeated -> should be all hits):")
c2 = HostKVCache(hk, hv, capacity=512, device="cuda")
nrows, mpr = 8, 16
ids = torch.randint(0, 100, (nrows, mpr), dtype=torch.int32).cuda()
rl = torch.full((nrows,), mpr * BLK, dtype=torch.int32, device="cuda")
c2.tick(); check(c2, ids.clone(), rl, nrows, mpr, "first pass (cold)")
nd_ = int(torch.unique(ids).numel())
f1 = c2.stats()["fetches"]
c2.tick(); allok &= check(c2, ids.clone(), rl, nrows, mpr, "second pass (warm)")
f2 = c2.stats()["fetches"]
print(f"    distinct={nd_}  PCIe copies: cold={f1} warm={f2-f1}  "
      f"(entries/pass={nrows*mpr})")
reuse_ok = (f1 == nd_) and (f2 - f1 == 0)
allok &= reuse_ok
print(f"    {'ok: cold fetched each distinct once, warm fetched NOTHING' if reuse_ok else 'FAIL: warm pass re-fetched'}")
# partial overlap: only the new blocks should cross the bus
ids2 = ids.clone(); ids2[:, :4] = torch.randint(2000, 3000, (nrows, 4), dtype=torch.int32).cuda()
c2.tick(); allok &= check(c2, ids2.clone(), rl, nrows, mpr, "third pass (50% new)")
f3 = c2.stats()["fetches"]
new_ = int(torch.unique(ids2).numel()) - len(set(torch.unique(ids).tolist()) & set(torch.unique(ids2).tolist()))
print(f"    new distinct blocks={new_}  PCIe copies={f3-f2}  "
      f"{'ok: only misses copied' if f3-f2 == new_ else 'FAIL'}")
allok &= (f3 - f2 == new_)

# ---- 3. duplicate ids within a row (dedup path) ---------------------------
print("\ndedup (every row asks for the same 4 blocks):")
c3 = HostKVCache(hk, hv, capacity=256, device="cuda")
nrows, mpr = 32, 16
ids = torch.tensor([[7,7,7,7,11,11,11,11,3,3,3,3,255,255,255,255]]*nrows, dtype=torch.int32)
rl = torch.full((nrows,), mpr * BLK, dtype=torch.int32, device="cuda")
c3.tick(); allok &= check(c3, ids.cuda(), rl, nrows, mpr, "512 entries, 4 distinct")
print(f"    resident={c3.stats()['resident']} (expect 4)")
allok &= c3.stats()["resident"] == 4

# ---- 4. ragged rows: entries past row_len must be ignored ----------------
print("\nragged rows (short rows leave stale tail entries):")
c4 = HostKVCache(hk, hv, capacity=256, device="cuda")
nrows, mpr = 8, 16
ids = torch.randint(0, 300, (nrows, mpr), dtype=torch.int32)
rl = torch.tensor([1,2,3,16,0,5,9,16], dtype=torch.int32, device="cuda") * BLK
c4.tick(); allok &= check(c4, ids.cuda(), rl, nrows, mpr, "row_lens 1..16 incl. 0")

# ---- 5. cudagraph ---------------------------------------------------------
print("\ncudagraph capture + replay with different selections:")
c5 = HostKVCache(hk, hv, capacity=1024, device="cuda")
nrows, mpr = 8, 16
tbl = torch.zeros(nrows, mpr, dtype=torch.int32, device="cuda")
rl = torch.full((nrows,), mpr * BLK, dtype=torch.int32, device="cuda")
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        tbl.copy_(torch.randint(0,500,(nrows,mpr),dtype=torch.int32))
        c5.tick(); c5.fetch(tbl, row_lens=rl, num_rows=nrows, max_per_row=mpr)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
tbl.copy_(torch.randint(0,500,(nrows,mpr),dtype=torch.int32))
with torch.cuda.graph(g):
    c5.tick()
    _dk5, _dv5, out5 = c5.fetch(tbl, row_lens=rl, num_rows=nrows, max_per_row=mpr)
print("  captured OK")
for rep, nd in enumerate([40, 900, 5, 2000], start=1):
    src = torch.randint(0, nd, (nrows, mpr), dtype=torch.int32)
    tbl.copy_(src.cuda()); g.replay(); torch.cuda.synchronize()
    bad = 0
    for r in range(nrows):
        slot = out5[r].cpu().long()
        if (slot<0).any() or (slot>=c5.capacity).any(): bad+=1; continue
        e = (c5.dev_k[slot].float().cpu() - hk[src[r].long()].float()).abs().max()
        if float(e)!=0.0: bad+=1
    ok = bad==0 and int(c5.overflow.item())==0
    allok &= ok
    print(f"  replay {rep}: gen={c5.stats()['generation']:<3} distinct<={nd:<5} "
          f"{'ok' if ok else f'FAIL({bad})'}")


# ---- 6. CSR mode (flashinfer names selections as an indptr + flat indices) --
print("\nCSR mode (flashinfer indptr):")
c6 = HostKVCache(hk, hv, capacity=512, device="cuda")
nrows = 8
lens = [12, 3, 16, 0, 7, 16, 1, 9]
indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
flat = torch.randint(0, 400, (int(indptr[-1]),), dtype=torch.int32, device="cuda")
orig = flat.clone()
c6.tick()
dk, dv, out6 = c6.fetch(flat, indptr=indptr, num_rows=nrows, max_per_row=max(lens))
torch.cuda.synchronize()
bad = 0
assert bool((flat == orig).all()), "CSR: input indices were mutated"
for r in range(nrows):
    lo, hi = int(indptr[r]), int(indptr[r+1])
    if hi == lo: continue
    slot = out6[lo:hi].cpu().long()
    if (slot < 0).any() or (slot >= c6.capacity).any(): bad += 1; continue
    e = (dk[slot].float().cpu() - hk[orig[lo:hi].cpu().long()].float()).abs().max()
    if float(e) != 0.0: bad += 1
ok = bad == 0 and int(c6.overflow.item()) == 0
allok &= ok
print(f"  ragged CSR rows {lens}  {'ok' if ok else f'FAIL({bad})'}")

# ---- 7. invalidate_locs: a rewritten token must not be served from cache ----
# This is the decode-step hazard: set_kv_buffer writes the new token into host
# memory, but the block covering it may already be resident with pre-write
# contents -- and the local window selects exactly that block every step.
print("\ninvalidate_locs (new token written into an already-resident block):")
c7 = HostKVCache(hk, hv, capacity=256, device="cuda")
PAGE, NKVH = BLK, 4
nrows, mpr = 4, 8
ids = torch.randint(0, 200, (nrows, mpr), dtype=torch.int32).cuda()
rl = torch.full((nrows,), mpr * BLK, dtype=torch.int32, device="cuda")
c7.tick(); c7.fetch(ids.clone(), row_lens=rl, num_rows=nrows, max_per_row=mpr)
torch.cuda.synchronize()
# pick a token position whose block is resident, mutate host KV, invalidate
resident = (c7.owner_of >= 0).nonzero().flatten()
blk_id = int(c7.owner_of[resident[0]].item())
pos = blk_id * BLK          # head 0, so position_trans == blk*BLK maps back directly
hk[blk_id].normal_()        # simulate set_kv_buffer writing new K
loc = torch.tensor([pos], dtype=torch.int64, device="cuda")
before = int(c7.slot_of[blk_id].item())
c7.invalidate_locs(loc, PAGE, 1)
torch.cuda.synchronize()
after = int(c7.slot_of[blk_id].item())
inv_ok = before >= 0 and after == -1
allok &= inv_ok
print(f"  block {blk_id}: slot {before} -> {after}  "
      f"{'ok: evicted so the next fetch re-reads host' if inv_ok else 'FAIL'}")
# and the refetch must now see the NEW contents
tbl2 = torch.full((1, 1), blk_id, dtype=torch.int32, device="cuda")
rl2 = torch.tensor([BLK], dtype=torch.int32, device="cuda")
c7.tick()
dk2, _dv2, out7 = c7.fetch(tbl2, row_lens=rl2, num_rows=1, max_per_row=1)
torch.cuda.synchronize()
e = float((dk2[int(out7[0,0])].float().cpu() - hk[blk_id].float()).abs().max())
allok &= e == 0.0
print(f"  refetch err={e:.1e} {'ok: serves the new K' if e==0.0 else 'FAIL: served stale K'}")


# ---- 8. invalidate_locs with the MHA head interleave (nkvh > 1) --------------
# The MHA pool stores KV block-interleaved by head, so ONE token touches one
# block per KV head at scattered ids. MLA has a single head and reduces to
# pos//block. Both mappings must match set_kv_buffer_kernel's arithmetic exactly,
# or a write lands in a block the pool still serves from a stale slot.
print("\ninvalidate_locs with head interleave (nkvh=4):")
NKVH, PAGE = 4, BLK
c8 = HostKVCache(hk, hv, capacity=512, device="cuda")
pos = 1000
want = set()
for h in range(NKVH):
    trans = (pos // PAGE) * (PAGE * NKVH) + h * PAGE + pos % PAGE
    want.add(trans // BLK)
# make all of them resident
ids = torch.tensor([sorted(want)], dtype=torch.int32).cuda()
rl = torch.tensor([len(want) * BLK], dtype=torch.int32, device="cuda")
c8.tick(); c8.fetch(ids.clone(), row_lens=rl, num_rows=1, max_per_row=len(want))
torch.cuda.synchronize()
before = {b: int(c8.slot_of[b].item()) for b in want}
c8.invalidate_locs(torch.tensor([pos], dtype=torch.int64, device="cuda"), PAGE, NKVH)
torch.cuda.synchronize()
after = {b: int(c8.slot_of[b].item()) for b in want}
ok = all(before[b] >= 0 for b in want) and all(after[b] == -1 for b in want)
allok &= ok
print(f"  token {pos} -> blocks {sorted(want)}")
print(f"  slots before {[before[b] for b in sorted(want)]} after {[after[b] for b in sorted(want)]} "
      f"{'ok: every head-block evicted' if ok else 'FAIL'}")
# and nothing else was evicted
other = [b for b in range(200) if b not in want]
ids2 = torch.tensor([other[:8]], dtype=torch.int32).cuda()
rl2 = torch.tensor([8 * BLK], dtype=torch.int32, device="cuda")
c8.tick(); c8.fetch(ids2.clone(), row_lens=rl2, num_rows=1, max_per_row=8)
torch.cuda.synchronize()
c8.invalidate_locs(torch.tensor([pos], dtype=torch.int64, device="cuda"), PAGE, NKVH)
torch.cuda.synchronize()
kept = sum(1 for b in other[:8] if int(c8.slot_of[b].item()) >= 0)
allok &= kept == 8
print(f"  unrelated blocks still resident: {kept}/8 "
      f"{'ok: eviction is targeted' if kept == 8 else 'FAIL: over-evicted'}")

print("\n" + ("ALL HOST-KV CACHE TESTS PASS" if allok else "*** FAILURES ***"))
raise SystemExit(0 if allok else 1)
