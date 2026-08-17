"""The set-associative cache and every policy must be cuda-graph safe.

Three specific hazards were introduced with the policy layer, and none of them are
covered by the accuracy sweeps (a captured graph that misbehaves usually still
produces *plausible* output):

1. **New device state** (``age``, ``ins``, ``ins_ctr``) is read AND written inside the
   captured region. It must be preallocated — a first-use allocation inside a graph is
   captured into it.
2. **The intra-set retry loop** has a data-dependent trip count. Triton must lower it
   to a bounded loop, not something that changes the launch shape per replay.
3. **Policy state must evolve across replays.** If ``ins_ctr`` were frozen at capture
   (like the earlier generation-counter bug), every replay would reuse one stamp and
   the recency order would stop working — silently.

Run: python examples/misc/test_host_kv_graph_policy.py
"""
import os, sys, torch
# The tree this file lives in — see the note in test_host_kv_policy.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vortex_torch.engine.sgl.host_kv import HostKVCache
from vortex_torch.engine.sgl.cache_policy import POLICIES, WAYS

BLK, D, NB = 32, 128, 4096
hk = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True); hk.normal_()
hv = torch.zeros(NB, BLK, D, dtype=torch.bfloat16, pin_memory=True); hv.normal_()

ok = True
for pol in POLICIES:
    cap = (-(-NB // (WAYS - 1)) * WAYS) if pol == "full" else 1024
    c = HostKVCache(hk, hv, cap, "cuda", policy=pol)
    rows, mpr = 8, 32
    tbl = torch.zeros(rows, mpr, dtype=torch.int32, device="cuda")
    rl = torch.full((rows,), mpr * BLK, dtype=torch.int32, device="cuda")
    c.reserve_remap(tbl)                       # preallocate OUTSIDE the graph

    # warm on a side stream, then capture
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            tbl.copy_(torch.randint(0, NB, (rows, mpr), dtype=torch.int32))
            c.tick(); c.fetch(tbl, row_lens=rl, num_rows=rows, max_per_row=mpr)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    tbl.copy_(torch.randint(0, NB, (rows, mpr), dtype=torch.int32))
    with torch.cuda.graph(g):
        c.tick()
        dk, dv, out = c.fetch(tbl, row_lens=rl, num_rows=rows, max_per_row=mpr)
    captured = True

    gens, stamps, bad = [], [], 0
    for rep in range(5):
        src = torch.randint(0, NB, (rows, mpr), dtype=torch.int32)
        tbl.copy_(src.cuda()); g.replay(); torch.cuda.synchronize()
        gens.append(int(c.gen.item())); stamps.append(int(c.ins_ctr.item()))
        slot = out.cpu().long()
        for r in range(rows):
            for j in range(mpr):
                sl = int(slot[r, j])
                if sl == c.zero_slot:
                    continue                    # counted refusal, contributes zeros
                if sl < 0 or sl >= c.capacity:
                    bad += 1; continue
                if float((dk[sl].float().cpu()
                          - hk[int(src[r, j])].float()).abs().max()) != 0.0:
                    bad += 1
    gen_moves = gens == sorted(gens) and gens[-1] > gens[0]
    stamp_moves = stamps[-1] > stamps[0]
    good = captured and bad == 0 and gen_moves and stamp_moves
    ok &= good
    print(f"  {pol:<5} captured={captured} data_errors={bad} "
          f"gen {gens[0]}->{gens[-1]} stamp {stamps[0]}->{stamps[-1]} "
          f"{'ok' if good else 'FAIL'}")
    del c, g; torch.cuda.empty_cache()

print("\n" + ("CUDAGRAPH x POLICY TESTS PASS" if ok else "*** FAILURES ***"))
raise SystemExit(0 if ok else 1)
