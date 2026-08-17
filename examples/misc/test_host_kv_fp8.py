"""Does the set-associative cache work with an fp8 KV cache?

Every KV path in host_kv.py is a load/store copy with no arithmetic, so it should be
dtype-transparent. Two spots deserve a real check rather than an assumption:
  * ``dev_k[zero_slot].zero_()`` -- fp8 tensors reject some torch ops;
  * the MLA latent writer, which slices the payload into (kv_c, k_pe) halves.
Also verify bit-exactness: an fp8 copy must be exact, not merely close.

**Expected failure on pre-sm_89 GPUs.** ``fp8_e4m3`` (Triton's ``fp8e4nv``) has no hardware
support below sm_89, so on an A100 (sm_80) every e4m3 case dies at Triton compile time with
"type fp8e4nv not supported in this architecture". That is the GPU, not vortex: a bare
``tl.store`` of ``tl.float8e4nv`` in a kernel with no vortex code refuses to compile too,
while ``fp8e5`` compiles fine. Treat e4m3 failures as expected on A100 and read the
``fp8_e5m2`` rows for the real signal; both dtypes must pass on sm_89+.
"""
import sys, torch
from vortex_torch.engine.sgl.host_kv import HostKVCache, set_host_latent, gather_host_tokens
from vortex_torch.engine.sgl.cache_policy import POLICIES, WAYS

BLK, D, NB = 32, 128, 2048
ok = True
for dt_name, dt in (("bfloat16", torch.bfloat16),
                    ("fp8_e4m3", torch.float8_e4m3fn),
                    ("fp8_e5m2", torch.float8_e5m2)):
    # build a pinned host buffer of this dtype (fp8 has no .normal_(), so cast)
    src = (torch.randn(NB, BLK, D) * 0.5).to(dt)
    hk = torch.empty(NB, BLK, D, dtype=dt, pin_memory=True); hk.copy_(src)
    hv = torch.empty(NB, BLK, D, dtype=dt, pin_memory=True); hv.copy_(src)
    for pol in POLICIES:
        cap = (-(-NB // (WAYS - 1)) * WAYS) if pol == "full" else 1024
        try:
            c = HostKVCache(hk, hv, cap, "cuda", policy=pol)
        except Exception as e:
            print(f"  {dt_name:<9} {pol:<5} CONSTRUCT FAILED: {type(e).__name__}: {e}")
            ok = False; continue
        rows, mpr = 8, 32
        ids = torch.randint(0, NB, (rows, mpr), device="cuda", dtype=torch.int32)
        rl = torch.full((rows,), mpr * BLK, dtype=torch.int32, device="cuda")
        c.reserve_remap(ids)
        try:
            c.tick()
            dk, dv, out = c.fetch(ids.clone(), row_lens=rl, num_rows=rows, max_per_row=mpr)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  {dt_name:<9} {pol:<5} FETCH FAILED: {type(e).__name__}: {e}")
            ok = False; del c; continue
        # bit-exact comparison: view as uint8 so fp8 needs no arithmetic
        slot = out.cpu().long(); want = ids.cpu().long()
        dk8 = dk.view(torch.uint8).cpu(); hk8 = hk.view(torch.uint8)
        bad = 0
        for r in range(rows):
            for j in range(mpr):
                s = int(slot[r, j])
                if s == c.zero_slot:
                    if int(dk8[s].max()) != 0: bad += 1
                    continue
                if not torch.equal(dk8[s], hk8[int(want[r, j])]): bad += 1
        good = bad == 0
        ok &= good
        print(f"  {dt_name:<9} {pol:<5} cap={c.capacity:<5} bit-exact_errors={bad} "
              f"overflow={int(c.overflow.item())} {'ok' if good else 'FAIL'}")
        del c; torch.cuda.empty_cache()

# MLA latent writer + token gather under fp8
print("\n  MLA helpers under fp8_e4m3:")
LAT, RANK, ROPE = 576, 512, 64
for dt_name, dt in (("bfloat16", torch.bfloat16), ("fp8_e4m3", torch.float8_e4m3fn)):
    buf = torch.zeros(4096, 1, LAT, dtype=dt, pin_memory=True)
    loc = torch.arange(8, device="cuda", dtype=torch.int64)
    nope = (torch.randn(8, 1, RANK) * .5).to(dt).cuda()
    rope = (torch.randn(8, 1, ROPE) * .5).to(dt).cuda()
    try:
        set_host_latent(buf, loc, nope, rope); torch.cuda.synchronize()
        # Compare in the NATIVE dtype. A uint8 view cannot be sliced with element
        # indices once the dtype is wider than a byte: bf16 doubles the last
        # dimension, so [:RANK] takes half the payload and the check fails for a
        # correct copy (fp8 happens to pass only because 1 byte == 1 element).
        # Copies here are exact in every dtype, so native equality is the right test.
        e1 = torch.equal(buf[:8, 0, :RANK], nope.cpu()[:, 0, :])
        e2 = torch.equal(buf[:8, 0, RANK:], rope.cpu()[:, 0, :])
        g = gather_host_tokens(buf.view(4096, LAT), loc.to(torch.int32))
        torch.cuda.synchronize()
        e3 = torch.equal(g.cpu()[:8], buf[:8, 0, :])
        good = e1 and e2 and e3
        ok &= good
        print(f"    {dt_name:<9} set_host_latent kv_c={e1} k_pe={e2} gather={e3} "
              f"{'ok' if good else 'FAIL'}")
    except Exception as e:
        print(f"    {dt_name:<9} FAILED: {type(e).__name__}: {e}")
        ok = False
    del buf; torch.cuda.empty_cache()

print("\n" + ("FP8 SUPPORTED across every policy" if ok else "*** FP8 ISSUES ***"))
raise SystemExit(0 if ok else 1)
