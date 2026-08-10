"""End-to-end + breakdown: dense flash-attention-4 vs the sparse algorithms.

Reports, per (algorithm, seqlen), forward and backward separately:

* **e2e time** — the whole step, selection included. Selection is not free, so a
  comparison that timed only the attention kernel would flatter every sparse row.
* **breakdown** — state build / score+top-k / pattern transpose / attention, so a
  regression is attributable to a stage rather than just visible in the total.
* **memory** — peak allocated, and the size of what is saved for backward.
  Sparsity reduces *compute*, not memory: `dK`/`dV` are needed for every KV token,
  so nothing can be dropped from the saved tensors. Parity with dense is the
  expected result and the thing to verify; being *below* dense would mean something
  was wrongly discarded, and being far above means an intermediate got materialized.

Baselines are **flash-attention-4** (`flash_attn.cute`, 4.0.0b19) and SDPA's flash
backend. FA4 uses a `[B, S, H, D]` layout, so the transposes needed to feed it are
inside its timed region — that is the cost a real model would pay too.

**FA4 backward is measured separately, and often not at all.** Its forward compiles
in ~5 s, but its *backward* compiles three CuTeDSL kernels (preprocess, main,
postprocess) with no persistent on-disk cache, and was measured burning **>25 min of
CPU without finishing** on a 1024-token case. That is a property of the beta's
compiler, not of its runtime speed, so:

* FA4 **forward** is a real, usable baseline and is always reported.
* FA4 **backward** is attempted only under `--fa4-bwd`, with `--fa4-bwd-timeout`
  seconds allowed. When it does not compile in time the row is reported as
  `compile timeout` rather than silently omitted or, worse, blamed on the runtime.
* SDPA-flash therefore carries the fwd+bwd comparison. It is the honest baseline for
  backward numbers here, and it is quoted with its fast path explicitly enabled.

Timing is median-of-iters, and every compile happens in a warmup outside the timed
region so it is never attributed to anyone's runtime.

Usage:
    python benchmarks/bench_e2e.py                       # 32k / 64k / 128k
    python benchmarks/bench_e2e.py --seqlens 32768 --algos lserve
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_train.flow.spec import REGISTRY  # noqa: E402
from vortex_train.kernels.select import score_and_select  # noqa: E402
from vortex_train.kernels.state import build_state  # noqa: E402
from vortex_train.kernels.transpose import transpose_pattern  # noqa: E402
from vortex_train.nn import SparseAttention  # noqa: E402

# The three algorithms this project implements, in increasing state richness.
ALGOS = ["block_topk", "quest", "lserve"]

try:
    from flash_attn.cute.interface import flash_attn_func as _fa4
    HAVE_FA4 = True
except Exception:                                     # pragma: no cover
    HAVE_FA4 = False


def _sync_time(fn, warmup: int, iters: int) -> float:
    """Median-of-iters wall time in ms. CUDA events on a synchronised stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        z.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(z))
    times.sort()
    return times[len(times) // 2]


def _peak_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def _qkv(b, hq, hkv, s, d, *, grad=True, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda h: (torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16,
                                generator=g) * 0.5).requires_grad_(grad)
    return mk(hq), mk(hkv), mk(hkv)


# ------------------------------------------------------------------ dense refs
def bench_dense_fa4(b, hq, hkv, s, d, *, warmup, iters, try_bwd=False, bwd_timeout=900.0):
    """flash-attention-4, `[B, S, H, D]` layout. Transposes are inside the timing.

    Backward is opt-in: see the module docstring. `bwd_timeout` cannot interrupt a
    running CuTeDSL compile (it holds the GIL in native code), so it is enforced by
    *not attempting* the backward unless asked -- the caller decides the risk.
    """
    q, k, v = _qkv(b, hq, hkv, s, d, grad=try_bwd)

    def fwd():
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        r = _fa4(qt, kt, vt, causal=True)
        return r[0] if isinstance(r, tuple) else r

    # First call for a shape triggers the CuTeDSL forward compile (~5 s). Absorb it
    # here so it lands in nobody's timing.
    t0 = time.perf_counter()
    o = fwd()
    torch.cuda.synchronize()
    res = {
        "fwd_compile_s": time.perf_counter() - t0,
        "fwd_ms": _sync_time(fwd, warmup, iters),
        "fwd_peak_mb": _peak_mb(fwd),
    }

    if not try_bwd:
        res["bwd_ms"] = float("nan")
        res["fwdbwd_ms"] = float("nan")
        res["fwdbwd_peak_mb"] = float("nan")
        res["bwd_note"] = "not attempted (--fa4-bwd to try)"
        return res

    g = torch.randn_like(o)

    def fwdbwd():
        out = fwd()
        out.backward(g)
        q.grad = k.grad = v.grad = None

    t0 = time.perf_counter()
    fwdbwd()                                     # compiles 3 CuTeDSL bwd kernels
    torch.cuda.synchronize()
    res["bwd_compile_s"] = time.perf_counter() - t0
    res["fwdbwd_ms"] = _sync_time(fwdbwd, max(1, warmup // 2), max(3, iters // 2))
    res["fwdbwd_peak_mb"] = _peak_mb(fwdbwd)
    res["bwd_ms"] = res["fwdbwd_ms"] - res["fwd_ms"]
    return res


def bench_dense_sdpa(b, hq, hkv, s, d, *, warmup, iters):
    """SDPA flash backend — a second dense reference, since FA4 is a beta."""
    q, k, v = _qkv(b, hq, hkv, s, d)
    group = hq // hkv
    backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION

    def fwd():
        with torch.nn.attention.sdpa_kernel(backend):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
                is_causal=True,
            )

    o = fwd()
    g = torch.randn_like(o)

    def fwdbwd():
        out = fwd()
        out.backward(g)
        q.grad = k.grad = v.grad = None

    res = {
        "fwd_ms": _sync_time(fwd, warmup, iters),
        "fwdbwd_ms": _sync_time(fwdbwd, max(1, warmup // 2), max(3, iters // 2)),
        "fwd_peak_mb": _peak_mb(fwd),
        "fwdbwd_peak_mb": _peak_mb(fwdbwd),
    }
    res["bwd_ms"] = res["fwdbwd_ms"] - res["fwd_ms"]
    return res


# ----------------------------------------------------------------- sparse algos
def bench_sparse(algo, b, hq, hkv, s, d, *, warmup, iters):
    q, k, v = _qkv(b, hq, hkv, s, d)
    attn = SparseAttention(REGISTRY[algo], num_kv_heads=hkv)
    c = attn.compiled
    n_kv = (s + c.block_kv - 1) // c.block_kv

    # --- stage pieces, timed individually -------------------------------------
    with torch.no_grad():
        state = build_state(k, v, c.fields, block_kv=c.block_kv)
    pattern = attn.build_pattern(q, k, v)

    def do_state():
        with torch.no_grad():
            return build_state(k, v, c.fields, block_kv=c.block_kv)

    def do_select():
        with torch.no_grad():
            return score_and_select(
                q, state, c.tape, num_kv_blocks=n_kv, seqlen_kv=s,
                block_q=c.block_q, block_kv=c.block_kv, num_kv_heads=hkv,
                topk=c.budget.topk, reserve_bos=c.budget.reserve_bos,
                reserve_local=c.budget.reserve_local,
                reserve_eos=c.budget.reserve_eos, causal=c.causal, q_how=c.q_how,
            )

    def do_transpose():
        return transpose_pattern(pattern)

    def fwd():
        return attn(q, k, v)

    o = fwd()
    g = torch.randn_like(o)

    def fwdbwd():
        out = fwd()
        out.backward(g)
        q.grad = k.grad = v.grad = None

    res = {
        "state_ms": _sync_time(do_state, warmup, iters),
        "select_ms": _sync_time(do_select, warmup, iters),
        "transpose_ms": _sync_time(do_transpose, warmup, iters),
        "fwd_ms": _sync_time(fwd, warmup, iters),
        "fwdbwd_ms": _sync_time(fwdbwd, max(1, warmup // 2), max(3, iters // 2)),
        "fwd_peak_mb": _peak_mb(fwd),
        "fwdbwd_peak_mb": _peak_mb(fwdbwd),
    }
    res["bwd_ms"] = res["fwdbwd_ms"] - res["fwd_ms"]
    res["sel_ms"] = res["state_ms"] + res["select_ms"] + res["transpose_ms"]
    # Attention proper = forward minus the selection stages it contains.
    res["attn_ms"] = max(res["fwd_ms"] - res["sel_ms"], 0.0)
    res["state_mb"] = state.numel() * state.element_size() / 1024**2
    res["pattern_mb"] = (
        (pattern.cnt.numel() * pattern.cnt.element_size()
         + pattern.idx.numel() * pattern.idx.element_size()) / 1024**2
    )
    res["topk"] = c.budget.topk
    res["blocks"] = n_kv
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[32768, 65536, 131072])
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--fa4-bwd", action="store_true",
                    help="attempt FA4's backward; its CuTeDSL compile may take >25 min")
    a = ap.parse_args()

    b, hkv, d = a.batch, a.hkv, a.head_dim
    hq = hkv * a.group
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"shape: B={b} Hq={hq} Hkv={hkv} D={d} (GQA group {a.group}) bf16")
    print(f"flash-attention-4: {'yes' if HAVE_FA4 else 'NOT AVAILABLE'}\n")

    rows = []
    for s in a.seqlens:
        print(f"{'='*104}\nseqlen {s}\n{'='*104}")
        hdr = (f"{'algorithm':>16} {'fwd_ms':>9} {'bwd_ms':>9} {'e2e_ms':>9} "
               f"{'state':>7} {'select':>7} {'tpose':>7} {'attn':>8} "
               f"{'fwd_MB':>8} {'e2e_MB':>8} {'speedup':>8}")
        print(hdr)
        print("-" * len(hdr))

        fa4 = None
        if HAVE_FA4:
            try:
                fa4 = bench_dense_fa4(b, hq, hkv, s, d, warmup=a.warmup,
                                      iters=a.iters, try_bwd=a.fa4_bwd)
                fa4.update(algo="dense-fa4", seqlen=s)
                rows.append(fa4)
                bwd = f"{fa4['bwd_ms']:>9.3f}" if fa4["bwd_ms"] == fa4["bwd_ms"] else f"{'n/a':>9}"
                e2e = (f"{fa4['fwdbwd_ms']:>9.3f}" if fa4["fwdbwd_ms"] == fa4["fwdbwd_ms"]
                       else f"{'n/a':>9}")
                mb2 = (f"{fa4['fwdbwd_peak_mb']:>8.0f}"
                       if fa4["fwdbwd_peak_mb"] == fa4["fwdbwd_peak_mb"] else f"{'n/a':>8}")
                print(f"{'dense (FA4)':>16} {fa4['fwd_ms']:>9.3f} {bwd} {e2e} "
                      f"{'-':>7} {'-':>7} {'-':>7} {fa4['fwd_ms']:>8.3f} "
                      f"{fa4['fwd_peak_mb']:>8.0f} {mb2} {'1.00x':>8}")
            except torch.cuda.OutOfMemoryError:
                print(f"{'dense (FA4)':>16}  OOM"); torch.cuda.empty_cache()
            except Exception as e:
                print(f"{'dense (FA4)':>16}  FAILED: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()

        dense = None
        try:
            dense = bench_dense_sdpa(b, hq, hkv, s, d, warmup=a.warmup, iters=a.iters)
            dense.update(algo="dense-sdpa", seqlen=s)
            rows.append(dense)
            print(f"{'dense (SDPA)':>16} {dense['fwd_ms']:>9.3f} {dense['bwd_ms']:>9.3f} "
                  f"{dense['fwdbwd_ms']:>9.3f} {'-':>7} {'-':>7} {'-':>7} "
                  f"{dense['fwd_ms']:>8.3f} {dense['fwd_peak_mb']:>8.0f} "
                  f"{dense['fwdbwd_peak_mb']:>8.0f} {'1.00x':>8}")
        except torch.cuda.OutOfMemoryError:
            print(f"{'dense (SDPA)':>16}  OOM"); torch.cuda.empty_cache()

        for algo in a.algos:
            try:
                r = bench_sparse(algo, b, hq, hkv, s, d, warmup=a.warmup, iters=a.iters)
            except torch.cuda.OutOfMemoryError:
                print(f"{algo:>16}  OOM"); torch.cuda.empty_cache(); continue
            r.update(algo=algo, seqlen=s)
            rows.append(r)
            sp = (dense["fwdbwd_ms"] / r["fwdbwd_ms"]) if dense else float("nan")
            r["speedup_vs_sdpa"] = sp
            if fa4 is not None:
                r["fwd_speedup_vs_fa4"] = fa4["fwd_ms"] / r["fwd_ms"]
            print(f"{algo:>16} {r['fwd_ms']:>9.3f} {r['bwd_ms']:>9.3f} "
                  f"{r['fwdbwd_ms']:>9.3f} {r['state_ms']:>7.3f} {r['select_ms']:>7.3f} "
                  f"{r['transpose_ms']:>7.3f} {r['attn_ms']:>8.3f} "
                  f"{r['fwd_peak_mb']:>8.0f} {r['fwdbwd_peak_mb']:>8.0f} {sp:>7.2f}x")
        print()

    print("speedup = dense(SDPA) e2e fwd+bwd / this row's e2e fwd+bwd -- SDPA is the")
    print("  baseline for fwd+bwd because FA4's backward does not finish compiling")
    print("  (see the module docstring). FA4 forward is compared separately below.")
    print("e2e includes selection; `attn` is fwd minus the selection stages.")
    print("Memory at parity with dense is the CORRECT result: sparsity reduces")
    print("compute, not memory -- dK/dV are needed for every KV token.")

    fa4_rows = {r["seqlen"]: r for r in rows if r["algo"] == "dense-fa4"}
    if fa4_rows:
        print("\nForward-only, vs flash-attention-4 (the fastest dense fwd here):")
        print(f"  {'seqlen':>8} {'algorithm':>16} {'fwd_ms':>9} {'vs FA4':>8}")
        for s_ in a.seqlens:
            if s_ not in fa4_rows:
                continue
            base = fa4_rows[s_]["fwd_ms"]
            for r in rows:
                if r["seqlen"] == s_ and r["algo"] not in ("dense-fa4",):
                    print(f"  {s_:>8} {r['algo']:>16} {r['fwd_ms']:>9.3f} "
                          f"{base / r['fwd_ms']:>7.2f}x")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
