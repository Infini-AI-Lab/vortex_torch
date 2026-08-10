"""Speed of the selection path: state build + fused score/top-k + transpose.

Selection is pure overhead added on top of the attention win, so it is the number
that decides whether the frontend is worth having. Two things this reports that a
kernel microbenchmark would hide:

1. **Selection as a fraction of the step.** A policy that costs 30% of the
   attention it saves is a bad trade regardless of how fast the kernel is in
   isolation.
2. **Fusion evidence.** Peak memory must not scale with ``Nkv²``. An unfused
   scorer would materialize the ``[B, Hkv, Mq, Nkv]`` score matrix — 0.06 GB per
   layer at 128k, 4 GB at 1M — so a flat memory curve across sequence length is
   what proves the score stayed in registers.

Usage:
    python benchmarks/bench_selection.py
    python benchmarks/bench_selection.py --seqlens 32768 131072 --policies block_topk quest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_train.flow.spec import REGISTRY  # noqa: E402
from vortex_train.kernels.select import score_and_select  # noqa: E402
from vortex_train.kernels.state import build_state  # noqa: E402
from vortex_train.kernels.transpose import transpose_pattern  # noqa: E402
from vortex_train.nn import SparseAttention  # noqa: E402


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    z = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    z.record()
    torch.cuda.synchronize()
    return a.elapsed_time(z) / iters


def bench(policy, seqlen, group, *, b=1, hkv=8, d=128):
    dev = "cuda"
    hq = hkv * group
    q = torch.randn(b, hq, seqlen, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, hkv, seqlen, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, hkv, seqlen, d, device=dev, dtype=torch.bfloat16, requires_grad=True)

    attn = SparseAttention(REGISTRY[policy], num_kv_heads=hkv)
    c = attn.compiled
    n_kv = (seqlen + c.block_kv - 1) // c.block_kv

    with torch.no_grad():
        state = build_state(k, v, c.fields, block_kv=c.block_kv)

    def do_state():
        with torch.no_grad():
            return build_state(k, v, c.fields, block_kv=c.block_kv)

    def do_select():
        with torch.no_grad():
            return score_and_select(
                q, state, c.tape, num_kv_blocks=n_kv, seqlen_kv=seqlen,
                block_q=c.block_q, block_kv=c.block_kv, num_kv_heads=hkv,
                topk=c.budget.topk, reserve_bos=c.budget.reserve_bos,
                reserve_local=c.budget.reserve_local, reserve_eos=c.budget.reserve_eos,
                causal=c.causal, q_how=c.q_how,
            )

    pattern = attn.build_pattern(q, k, v)

    def do_fwdbwd():
        o = attn(q, k, v)
        o.backward(torch.ones_like(o))
        q.grad = k.grad = v.grad = None

    res = {
        "state_ms": _time(do_state),
        "select_ms": _time(do_select),
        "transpose_ms": _time(lambda: transpose_pattern(pattern)),
        "total_ms": _time(do_fwdbwd, warmup=3, iters=10),
    }
    res["sel_total_ms"] = res["state_ms"] + res["select_ms"] + res["transpose_ms"]
    res["sel_pct"] = 100.0 * res["sel_total_ms"] / res["total_ms"]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    do_fwdbwd()
    torch.cuda.synchronize()
    res["peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    # What an unfused scorer would have had to materialize, for contrast.
    res["score_matrix_mb"] = b * hkv * (seqlen // c.block_q) * n_kv * 4 / 1024**2
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[4096, 16384, 65536])
    ap.add_argument("--groups", type=int, nargs="+", default=[4])
    ap.add_argument("--policies", nargs="+", default=["block_topk", "quest", "streaming"])
    a = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    hdr = (f"{'policy':>15} {'seqlen':>7} {'grp':>4} {'state':>7} {'select':>7} "
           f"{'tpose':>7} {'sel_tot':>8} {'step':>9} {'sel%':>6} {'peak_MB':>8} "
           f"{'unfused_MB':>10}")
    print(hdr)
    print("-" * len(hdr))
    for pol in a.policies:
        for s in a.seqlens:
            for g in a.groups:
                try:
                    r = bench(pol, s, g)
                except torch.cuda.OutOfMemoryError:
                    print(f"{pol:>15} {s:>7} {g:>4}  OOM")
                    torch.cuda.empty_cache()
                    continue
                print(f"{pol:>15} {s:>7} {g:>4} {r['state_ms']:>7.3f} {r['select_ms']:>7.3f} "
                      f"{r['transpose_ms']:>7.3f} {r['sel_total_ms']:>8.3f} "
                      f"{r['total_ms']:>9.3f} {r['sel_pct']:>5.1f}% {r['peak_mb']:>8.0f} "
                      f"{r['score_matrix_mb']:>10.1f}")
    print("\nsel% = selection cost as a fraction of the whole fwd+bwd step.")
    print("unfused_MB = the block-score matrix an unfused scorer would materialize;")
    print("peak_MB should be flat against it, which is the evidence of fusion.")


if __name__ == "__main__":
    main()
