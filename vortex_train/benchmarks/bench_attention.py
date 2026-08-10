"""Speed benchmark: block-sparse vs dense, forward and backward.

Reporting rules this script follows, because sparse-attention numbers are easy to
overstate:

1. **Compare against dense with its fast path on.** The baseline is
   `torch.nn.functional.scaled_dot_product_attention` with the flash backend
   (or flash_attn if installed) — not a masked-SDPA reference, which would be an
   unfairly slow baseline and inflate the speedup.
2. **Sweep GQA group size.** Group size, not sparsity, is often the binding
   variable for sparse-attention kernels: at small group the tile is too small to
   fill the MMA and padded math dominates. A speedup quoted without the group
   size is not interpretable.
3. **Report memory too** — not because sparsity saves memory (it does not; see
   docs/), but to *prove* nothing was materialized. Peak memory must be
   independent of topk.
4. **Include the pattern build.** The scorer/transpose cost is part of the price;
   timing only the attention kernel would flatter the result.

Usage:
    python benchmarks/bench_attention.py                  # default sweep
    python benchmarks/bench_attention.py --seqlens 8192 32768 --groups 1 4 8
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

# Runnable as a plain script from anywhere (`python benchmarks/bench_attention.py`)
# without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_train.kernels.transpose import transpose_pattern
from vortex_train.nn.functional import sparse_attention
from vortex_train.pattern import pattern_from_dense_mask


def _causal_topk_pattern(b, hkv, seqlen, block, topk, device):
    """The realistic recipe: BOS sink + local window + random top-k, causal."""
    m = n = seqlen // block
    mi = torch.arange(m, device=device)
    ni = torch.arange(n, device=device)
    causal = ni[None, :] <= mi[:, None]
    score = torch.rand((b, hkv, m, n), device=device).masked_fill(~causal[None, None], -1.0)
    k = min(topk, n)
    sel = score.topk(k, dim=3).indices
    mask = torch.zeros((b, hkv, m, n), dtype=torch.bool, device=device)
    mask.scatter_(3, sel, True)
    mask[:, :, :, 0] = True                                   # sink
    mask[:, :, mi, mi] = True                                 # own block
    mask &= causal[None, None]
    return pattern_from_dense_mask(
        mask, block_q=block, block_kv=block, seqlen_q=seqlen, seqlen_kv=seqlen
    )


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def bench(seqlen, group, *, b=1, hkv=8, d=128, block=64, topk=16, backward=True):
    dev = "cuda"
    hq = hkv * group
    q = torch.randn(b, hq, seqlen, d, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    k = torch.randn(b, hkv, seqlen, d, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    v = torch.randn(b, hkv, seqlen, d, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    pattern = _causal_topk_pattern(b, hkv, seqlen, block, topk, dev)

    def dense_fwd():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1), is_causal=True
            )

    def sparse_fwd():
        return sparse_attention(q, k, v, pattern)

    def pattern_build():
        return transpose_pattern(pattern)

    res = {"seqlen": seqlen, "group": group, "topk": topk, "block": block}
    with contextlib.suppress(Exception):
        res["dense_fwd_ms"] = _time(dense_fwd)
    res["sparse_fwd_ms"] = _time(sparse_fwd)
    res["transpose_ms"] = _time(pattern_build)
    res["sparse_peak_mb"] = _peak_mb(sparse_fwd)

    if backward:
        def dense_bwd():
            o = dense_fwd(); o.backward(torch.ones_like(o), retain_graph=False)
            q.grad = k.grad = v.grad = None

        def sparse_bwd():
            o = sparse_fwd(); o.backward(torch.ones_like(o), retain_graph=False)
            q.grad = k.grad = v.grad = None

        with contextlib.suppress(Exception):
            res["dense_fwdbwd_ms"] = _time(dense_bwd, warmup=3, iters=10)
        res["sparse_fwdbwd_ms"] = _time(sparse_bwd, warmup=3, iters=10)
        res["sparse_fwdbwd_peak_mb"] = _peak_mb(sparse_bwd)

    # FLOP accounting: causal dense is ~half of S^2; sparse is topk*block per query
    res["flop_ratio"] = (seqlen / 2) / (topk * block)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--no-backward", action="store_true")
    a = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    hdr = (f"{'seqlen':>7} {'grp':>4} {'dense_f':>9} {'sparse_f':>9} {'f_x':>6} "
           f"{'dense_fb':>9} {'sparse_fb':>10} {'fb_x':>6} {'tpose':>7} "
           f"{'peak_MB':>8} {'flop_x':>7}")
    print(hdr); print("-" * len(hdr))
    for s in a.seqlens:
        for g in a.groups:
            try:
                r = bench(s, g, topk=a.topk, block=a.block, backward=not a.no_backward)
            except torch.cuda.OutOfMemoryError:
                print(f"{s:>7} {g:>4}  OOM"); torch.cuda.empty_cache(); continue
            df, sf = r.get("dense_fwd_ms"), r["sparse_fwd_ms"]
            dfb, sfb = r.get("dense_fwdbwd_ms"), r.get("sparse_fwdbwd_ms")
            fx = f"{df/sf:.2f}x" if df else "n/a"
            fbx = f"{dfb/sfb:.2f}x" if (dfb and sfb) else "n/a"
            print(f"{s:>7} {g:>4} {df or float('nan'):>9.3f} {sf:>9.3f} {fx:>6} "
                  f"{dfb or float('nan'):>9.3f} {sfb or float('nan'):>10.3f} {fbx:>6} "
                  f"{r['transpose_ms']:>7.3f} {r.get('sparse_fwdbwd_peak_mb', r['sparse_peak_mb']):>8.0f} "
                  f"{r['flop_ratio']:>6.0f}x")
    print("\nf_x / fb_x = dense/sparse latency (>1 means sparse is faster).")
    print("flop_x is the theoretical attention-FLOP reduction — the gap between")
    print("flop_x and fb_x is kernel efficiency left on the table.")


if __name__ == "__main__":
    main()
