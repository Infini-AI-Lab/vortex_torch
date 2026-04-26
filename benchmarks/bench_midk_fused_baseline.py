"""Quick fused baseline measurement at mid-K (K in {64,128,256,512}).

Goal: establish the bar that any adaptive split implementation has to beat
before we commit to building / templating SELECTK_SORTK kernels.

Output: bench_results/midk_fused_baseline.csv
        + a printed table per K.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import vortex_torch_C as V


def time_kernel_us(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(repeat))
    n = len(times)
    mean = sum(times) / n
    var  = sum((t - mean) ** 2 for t in times) / n
    return dict(mean=mean, p50=times[n // 2], p90=times[min(n - 1, int(round(n * 0.9)))],
                min=times[0], max=times[-1], std=math.sqrt(var))


def make_inputs(B, pages, K, dtype=torch.bfloat16, reserved_bos=1, reserved_eos=2):
    device = torch.device("cuda")
    dense_kv_indptr  = torch.arange(B + 1, device=device, dtype=torch.int32) * pages
    sparse_kv_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * (K + reserved_bos + reserved_eos)
    total = B * pages
    torch.manual_seed(0)
    scores = torch.randn(total, device=device, dtype=dtype)
    dense_kv_indices = torch.arange(total, device=device, dtype=torch.int32)
    out = torch.full((B * (K + reserved_bos + reserved_eos),), -1, device=device, dtype=torch.int32)
    return scores, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, out


def call_fused(scores, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, out,
               B, K, reserved_bos, reserved_eos, pages, mapping_mode):
    V.topk_output_sglang_fused(
        scores, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, out,
        B, K, reserved_bos, reserved_eos, pages,
        mapping_mode, 0.5, None, None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results/midk_fused_baseline.csv")
    ap.add_argument("--pages", nargs="+", type=int, default=[16384, 32768, 65536, 131072])
    ap.add_argument("--ks", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--mappings", nargs="+", type=int, default=[0, 8])  # NONE, TRUNC8
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    device = torch.cuda.get_device_properties(0)
    print(f"# GPU: {device.name}, SMs={device.multi_processor_count}")
    print(f"# pages: {args.pages}")
    print(f"# Ks:    {args.ks}")
    print(f"# Bs:    {args.batches}")
    print(f"# maps:  {args.mappings} (0=NONE, 8=TRUNC8)")
    print()

    rows = []
    for K in args.ks:
        print(f"=== K={K} ===")
        print(f"{'pages':>8s} {'B':>3s} {'map':>5s}  {'mean_us':>10s} {'p50_us':>10s} "
              f"{'min_us':>10s} {'std_us':>8s}  status")
        for pages in args.pages:
            for B in args.batches:
                for mapping in args.mappings:
                    map_name = {0: "NONE", 8: "TRUNC8"}.get(mapping, str(mapping))
                    try:
                        ins = make_inputs(B, pages, K)
                        # warmup correctness check
                        call_fused(*ins, B, K, 1, 2, pages, mapping)
                        torch.cuda.synchronize()
                    except Exception as e:
                        print(f"{pages:>8d} {B:>3d} {map_name:>5s}  {'-':>10s} {'-':>10s} "
                              f"{'-':>10s} {'-':>8s}  FAILED: {str(e)[:80]}")
                        rows.append(dict(K=K, pages=pages, B=B, mapping=map_name,
                                         mean_us=None, p50_us=None, p90_us=None,
                                         min_us=None, max_us=None, std_us=None,
                                         status="failed", error=str(e)[:200]))
                        continue
                    t = time_kernel_us(
                        lambda: call_fused(*ins, B, K, 1, 2, pages, mapping),
                        warmup=args.warmup, repeat=args.repeat,
                    )
                    print(f"{pages:>8d} {B:>3d} {map_name:>5s}  {t['mean']:>10.3f} "
                          f"{t['p50']:>10.3f} {t['min']:>10.3f} {t['std']:>8.3f}  ok")
                    rows.append(dict(K=K, pages=pages, B=B, mapping=map_name,
                                     mean_us=t['mean'], p50_us=t['p50'], p90_us=t['p90'],
                                     min_us=t['min'], max_us=t['max'], std_us=t['std'],
                                     status="ok", error=""))
                    del ins
        print()

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
