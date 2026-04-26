"""Decompose the adaptive split-2 TopK kernel's latency into Phase-1,
Phase-2, and barrier+launch overhead — and compare against the naive
CUB sort (topk.cu) and the single-CTA radix baseline (topk_sglang.cu).

No remap (mode=0), bfloat16 scores only, to keep the comparison clean.

Usage:
    python benchmarks/profile_adaptive_overhead.py [--gpu 4]
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List

import torch

from vortex_torch_C import (
    topk_output,
    topk_output_sglang,
    topk_output_adaptive,
    topk_adaptive_phase1_only,
    topk_adaptive_phase2_only,
)


def make_inputs(bs: int, pages: int, K: int, reserved_bos: int = 1, reserved_eos: int = 1,
                device: str = "cuda") -> Dict[str, torch.Tensor]:
    per_row = pages + reserved_bos + reserved_eos
    dense_kv_indptr = torch.arange(
        0, (bs + 1) * per_row, per_row, device=device, dtype=torch.int32)
    dense_kv_indices = torch.arange(bs * per_row, device=device, dtype=torch.int32)
    per_sparse = K + reserved_bos + reserved_eos
    sparse_kv_indptr = torch.arange(
        0, (bs + 1) * per_sparse, per_sparse, device=device, dtype=torch.int32)
    sparse_kv_indices = torch.zeros(bs * per_sparse, device=device, dtype=torch.int32)
    x = torch.randn(bs * per_row, device=device, dtype=torch.bfloat16)
    partial_scores = torch.empty(bs * 2 * K, device=device, dtype=torch.float32)
    partial_indices = torch.empty(bs * 2 * K, device=device, dtype=torch.int32)
    return dict(
        x=x,
        dense_kv_indptr=dense_kv_indptr,
        dense_kv_indices=dense_kv_indices,
        sparse_kv_indptr=sparse_kv_indptr,
        sparse_kv_indices=sparse_kv_indices,
        partial_scores=partial_scores,
        partial_indices=partial_indices,
    )


def time_kernel(fn, args, warmup: int = 20, repeat: int = 200) -> float:
    """Return mean ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times) / len(times)


def run_config(bs: int, pages: int, K: int, reserved_bos: int = 1, reserved_eos: int = 1,
               warmup: int = 20, repeat: int = 200) -> Dict[str, float]:
    inp = make_inputs(bs, pages, K, reserved_bos, reserved_eos)

    # --- baseline: topk_output_sglang (single-CTA radix-select, mode=0) ---
    sglang_args = (
        inp["x"], inp["dense_kv_indptr"], inp["sparse_kv_indptr"],
        inp["dense_kv_indices"], inp["sparse_kv_indices"],
        bs, K, reserved_bos, reserved_eos, pages,
    )
    sglang_ms = time_kernel(topk_output_sglang, sglang_args, warmup, repeat)

    # --- naive CUB sort: topk_output (only if pages <= 8192 — template ladder limit) ---
    naive_ms = float("nan")
    if pages <= 8192:
        naive_args = (
            inp["x"], inp["dense_kv_indptr"], inp["dense_kv_indices"],
            inp["sparse_kv_indptr"], inp["sparse_kv_indices"],
            bs, K, reserved_bos, reserved_eos, pages,
        )
        try:
            naive_ms = time_kernel(topk_output, naive_args, warmup, repeat)
        except RuntimeError as e:
            print(f"[naive skip] bs={bs} pages={pages} K={K}: {e}")

    # --- adaptive full ---
    adaptive_args = (
        inp["x"], inp["dense_kv_indptr"], inp["sparse_kv_indptr"],
        inp["dense_kv_indices"], inp["sparse_kv_indices"],
        bs, K, reserved_bos, reserved_eos, pages,
        0,    # mapping_mode = NONE
        0.5,  # mapping_power (unused)
    )
    adaptive_ms = time_kernel(topk_output_adaptive, adaptive_args, warmup, repeat)

    # --- adaptive Phase 1 only ---
    p1_args = (
        inp["x"], inp["dense_kv_indptr"], inp["dense_kv_indices"],
        inp["partial_scores"], inp["partial_indices"],
        bs, K, reserved_bos, reserved_eos, pages,
    )
    p1_ms = time_kernel(topk_adaptive_phase1_only, p1_args, warmup, repeat)

    # --- adaptive Phase 2 only (workspace pre-populated by the last p1 call) ---
    p2_args = (
        inp["partial_scores"], inp["partial_indices"],
        inp["sparse_kv_indptr"], inp["sparse_kv_indices"],
        bs, K, reserved_bos,
    )
    p2_ms = time_kernel(topk_adaptive_phase2_only, p2_args, warmup, repeat)

    overhead_ms = adaptive_ms - (p1_ms + p2_ms)

    return {
        "bs": bs, "pages": pages, "K": K,
        "naive_ms": naive_ms,
        "sglang_ms": sglang_ms,
        "adaptive_ms": adaptive_ms,
        "phase1_ms": p1_ms,
        "phase2_ms": p2_ms,
        "p1_plus_p2_ms": p1_ms + p2_ms,
        "overhead_ms": overhead_ms,
        "overhead_frac": overhead_ms / adaptive_ms if adaptive_ms else 0.0,
        "adaptive_vs_sglang": adaptive_ms / sglang_ms if sglang_ms else float("nan"),
    }


def _fmt(v, w=9):
    if isinstance(v, float) and math.isnan(v):
        return f"{'—':>{w}s}"
    if isinstance(v, float):
        return f"{v:>{w}.4f}"
    return f"{str(v):>{w}s}"


def print_table(rows: List[dict]) -> None:
    hdr = (f"{'bs':>3s} {'pages':>6s} {'K':>5s}  {'naive':>9s} {'sglang':>9s} "
           f"{'adaptive':>9s} {'phase1':>9s} {'phase2':>9s} {'p1+p2':>9s} "
           f"{'overhead':>9s} {'ovh%':>6s} {'a/sglang':>9s}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        ovh_pct = 100.0 * r["overhead_frac"]
        print(f"{r['bs']:>3d} {r['pages']:>6d} {r['K']:>5d}  "
              f"{_fmt(r['naive_ms'])} {_fmt(r['sglang_ms'])} "
              f"{_fmt(r['adaptive_ms'])} {_fmt(r['phase1_ms'])} {_fmt(r['phase2_ms'])} "
              f"{_fmt(r['p1_plus_p2_ms'])} {_fmt(r['overhead_ms'])} "
              f"{ovh_pct:>5.1f}% {r['adaptive_vs_sglang']:>8.3f}×")
    print(sep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)

    # Sweep: small/medium/large bs × pages × K matrix exercising both
    # the light path (K=30) and heavy path (K=2048).
    configs = [
        # bs, pages, K
        (1,  4096,   30),
        (1, 16384,   30),
        (1, 32768,   30),
        (4,  4096,   30),
        (4, 16384,   30),
        (4, 32768,   30),
        (16, 4096,   30),
        (16, 32768,  30),
        # heavy
        (1,  4096, 2048),
        (1, 16384, 2048),
        (1, 32768, 2048),
        (4,  4096, 2048),
        (4, 16384, 2048),
        (4, 32768, 2048),
        (16, 4096, 2048),
        (16, 32768, 2048),
    ]

    rows = []
    for (bs, pages, K) in configs:
        try:
            row = run_config(bs, pages, K, warmup=args.warmup, repeat=args.repeat)
            rows.append(row)
            print(f"[done] bs={bs} pages={pages} K={K}")
        except RuntimeError as e:
            print(f"[skip] bs={bs} pages={pages} K={K}: {e}")

    print_table(rows)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
