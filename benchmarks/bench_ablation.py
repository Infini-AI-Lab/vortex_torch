"""Phase + merge ablation for the K=30 random-split parallel kernel.

Splits production latency into per-phase pieces and compares merge variants
on identical pre-filled workspaces. The fixture lives in
csrc/topk_adaptive_profile.cu (NOT in topk_sglang_merge.cu); the production
kernel still uses the SPLITS-specialised merge described in
csrc/topk_sglang_merge.cu's file header.

Ablation modes (must match the kAblMode_* constants in topk_adaptive_profile.cu):

  0  full_parallel             (re-enters the production workspace API)
  1  local_only                (Stage 1 sort + workspace write only)
  2  local_no_workspace        (Stage 1 sort, scratch sink — no ws write)
  3  workspace_write_only      (write 32 dummy entries / split)
  4  atomic_only               (done_counter atomic + last-CTA test only)
  5  merge_prod_default        (legacy per-SPLITS dispatch: 2-way/pairwise/k-way)
  6  merge_only_cub_warp       (cub::WarpMergeSort — current production merge)
  7  merge_only_cub_block      (cub::BlockMergeSort benchmark)
  8  memset_only               (host cudaMemsetAsync of done_counter)
  9  merge_only_2way_manual    (SPLITS=2 only)
  10 merge_only_pairwise_tree_4(SPLITS=4 only)
  11 merge_kway_all            (force k-way for all SPLITS)

Benchmark matrix (default; override on the CLI):
  B           ∈ {1, 2, 4, 8, 16, 32, 128}
  pages       ∈ {8192, 16384, 32768}
  topk_val    = 30
  partition   = contiguous
  forced_splits ∈ {2, 4, 8, 16, 32}

Outputs `bench_results/k30_ablation.csv` (long-form per-row records) and a
wide table `…_summary.csv` with the columns the spec asks for:
  B, pages, split, merge_mode, full_adaptive_us, local_only_us,
  workspace_write_us, atomic_only_us, merge_only_us, fused_us,
  speedup_vs_fused.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import vortex_torch_C as C


MODES = [
    (0,  "full_parallel"),
    (1,  "local_only"),
    (2,  "local_no_workspace"),
    (3,  "workspace_write_only"),
    (4,  "atomic_only"),
    (5,  "merge_prod_default"),            # legacy: 2-way/pairwise/k-way per SPLITS
    (6,  "merge_only_cub_warp"),          # current production (WarpMergeSort)
    (7,  "merge_only_cub_block"),
    (8,  "memset_only"),
    (9,  "merge_only_2way_manual"),       # SPLITS=2 only
    (10, "merge_only_pairwise_tree_4"),   # SPLITS=4 only
    (11, "merge_kway_all"),               # force k-way for all SPLITS
]


# ---------- input setup -------------------------------------------------------

def make_inputs(eff_bs: int, pages: int, topk_val: int = 30,
                bos: int = 0, eos: int = 0, seed: int = 0,
                dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(eff_bs * pages, dtype=dtype, device=device)
    dense_kv_indptr = torch.arange(eff_bs + 1, dtype=torch.int32, device=device) * pages
    dense_kv_indices = torch.arange(eff_bs * pages, dtype=torch.int32, device=device)
    out_per_row = bos + eos + topk_val
    sparse_kv_indptr = torch.arange(eff_bs + 1, dtype=torch.int32, device=device) * out_per_row
    sparse_kv_indices = torch.full((eff_bs * out_per_row,), -1,
                                   dtype=torch.int32, device=device)
    return {
        "x": x,
        "dense_kv_indptr": dense_kv_indptr,
        "sparse_kv_indptr": sparse_kv_indptr,
        "dense_kv_indices": dense_kv_indices,
        "sparse_kv_indices": sparse_kv_indices,
    }


def make_workspace(eff_bs: int):
    opts = dict(dtype=torch.int32, device="cuda")
    n = eff_bs * 32 * 32  # max splits=32, local_k=32
    return {
        "partial_keys":    torch.empty(n, **opts),
        "partial_indices": torch.empty(n, **opts),
        "done_counter":    torch.empty(eff_bs, **opts),
        "scratch":         torch.empty(eff_bs * 32, **opts),
    }


def fill_workspace_for_merge(ws, eff_bs, splits, seed=1):
    """Pre-fill partial_keys/indices with sorted top-32 lists per split.

    Production layout: `[B, SPLITS, 32]` flattened to a 1-D int32 tensor.
    Each (b, split) slot is sorted descending by uint32 key. Indices are
    distinct global page IDs (no -1 sentinels in the prefilled portion).
    """
    torch.manual_seed(seed)
    n = eff_bs * splits * 32
    keys_base = torch.randint(0, 2**31 - 1, (eff_bs * splits, 32),
                              dtype=torch.int64, device="cuda").to(torch.int32)
    keys_sorted = keys_base.sort(dim=1, descending=True).values
    ws["partial_keys"][:n] = keys_sorted.flatten()
    indices = torch.arange(n, dtype=torch.int32, device="cuda")
    ws["partial_indices"][:n] = indices


# ---------- kernel calls ------------------------------------------------------

def call_ablation(inputs, ws, eff_bs, pages, topk_val, mode, splits,
                  bos=0, eos=0):
    inputs["sparse_kv_indices"].fill_(-1)
    C.topk_output_adaptive_workspace_ablation(
        inputs["x"], inputs["dense_kv_indptr"], inputs["sparse_kv_indptr"],
        inputs["dense_kv_indices"], inputs["sparse_kv_indices"],
        ws["partial_keys"], ws["partial_indices"],
        ws["done_counter"], ws["scratch"],
        eff_bs, topk_val, bos, eos, pages,
        mode, splits,
    )


def call_fused(inputs, eff_bs, pages, topk_val, bos=0, eos=0):
    inputs["sparse_kv_indices"].fill_(-1)
    C.topk_output_sglang_fused(
        inputs["x"], inputs["dense_kv_indptr"], inputs["sparse_kv_indptr"],
        inputs["dense_kv_indices"], inputs["sparse_kv_indices"],
        eff_bs, topk_val, bos, eos, pages, 0, 0.0, None, None,
    )


def bench(fn, *args, warmup=20, repeat=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    samples.sort()
    return {
        "mean": statistics.mean(samples),
        "p50":  samples[len(samples) // 2],
        "p90":  samples[int(len(samples) * 0.9)],
        "min":  samples[0],
        "max":  samples[-1],
    }


# ---------- correctness check -------------------------------------------------

def verify_merge_only(ws, eff_bs, splits, topk_val, bos=0):
    """Check that the merge_only_prod_default kernel returns the true top-K.

    Builds a reference by reading partial_keys/indices into Python, picking
    the largest topk_val keys per row, and comparing against the kernel's
    output as a SET (production merge order is unspecified for ties).
    """
    inputs = make_inputs(eff_bs, pages=8192, topk_val=topk_val, bos=bos)
    fill_workspace_for_merge(ws, eff_bs, splits, seed=42)
    inputs["sparse_kv_indices"].fill_(-1)
    C.topk_output_adaptive_workspace_ablation(
        inputs["x"], inputs["dense_kv_indptr"], inputs["sparse_kv_indptr"],
        inputs["dense_kv_indices"], inputs["sparse_kv_indices"],
        ws["partial_keys"], ws["partial_indices"],
        ws["done_counter"], ws["scratch"],
        eff_bs, topk_val, bos, 0, 8192,
        11, splits,   # mode 11 = prod_default
    )
    torch.cuda.synchronize()

    # Reference top-K from the prefilled workspace.
    n = eff_bs * splits * 32
    keys = ws["partial_keys"][:n].view(eff_bs, splits * 32).to(torch.int64) & 0xFFFFFFFF
    idx  = ws["partial_indices"][:n].view(eff_bs, splits * 32)
    out_per_row = bos + topk_val
    out = inputs["sparse_kv_indices"]

    failures = 0
    for b in range(eff_bs):
        ref_topk = keys[b].topk(topk_val).indices  # local positions
        ref_set = set(idx[b, ref_topk].tolist())
        got = out[b * out_per_row + bos : b * out_per_row + bos + topk_val]
        got_set = set(got.tolist()) - {-1}
        if ref_set != got_set:
            failures += 1
            if failures <= 3:
                print(f"    MERGE CORRECTNESS FAIL b={b} splits={splits} K={topk_val}: "
                      f"|sym_diff|={len(ref_set ^ got_set)}")
    return failures == 0


# ---------- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results/k30_ablation.csv")
    ap.add_argument("--summary-out", default="bench_results/k30_ablation_summary.csv")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--pages", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 128])
    ap.add_argument("--splits", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- correctness gate first --
    if not args.skip_correctness:
        print("=== Merge correctness check (mode=11 merge_kway_all) ===")
        ws_check = make_workspace(max(args.bs))
        all_ok = True
        for splits in (2, 4, 8, 16, 32):
            for K in (1, 4, 8, 16, 30, 32):
                ok = verify_merge_only(ws_check, eff_bs=4, splits=splits, topk_val=K)
                tag = "OK  " if ok else "FAIL"
                print(f"  splits={splits:2d} K={K:2d} : {tag}")
                all_ok &= ok
            # reserved_bos cover
            ok = verify_merge_only(ws_check, eff_bs=4, splits=splits, topk_val=30, bos=2)
            print(f"  splits={splits:2d} K=30 bos=2: {'OK  ' if ok else 'FAIL'}")
            all_ok &= ok
        if not all_ok:
            print("CORRECTNESS FAILURES — aborting bench")
            return 1
        print("All merge-only correctness checks passed.\n")

    long_rows = []
    # cell -> {ablation_name: mean_ms}
    cells = defaultdict(dict)

    for pages in args.pages:
        for B in args.bs:
            inputs = make_inputs(B, pages)
            ws = make_workspace(B)

            # Reference: fused.
            call_fused(inputs, B, pages, 30)
            torch.cuda.synchronize()
            s_fused = bench(call_fused, inputs, B, pages, 30,
                            warmup=args.warmup, repeat=args.repeat)
            long_rows.append({
                "pages": pages, "B": B, "splits": 0,
                "ablation": "fused_baseline",
                **{k: f"{v:.4f}" for k, v in s_fused.items()},
            })
            print(f"\n=== pages={pages} B={B} ===  fused = {s_fused['mean']*1000:.2f} us")

            for splits in args.splits:
                # Pre-fill workspace ahead of the merge-only modes.
                fill_workspace_for_merge(ws, B, splits)
                torch.cuda.synchronize()

                for mode_id, mode_name in MODES:
                    if mode_id == 9 and splits != 2: continue
                    if mode_id == 10 and splits != 4: continue
                    # Re-prefill before merge-only calls so input layout is fresh.
                    if mode_id in (5, 6, 7, 9, 10, 11):
                        fill_workspace_for_merge(ws, B, splits)
                        torch.cuda.synchronize()
                    try:
                        call_ablation(inputs, ws, B, pages, 30, mode_id, splits)
                        torch.cuda.synchronize()
                    except RuntimeError as e:
                        print(f"  split={splits} {mode_name}: SKIP ({e})")
                        continue
                    stats = bench(call_ablation, inputs, ws, B, pages, 30,
                                  mode_id, splits,
                                  warmup=args.warmup, repeat=args.repeat)
                    long_rows.append({
                        "pages": pages, "B": B, "splits": splits,
                        "ablation": mode_name,
                        **{k: f"{v:.4f}" for k, v in stats.items()},
                    })
                    cells[(pages, B, splits)][mode_name] = stats["mean"]
                    cells[(pages, B, splits)]["__fused"] = s_fused["mean"]
                    pct = stats["mean"] / s_fused["mean"] * 100
                    print(f"  split={splits:2d} {mode_name:<28s}"
                          f" mean={stats['mean']*1000:7.2f} us"
                          f"  ({pct:5.1f}% of fused)")

    # ---------- write long-form CSV ----------
    long_cols = ["pages", "B", "splits", "ablation", "mean", "p50", "p90", "min", "max"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(long_rows)
    print(f"\nlong-form rows → {out_path}  ({len(long_rows)} rows)")

    # ---------- write spec-shaped summary ----------
    summary_path = Path(args.summary_out)
    summary_cols = ["B", "pages", "split", "merge_mode",
                    "full_adaptive_us", "local_only_us", "workspace_write_us",
                    "atomic_only_us", "merge_only_us", "fused_us",
                    "speedup_vs_fused"]

    def _us(ms):
        return f"{ms * 1000:.2f}" if ms is not None else ""

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary_cols)
        for (pages, B, splits), data in sorted(cells.items()):
            full     = data.get("full_parallel")
            local    = data.get("local_only")
            ws_write = data.get("workspace_write_only")
            atomic   = data.get("atomic_only")
            fused    = data.get("__fused")
            for merge_name in ("merge_prod_default", "merge_only_cub_warp",
                               "merge_only_cub_block", "merge_kway_all",
                               "merge_only_2way_manual",
                               "merge_only_pairwise_tree_4"):
                merge_t = data.get(merge_name)
                if merge_t is None: continue
                speedup = (fused / full) if (full and fused) else float("nan")
                w.writerow([B, pages, splits, merge_name,
                            _us(full), _us(local), _us(ws_write),
                            _us(atomic), _us(merge_t), _us(fused),
                            f"{speedup:.3f}" if speedup == speedup else ""])
    print(f"summary table → {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
