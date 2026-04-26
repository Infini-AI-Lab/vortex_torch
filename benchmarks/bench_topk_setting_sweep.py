#!/usr/bin/env python
"""Comprehensive (pages, K, batch, split, mapping, dtype) latency sweep
comparing the three TopK kernels in this repo:

  topk_sglang_merge.cu  -> topk_output_adaptive_workspace  (adaptive split path)
  topk_sglang.cu        -> topk_output_sglang_fused        (fused two-stage radix)
  topk.cu               -> topk_output                     (CUB BlockRadixSort full sort)

Outputs four files under <output-dir>:
  topk_setting_sweep_raw.csv               long-form, one row per measurement
  topk_setting_sweep_best_adaptive.csv     best adaptive split per (pages,K,B,mapping,dtype)
  topk_parallel_advantage_summary.csv      win/loss region rollup
  topk_setting_sweep_report.md             human-readable analysis

See module-level docstring of topk_sglang_merge.cu for the dispatcher
contract this script mirrors when labeling actual_path.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import vortex_torch_C as V

# --------------------------------------------------------------------------- #
# Mapping mode constants — must match csrc/topk_mapping.cuh.
# --------------------------------------------------------------------------- #
MAPPING_NONE   = 0
MAPPING_POWER  = 3
MAPPING_LOG    = 4
MAPPING_ASINH  = 6
MAPPING_LOG1P  = 7
MAPPING_TRUNC8 = 8
MAPPING_ERF    = 9
MAPPING_TANH   = 10

MAPPING_NAMES = {
    MAPPING_NONE:   "NONE",
    MAPPING_POWER:  "POWER",
    MAPPING_LOG:    "LOG",
    MAPPING_ASINH:  "ASINH",
    MAPPING_LOG1P:  "LOG1P",
    MAPPING_TRUNC8: "TRUNC8",
    MAPPING_ERF:    "ERF",
    MAPPING_TANH:   "TANH",
}
MAPPING_BY_NAME = {v: k for k, v in MAPPING_NAMES.items()}

# Mirrors the dispatcher in csrc/topk_sglang_merge.cu.
K_MAX_ADAPTIVE   = 32     # K <= 32 stays on the adaptive K=30 path
K_FUSED_FALLBACK = 1024   # K >= 1024 routes to fused, even from adaptive entry

LOCAL_BLOCK_FULL_SORT = 0
LOCAL_SELECT32_SORT32 = 1

# topk.cu template ladder caps at 8192 pages.
TOPK_CU_MAX_PAGES = 8192

# kCfg* capacity table from csrc/topk_sglang_merge.cu (BLOCK_FULL_SORT only).
BLOCK_FULL_SORT_CAPACITY = {1: 8192, 2: 8192, 4: 4096, 8: 4096, 16: 2048, 32: 1024}

DEFAULT_PAGES   = [4096, 8192, 16384, 32768, 65536]
DEFAULT_KS      = [30, 64, 128, 256, 512, 1024, 2048]
DEFAULT_BATCHES = [1, 2, 4, 8, 16]
DEFAULT_SPLITS  = [1, 2, 4, 8, 16, 32]
DEFAULT_MAPPING_NAMES = ["NONE"]
DEFAULT_DTYPES  = ["bfloat16"]

WIN_THRESHOLD = 1.03  # adaptive "wins" if speedup_vs_sglang >= this

# Production merge mode wired into TopK30_RandomSplit_Select32_Kernel /
# TopK30_RandomSplit_Parallel_Kernel — cub::WarpMergeSort. The merge-only
# ablation sub-sweep timings are written separately to topk_merge_mode_summary.csv.
PROD_MERGE_NAME = "warp_cub"
LOCAL_MODE_NAMES = {LOCAL_BLOCK_FULL_SORT: "BLOCK_FULL_SORT",
                    LOCAL_SELECT32_SORT32: "SELECT32_SORT32"}

# Ablation mode IDs from topk_adaptive_profile.cu.
ABL_LOCAL_WITH_WORKSPACE = 1   # populates partial workspace, no merge
ABL_MERGE_PROD_DEFAULT   = 5
ABL_MERGE_CUB_WARP       = 6
ABL_MERGE_CUB_BLOCK      = 7
ABL_MERGE_KWAY           = 11
MERGE_ABL_NAMES = {
    ABL_MERGE_PROD_DEFAULT: "prod_default(legacy)",
    ABL_MERGE_CUB_WARP:     "warp_cub",
    ABL_MERGE_CUB_BLOCK:    "block_cub",
    ABL_MERGE_KWAY:         "kway",
}

# --------------------------------------------------------------------------- #

def _dtype_str_to_torch(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float": torch.float32, "float32": torch.float32}[s]


def apply_remap_torch(x: torch.Tensor, mode: int, p: float) -> torch.Tensor:
    """Reference-side remap, kept in sync with apply_transform_tmpl in topk_mapping.cuh.

    Only modes used by this sweep are implemented. Adding more requires editing
    topk_mapping.cuh and propagating to this function.
    """
    if mode in (MAPPING_NONE, MAPPING_TRUNC8):
        return x
    if mode == MAPPING_POWER:
        return torch.copysign(torch.abs(x).pow(p), x)
    if mode == MAPPING_LOG:
        return torch.copysign(torch.log(torch.abs(x) + 1.0), x)
    if mode == MAPPING_ASINH:
        return torch.asinh(p * x)
    if mode == MAPPING_LOG1P:
        return torch.copysign(torch.log1p(p * torch.abs(x)), x)
    if mode == MAPPING_ERF:
        return torch.erf(p * x)
    if mode == MAPPING_TANH:
        return torch.tanh(p * x)
    raise ValueError(f"reference remap not implemented for mapping_mode={mode}")


# --------------------------------------------------------------------------- #
# Tensor / workspace setup.
# --------------------------------------------------------------------------- #
@dataclass
class Inputs:
    scores: torch.Tensor
    dense_kv_indptr: torch.Tensor
    sparse_kv_indptr: torch.Tensor
    dense_kv_indices: torch.Tensor
    out: torch.Tensor          # int32 sparse_kv_indices
    B: int
    pages: int
    K: int
    reserved_bos: int
    reserved_eos: int


def make_inputs(B: int, pages: int, K: int, dtype: torch.dtype,
                reserved_bos: int = 1, reserved_eos: int = 2,
                seed: int = 0) -> Inputs:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dense_kv_indptr  = torch.arange(B + 1, device=device, dtype=torch.int32) * pages
    sparse_kv_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * (K + reserved_bos + reserved_eos)
    total = B * pages
    scores = torch.randn(total, device=device, dtype=dtype)
    dense_kv_indices = torch.arange(total, device=device, dtype=torch.int32)
    out = torch.full((B * (K + reserved_bos + reserved_eos),), -1, device=device, dtype=torch.int32)
    return Inputs(scores, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, out,
                  B, pages, K, reserved_bos, reserved_eos)


def make_workspace(B_max: int, max_split: int = 32, K_local: int = 32):
    device = torch.device("cuda")
    ws_elems = max(B_max * max_split * K_local, 64)
    return dict(
        partial_keys    = torch.zeros(ws_elems,        device=device, dtype=torch.int32),
        partial_indices = torch.zeros(ws_elems,        device=device, dtype=torch.int32),
        done_counter    = torch.zeros(max(B_max, 1),   device=device, dtype=torch.int32),
    )


# --------------------------------------------------------------------------- #
# Reference top-K and correctness.
# --------------------------------------------------------------------------- #
def reference_topk(inp: Inputs, mapping_mode: int, mapping_power: float):
    """Returns (ref_sets[B], ref_remapped[B] (cpu fp32), threshold_per_row[B])."""
    ref_sets = []
    ref_remapped = []
    thresholds = []
    for b in range(inp.B):
        row = inp.scores[b * inp.pages + inp.reserved_bos
                         : (b + 1) * inp.pages - inp.reserved_eos].float()
        remapped = apply_remap_torch(row, mapping_mode, mapping_power)
        vals, idx_within = torch.topk(remapped, inp.K)
        global_idx = (idx_within + b * inp.pages + inp.reserved_bos).cpu().tolist()
        ref_sets.append(set(global_idx))
        ref_remapped.append(remapped.cpu())
        thresholds.append(vals.min().item())
    return ref_sets, ref_remapped, thresholds


def check_correctness(inp: Inputs, ref_sets, ref_remapped, thresholds,
                      mapping_mode: int, mapping_power: float) -> Tuple[bool, str]:
    """Set equality with tie tolerance.

    Returns (ok, note). On failure, `note` describes the failure.
    """
    out = inp.out.cpu()
    for b in range(inp.B):
        slot_start = b * (inp.K + inp.reserved_bos + inp.reserved_eos) + inp.reserved_bos
        out_row = out[slot_start : slot_start + inp.K].tolist()
        out_set = set(out_row)
        if -1 in out_set:
            return False, f"row {b}: -1 in output (count={out_row.count(-1)})"
        if out_set == ref_sets[b]:
            continue
        # Tie tolerance: every kernel-selected score must reach the threshold,
        # within fp tolerance.
        row_offset = b * inp.pages
        out_within = [g - row_offset - inp.reserved_bos for g in out_set]
        npages_eff = inp.pages - inp.reserved_bos - inp.reserved_eos
        if any(i < 0 or i >= npages_eff for i in out_within):
            return False, f"row {b}: out-of-range global idx"
        out_scores = ref_remapped[b][out_within]
        thresh = thresholds[b]
        tol = max(1e-6, 1e-3 * abs(thresh))
        min_out = out_scores.min().item()
        if min_out < thresh - tol:
            return False, (f"row {b}: min selected score={min_out:.4f} < "
                           f"K-th ref score={thresh:.4f} (tol={tol:.2e})")
    return True, ""


# --------------------------------------------------------------------------- #
# Timing.
# --------------------------------------------------------------------------- #
def time_kernel_us(fn, warmup: int, repeat: int) -> Optional[Dict[str, float]]:
    """Per-call event timing. Returns dict with mean/p50/p90/min/max/std (us)
    or None if the kernel raised."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return None
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    try:
        for i in range(repeat):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
    except Exception:
        return None
    times_us = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(repeat))
    n = len(times_us)
    mean = sum(times_us) / n
    var  = sum((t - mean) ** 2 for t in times_us) / n
    return dict(
        mean=mean,
        p50=times_us[n // 2],
        p90=times_us[min(n - 1, int(round(n * 0.9)))],
        min=times_us[0],
        max=times_us[-1],
        std=math.sqrt(var),
    )


# --------------------------------------------------------------------------- #
# Method launchers.
# --------------------------------------------------------------------------- #
def call_fused(inp: Inputs, mapping_mode: int, mapping_power: float):
    # Caller is responsible for inp.out.fill_(-1) BEFORE the timed loop if it
    # cares about a clean baseline; the fill is its own kernel launch and would
    # otherwise pollute kernel-only timing.
    V.topk_output_sglang_fused(
        inp.scores, inp.dense_kv_indptr, inp.sparse_kv_indptr, inp.dense_kv_indices,
        inp.out, inp.B, inp.K, inp.reserved_bos, inp.reserved_eos, inp.pages,
        mapping_mode, mapping_power, None, None,
    )


def call_topk_cu(inp: Inputs):
    # NOTE: arg order differs from sglang variants -
    #   (x, dense_kv_indptr, dense_kv_indices, sparse_kv_indptr, ...)
    V.topk_output(
        inp.scores, inp.dense_kv_indptr, inp.dense_kv_indices, inp.sparse_kv_indptr,
        inp.out, inp.B, inp.K, inp.reserved_bos, inp.reserved_eos, inp.pages,
    )


def call_adaptive(inp: Inputs, ws: dict, mapping_mode: int, mapping_power: float,
                  forced_split: int, forced_partition: int, local_mode: int):
    V.topk_output_adaptive_workspace(
        inp.scores, inp.dense_kv_indptr, inp.sparse_kv_indptr, inp.dense_kv_indices,
        inp.out, ws["partial_keys"], ws["partial_indices"], ws["done_counter"],
        inp.B, inp.K, inp.reserved_bos, inp.reserved_eos, inp.pages,
        mapping_mode, mapping_power,
        forced_split, forced_partition, local_mode,
    )


# --------------------------------------------------------------------------- #
# actual_path classification — mirrors the C++ dispatcher contract.
# --------------------------------------------------------------------------- #
def classify_adaptive_actual_path(K: int, split: int, pages: int,
                                  local_mode: int) -> Tuple[str, Optional[int], bool]:
    """Returns (actual_path, actual_split, is_supported).

    actual_split is the effective split count (None for fused fallback).
    is_supported is False when the call would TORCH_CHECK fail.
    """
    if K >= K_FUSED_FALLBACK:
        return ("fused_fallback_large_k", None, True)
    if K > K_MAX_ADAPTIVE:
        return ("fused_fallback_mid_k", None, True)
    if local_mode == LOCAL_BLOCK_FULL_SORT:
        chunk_max = (pages + split - 1) // split
        cap = BLOCK_FULL_SORT_CAPACITY.get(split, 0)
        if cap < chunk_max:
            return ("unsupported_capacity", split, False)
        return ("adaptive_block_full_sort", split, True)
    return ("adaptive_select32_sort32", split, True)


# --------------------------------------------------------------------------- #
# Sweep driver.
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    device_name: str
    sm_count: int
    dtype: str
    mapping_mode: int
    mapping_name: str
    mapping_power: float
    pages: int
    topk: int
    batch: int
    method: str                # "topk_sglang_fused", "topk_cu", "adaptive_merge"
    requested_split: Optional[int]
    actual_split: Optional[int]
    local_mode: str            # "BLOCK_FULL_SORT" / "SELECT32_SORT32" / "n/a"
    merge_mode: str            # "warp_cub" (production); "n/a" if no merge
    candidate_count: Optional[int]   # split * local_k; None if no merge
    actual_path: str
    mean_us: Optional[float]
    p50_us: Optional[float]
    p90_us: Optional[float]
    min_us: Optional[float]
    max_us: Optional[float]
    std_us: Optional[float]
    correctness: Optional[bool]
    speedup_vs_sglang_fused: Optional[float]
    speedup_vs_topk_cu: Optional[float]
    notes: str


def run_one_setting(pages: int, K: int, B: int, mapping_mode: int, mapping_power: float,
                    dtype_str: str, splits: List[int], local_mode: int,
                    warmup: int, repeat: int, ws: dict,
                    device_name: str, sm_count: int) -> List[Row]:
    """Bench every method at one (pages, K, B, mapping, dtype) cell."""
    rows: List[Row] = []
    inp = make_inputs(B, pages, K, _dtype_str_to_torch(dtype_str))
    ref_sets, ref_remapped, thresholds = reference_topk(inp, mapping_mode, mapping_power)
    map_name = MAPPING_NAMES.get(mapping_mode, str(mapping_mode))

    def _row(**kw):
        defaults = dict(
            device_name=device_name, sm_count=sm_count, dtype=dtype_str,
            mapping_mode=mapping_mode, mapping_name=map_name, mapping_power=mapping_power,
            pages=pages, topk=K, batch=B,
            requested_split=None, actual_split=None,
            local_mode="n/a", merge_mode="n/a", candidate_count=None,
            mean_us=None, p50_us=None, p90_us=None, min_us=None, max_us=None, std_us=None,
            correctness=None, speedup_vs_sglang_fused=None, speedup_vs_topk_cu=None,
            notes="",
        )
        defaults.update(kw)
        return Row(**defaults)

    # ---------- topk_sglang.cu fused baseline ----------
    fused_us: Optional[float] = None
    try:
        inp.out.fill_(-1)
        call_fused(inp, mapping_mode, mapping_power)
        torch.cuda.synchronize()
        ok, note = check_correctness(inp, ref_sets, ref_remapped, thresholds,
                                     mapping_mode, mapping_power)
    except RuntimeError as e:
        # Most common: pages > fused dynamic-smem ceiling (~96KB → ~96k pages).
        msg = str(e)
        path = "fused_unavailable_smem" if "exceeds" in msg or "smem" in msg.lower() else "error"
        rows.append(_row(method="topk_sglang_fused", actual_path=path,
                         correctness=False, notes=f"raised: {msg[:160]}"))
    except Exception as e:
        rows.append(_row(method="topk_sglang_fused", actual_path="error",
                         correctness=False, notes=f"raised: {e}"))
    else:
        t = time_kernel_us(lambda: call_fused(inp, mapping_mode, mapping_power),
                           warmup, repeat)
        if t is None:
            rows.append(_row(method="topk_sglang_fused", actual_path="error",
                             correctness=ok, notes="time_kernel_us returned None"))
        else:
            fused_us = t["mean"]
            rows.append(_row(
                method="topk_sglang_fused", actual_path="fused",
                mean_us=t["mean"], p50_us=t["p50"], p90_us=t["p90"],
                min_us=t["min"], max_us=t["max"], std_us=t["std"],
                correctness=ok,
                speedup_vs_sglang_fused=1.0,
                notes=note,
            ))

    # ---------- topk.cu baseline (CUB full sort) ----------
    cub_us: Optional[float] = None
    if pages > TOPK_CU_MAX_PAGES:
        rows.append(_row(method="topk_cu", actual_path="topk_cu_unsupported",
                         notes=f"pages={pages} > template ladder cap {TOPK_CU_MAX_PAGES}"))
    else:
        try:
            inp.out.fill_(-1)
            call_topk_cu(inp)
            torch.cuda.synchronize()
            # topk.cu doesn't apply remap, so its output is for raw scores —
            # ALWAYS check against the unmapped reference for fairness.
            if mapping_mode in (MAPPING_NONE, MAPPING_TRUNC8):
                ok, note = check_correctness(inp, ref_sets, ref_remapped, thresholds,
                                             mapping_mode, mapping_power)
            else:
                ok, note = True, "remap unsupported by topk.cu; correctness skipped"
        except Exception as e:
            rows.append(_row(method="topk_cu", actual_path="error",
                             correctness=False, notes=f"raised: {e}"))
        else:
            t = time_kernel_us(lambda: call_topk_cu(inp), warmup, repeat)
            if t is None:
                rows.append(_row(method="topk_cu", actual_path="error",
                                 correctness=ok, notes="time_kernel_us returned None"))
            else:
                cub_us = t["mean"]
                rows.append(_row(
                    method="topk_cu", actual_path="cub_full_sort",
                    mean_us=t["mean"], p50_us=t["p50"], p90_us=t["p90"],
                    min_us=t["min"], max_us=t["max"], std_us=t["std"],
                    correctness=ok,
                    speedup_vs_sglang_fused=(fused_us / t["mean"]) if fused_us else None,
                    speedup_vs_topk_cu=1.0,
                    notes=note,
                ))

    # ---------- topk_sglang_merge.cu adaptive (one row per requested split) ----------
    local_mode_str = LOCAL_MODE_NAMES.get(local_mode, "unknown")
    for split in splits:
        actual_path, actual_split, is_supported = classify_adaptive_actual_path(
            K, split, pages, local_mode)
        # Adaptive paths use cub::WarpMergeSort over (split * 32) candidates;
        # split=1 has no merge stage at all.
        on_adaptive_path = actual_path.startswith("adaptive_")
        merge_mode = PROD_MERGE_NAME if (on_adaptive_path and split > 1) else "n/a"
        candidate_count = (split * 32) if (on_adaptive_path and split > 1) else None
        local_mode_for_row = local_mode_str if on_adaptive_path else "n/a"

        if not is_supported:
            rows.append(_row(
                method="adaptive_merge", requested_split=split, actual_split=actual_split,
                local_mode=local_mode_for_row, merge_mode=merge_mode,
                candidate_count=candidate_count, actual_path=actual_path,
                notes=(f"BLOCK_FULL_SORT cap={BLOCK_FULL_SORT_CAPACITY.get(split,0)} "
                       f"< chunk_max={(pages + split - 1)//split}"),
            ))
            continue

        try:
            inp.out.fill_(-1)
            call_adaptive(inp, ws, mapping_mode, mapping_power,
                          forced_split=split, forced_partition=1,  # CONTIGUOUS
                          local_mode=local_mode)
            torch.cuda.synchronize()
            ok, note = check_correctness(inp, ref_sets, ref_remapped, thresholds,
                                         mapping_mode, mapping_power)
        except RuntimeError as e:
            # K>32 and pages too large for fused fallback's smem.
            msg = str(e)
            err_path = ("fused_fallback_unavailable_smem"
                        if (not on_adaptive_path and ("exceeds" in msg or "smem" in msg.lower()))
                        else actual_path + "_error")
            rows.append(_row(method="adaptive_merge",
                             requested_split=split, actual_split=actual_split,
                             local_mode=local_mode_for_row, merge_mode=merge_mode,
                             candidate_count=candidate_count,
                             actual_path=err_path,
                             correctness=False, notes=f"raised: {msg[:160]}"))
            continue
        except Exception as e:
            rows.append(_row(method="adaptive_merge",
                             requested_split=split, actual_split=actual_split,
                             local_mode=local_mode_for_row, merge_mode=merge_mode,
                             candidate_count=candidate_count, actual_path=actual_path,
                             correctness=False, notes=f"raised: {e}"))
            continue

        # For fused-fallback paths, all forced_split values produce identical
        # timings (same fused kernel called); we still time each entry to
        # quantify dispatcher overhead.
        t = time_kernel_us(
            lambda: call_adaptive(inp, ws, mapping_mode, mapping_power,
                                   forced_split=split, forced_partition=1,
                                   local_mode=local_mode),
            warmup, repeat,
        )
        if t is None:
            rows.append(_row(method="adaptive_merge",
                             requested_split=split, actual_split=actual_split,
                             local_mode=local_mode_for_row, merge_mode=merge_mode,
                             candidate_count=candidate_count, actual_path=actual_path,
                             correctness=ok, notes="time_kernel_us returned None"))
            continue

        rows.append(_row(
            method="adaptive_merge",
            requested_split=split, actual_split=actual_split,
            local_mode=local_mode_for_row, merge_mode=merge_mode,
            candidate_count=candidate_count, actual_path=actual_path,
            mean_us=t["mean"], p50_us=t["p50"], p90_us=t["p90"],
            min_us=t["min"], max_us=t["max"], std_us=t["std"],
            correctness=ok,
            speedup_vs_sglang_fused=(fused_us / t["mean"]) if fused_us else None,
            speedup_vs_topk_cu=(cub_us / t["mean"]) if cub_us else None,
            notes=note,
        ))

    return rows


# --------------------------------------------------------------------------- #
# Merge-mode ablation (K=30 only — the ablation kernels in
# topk_adaptive_profile.cu are hardcoded to kLocalK_Top30 = 32).
# --------------------------------------------------------------------------- #
def call_ablation(inp: Inputs, ws: dict, scratch: torch.Tensor,
                  ablation_mode: int, forced_split: int):
    V.topk_output_adaptive_workspace_ablation(
        inp.scores, inp.dense_kv_indptr, inp.sparse_kv_indptr, inp.dense_kv_indices,
        inp.out, ws["partial_keys"], ws["partial_indices"], ws["done_counter"],
        scratch,
        inp.B, inp.K, inp.reserved_bos, inp.reserved_eos, inp.pages,
        ablation_mode, forced_split,
    )


def run_merge_ablation(pages_list, batches, splits, warmup, repeat,
                       ws, device_name, sm_count) -> List[dict]:
    """Per (pages, B, split, merge_mode), measure merge-only latency.

    Workflow per cell:
      1. Populate the workspace via ablation_mode = LocalWithWorkspace (mode 1).
      2. For each merge variant, time merge-only kernel (modes 5/6/7/11).

    Returns list of dicts (CSV-ready)."""
    rows = []
    device = torch.device("cuda")
    scratch = torch.zeros(max(1, max(batches) * max(splits)),
                           device=device, dtype=torch.int32)
    K = 30  # ablation harness is K<=32 only
    for pages in pages_list:
        for B in batches:
            inp = make_inputs(B, pages, K, torch.bfloat16)
            for split in splits:
                if split <= 1:
                    continue  # nothing to merge
                # Step 1: populate the workspace (ablation_mode=1).
                try:
                    call_ablation(inp, ws, scratch,
                                  ABL_LOCAL_WITH_WORKSPACE, split)
                    torch.cuda.synchronize()
                except Exception as e:
                    rows.append(dict(pages=pages, batch=B, split=split,
                                     merge_mode="setup_failed",
                                     mean_us=None, notes=f"populate raised: {e}"))
                    continue
                # Step 2: merge variants. Some require specific splits.
                variants = [ABL_MERGE_PROD_DEFAULT, ABL_MERGE_CUB_WARP,
                            ABL_MERGE_CUB_BLOCK, ABL_MERGE_KWAY]
                for ablv in variants:
                    name = MERGE_ABL_NAMES[ablv]
                    try:
                        call_ablation(inp, ws, scratch, ablv, split)
                        torch.cuda.synchronize()
                    except Exception as e:
                        rows.append(dict(pages=pages, batch=B, split=split,
                                         merge_mode=name, candidate_count=split * 32,
                                         mean_us=None,
                                         notes=f"raised: {repr(e)[:120]}"))
                        continue
                    t = time_kernel_us(
                        lambda av=ablv, sp=split: call_ablation(inp, ws, scratch, av, sp),
                        warmup, repeat,
                    )
                    if t is None:
                        rows.append(dict(pages=pages, batch=B, split=split,
                                         merge_mode=name, candidate_count=split * 32,
                                         mean_us=None, notes="time_kernel_us failed"))
                        continue
                    rows.append(dict(
                        device_name=device_name, sm_count=sm_count,
                        pages=pages, batch=B, split=split,
                        merge_mode=name, candidate_count=split * 32,
                        mean_us=t["mean"], p50_us=t["p50"], p90_us=t["p90"],
                        min_us=t["min"], max_us=t["max"], std_us=t["std"],
                        notes="",
                    ))
    return rows


def write_merge_mode_csv(merge_rows: List[dict], path: Path):
    if not merge_rows:
        with path.open("w") as f:
            f.write("# merge ablation skipped (use --merge-ablation to enable)\n")
        return
    # Pivot to wide form: one row per (pages, batch, split) with columns per merge mode.
    by_key = {}
    for r in merge_rows:
        key = (r["pages"], r["batch"], r["split"])
        by_key.setdefault(key, {"candidate_count": r.get("candidate_count")})
        by_key[key][r["merge_mode"]] = r.get("mean_us")
    cols = ["pages", "batch", "split", "candidate_count",
            "warp_cub_us", "block_cub_us", "kway_us", "prod_default_us",
            "best_merge_mode", "best_merge_us", "speedup_best_vs_warp"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key in sorted(by_key):
            pages, B, split = key
            d = by_key[key]
            warp  = d.get("warp_cub")
            block = d.get("block_cub")
            kway  = d.get("kway")
            prod  = d.get("prod_default(legacy)")
            choices = [(name, t) for name, t in
                       (("warp_cub", warp), ("block_cub", block),
                        ("kway", kway), ("prod_default", prod))
                       if t is not None]
            if choices:
                best_name, best_us = min(choices, key=lambda x: x[1])
                sp = (warp / best_us) if (warp and best_us) else None
            else:
                best_name, best_us, sp = "n/a", None, None
            w.writerow([pages, B, split, d.get("candidate_count"),
                        f"{warp:.3f}"  if warp  else "",
                        f"{block:.3f}" if block else "",
                        f"{kway:.3f}"  if kway  else "",
                        f"{prod:.3f}"  if prod  else "",
                        best_name,
                        f"{best_us:.3f}" if best_us else "",
                        f"{sp:.3f}"      if sp      else ""])
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Adversarial correctness — additional unit-test-style cases.
# --------------------------------------------------------------------------- #
def adversarial_correctness_test(local_mode: int) -> List[dict]:
    """Return list of dicts describing each adversarial case + per-method outcome."""
    device = torch.device("cuda")
    K, RES_BOS, RES_EOS, B, PAGES = 30, 1, 2, 2, 4096
    cases = []

    def build_scores(kind: str, dtype) -> torch.Tensor:
        n = B * PAGES
        if kind == "all_equal":
            return torch.full((n,), 1.5, device=device, dtype=dtype)
        if kind == "tie_heavy_high8":
            x = torch.randn(n, device=device, dtype=torch.float32)
            mask = torch.rand(n, device=device) < 0.05
            x[mask] = 100.0  # identical large values - ties for top-K
            return x.to(dtype)
        if kind == "mixed_sign":
            x = torch.randn(n, device=device, dtype=torch.float32) * 10
            return x.to(dtype)
        if kind == "threshold_overflow":
            x = torch.zeros(n, device=device, dtype=torch.float32)
            mask = torch.rand(n, device=device) < 0.10
            x[mask] = 1.0  # > K items at one bin
            return x.to(dtype)
        raise ValueError(kind)

    ws = make_workspace(B, max_split=32, K_local=32)
    for kind in ("all_equal", "tie_heavy_high8", "mixed_sign", "threshold_overflow"):
        scores = build_scores(kind, torch.bfloat16)
        inp = make_inputs(B, PAGES, K, torch.bfloat16)
        inp.scores.copy_(scores)
        ref_sets, ref_remapped, thresholds = reference_topk(inp, MAPPING_NONE, 0.5)
        # Fused
        try:
            inp.out.fill_(-1)
            call_fused(inp, MAPPING_NONE, 0.5); torch.cuda.synchronize()
            ok_f, note_f = check_correctness(inp, ref_sets, ref_remapped, thresholds, MAPPING_NONE, 0.5)
        except Exception as e:
            ok_f, note_f = False, f"raised: {e}"
        # Adaptive split=1 (production path).
        try:
            inp.out.fill_(-1)
            call_adaptive(inp, ws, MAPPING_NONE, 0.5,
                          forced_split=1, forced_partition=1, local_mode=local_mode)
            torch.cuda.synchronize()
            ok_a1, note_a1 = check_correctness(inp, ref_sets, ref_remapped, thresholds, MAPPING_NONE, 0.5)
        except Exception as e:
            ok_a1, note_a1 = False, f"raised: {e}"
        # Adaptive split=4 (merge path).
        try:
            inp.out.fill_(-1)
            call_adaptive(inp, ws, MAPPING_NONE, 0.5,
                          forced_split=4, forced_partition=1, local_mode=local_mode)
            torch.cuda.synchronize()
            ok_a4, note_a4 = check_correctness(inp, ref_sets, ref_remapped, thresholds, MAPPING_NONE, 0.5)
        except Exception as e:
            ok_a4, note_a4 = False, f"raised: {e}"
        cases.append(dict(case=kind, fused_ok=ok_f, fused_note=note_f,
                          adapt_split1_ok=ok_a1, adapt_split1_note=note_a1,
                          adapt_split4_ok=ok_a4, adapt_split4_note=note_a4))
    return cases


# --------------------------------------------------------------------------- #
# CSV / report writers.
# --------------------------------------------------------------------------- #
RAW_COLUMNS = [
    "device_name", "sm_count", "dtype", "mapping_mode", "mapping_name", "mapping_power",
    "pages", "topk", "batch", "method",
    "requested_split", "actual_split",
    "local_mode", "merge_mode", "candidate_count",
    "actual_path",
    "mean_us", "p50_us", "p90_us", "min_us", "max_us", "std_us",
    "correctness", "speedup_vs_sglang_fused", "speedup_vs_topk_cu", "notes",
]


def write_raw_csv(rows: List[Row], path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RAW_COLUMNS)
        for r in rows:
            d = asdict(r)
            w.writerow([d[c] for c in RAW_COLUMNS])
    print(f"wrote {path}  ({len(rows)} rows)")


def write_best_adaptive_csv(rows: List[Row], path: Path):
    """One row per (pages,K,B,mapping,dtype). Choose best adaptive split among
    rows whose actual_path starts with 'adaptive_'. Also emit fused/cub for ref."""
    cols = [
        "pages", "topk", "batch", "mapping_name", "dtype",
        "best_adaptive_split", "best_adaptive_local_mode", "best_adaptive_merge_mode",
        "best_adaptive_latency_us", "best_adaptive_actual_path",
        "sglang_fused_latency_us", "topk_cu_latency_us",
        "speedup_best_adaptive_vs_sglang", "speedup_best_adaptive_vs_topk_cu",
        "adaptive_wins_vs_sglang", "adaptive_wins_vs_topk_cu",
    ]
    by_key: Dict[Tuple, Dict[str, object]] = {}
    for r in rows:
        key = (r.pages, r.topk, r.batch, r.mapping_name, r.dtype)
        rec = by_key.setdefault(key, dict(adaptive=[], fused_us=None, cub_us=None))
        if r.method == "topk_sglang_fused" and r.correctness and r.mean_us is not None:
            rec["fused_us"] = r.mean_us
        elif r.method == "topk_cu" and r.correctness and r.mean_us is not None:
            rec["cub_us"] = r.mean_us
        elif (r.method == "adaptive_merge" and r.correctness
              and r.actual_path.startswith("adaptive_")
              and r.mean_us is not None):
            rec["adaptive"].append((r.mean_us, r.requested_split, r.local_mode,
                                    r.merge_mode, r.actual_path))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, rec in sorted(by_key.items()):
            pages, K, B, mapping_name, dtype = key
            best = min(rec["adaptive"], default=None)
            if best is None:
                best_us, best_split, best_local, best_merge, best_path = None, None, "n/a", "n/a", "n/a"
            else:
                best_us, best_split, best_local, best_merge, best_path = best
            fused_us = rec["fused_us"]
            cub_us   = rec["cub_us"]
            sp_f = (fused_us / best_us) if (best_us and fused_us) else None
            sp_c = (cub_us   / best_us) if (best_us and cub_us)   else None
            wins_f = (sp_f is not None and sp_f >= WIN_THRESHOLD)
            wins_c = (sp_c is not None and sp_c >= WIN_THRESHOLD)
            w.writerow([
                pages, K, B, mapping_name, dtype,
                best_split, best_local, best_merge,
                f"{best_us:.3f}" if best_us is not None else "",
                best_path,
                f"{fused_us:.3f}" if fused_us is not None else "",
                f"{cub_us:.3f}"   if cub_us   is not None else "",
                f"{sp_f:.3f}"     if sp_f     is not None else "",
                f"{sp_c:.3f}"     if sp_c     is not None else "",
                wins_f, wins_c,
            ])
    print(f"wrote {path}")


def k_bucket(K: int) -> str:
    if K <= 32:    return "small_K(<=32)"
    if K <= 512:   return "mid_K(64-512)"
    return "large_K(>=1024)"


def write_advantage_summary_csv(rows: List[Row], path: Path):
    """Group by (k_bucket, pages, batch). Count adaptive wins, mean/best speedup,
    best split distribution, common actual_path."""
    by_key: Dict[Tuple, Dict[str, list]] = {}
    # Build per-cell best_adaptive entries (one per setting).
    setting_best: Dict[Tuple, Dict] = {}
    for r in rows:
        if not (r.method == "adaptive_merge" and r.correctness and r.mean_us is not None):
            continue
        if not r.actual_path.startswith("adaptive_"):
            continue
        key = (r.pages, r.topk, r.batch, r.mapping_name, r.dtype)
        rec = setting_best.setdefault(key, {"best_us": float("inf"), "split": None, "path": None})
        if r.mean_us < rec["best_us"]:
            rec["best_us"] = r.mean_us; rec["split"] = r.requested_split; rec["path"] = r.actual_path
    fused_lookup = {(r.pages, r.topk, r.batch, r.mapping_name, r.dtype): r.mean_us
                    for r in rows
                    if r.method == "topk_sglang_fused" and r.correctness and r.mean_us is not None}

    for setting, best in setting_best.items():
        pages, K, B, mapping_name, dtype = setting
        bucket = k_bucket(K)
        gkey = (bucket, pages, B, mapping_name, dtype)
        agg = by_key.setdefault(gkey, dict(speedups=[], splits=[], paths=[], total=0, wins=0))
        agg["total"] += 1
        fused_us = fused_lookup.get(setting)
        if fused_us:
            sp = fused_us / best["best_us"]
            agg["speedups"].append(sp)
            if sp >= WIN_THRESHOLD: agg["wins"] += 1
        agg["splits"].append(best["split"])
        agg["paths"].append(best["path"])

    cols = ["k_bucket", "pages", "batch", "mapping_name", "dtype",
            "n_settings", "n_adaptive_wins", "win_rate",
            "best_speedup", "mean_speedup", "median_speedup",
            "best_split_mode", "best_split_distribution", "common_actual_path"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for gkey, agg in sorted(by_key.items()):
            bucket, pages, B, mapping_name, dtype = gkey
            sps = agg["speedups"]
            splits = [s for s in agg["splits"] if s is not None]
            mode_split = (statistics.mode(splits) if splits else None)
            split_dist = "|".join(f"{s}:{splits.count(s)}" for s in sorted(set(splits))) if splits else ""
            paths = [p for p in agg["paths"] if p]
            common_path = statistics.mode(paths) if paths else ""
            w.writerow([
                bucket, pages, B, mapping_name, dtype,
                agg["total"], agg["wins"],
                f"{agg['wins']/agg['total']:.2f}" if agg["total"] else "",
                f"{max(sps):.3f}"             if sps else "",
                f"{statistics.mean(sps):.3f}" if sps else "",
                f"{statistics.median(sps):.3f}" if sps else "",
                mode_split, split_dist, common_path,
            ])
    print(f"wrote {path}")


def _fmt(v, nd=2, default="    -"):
    if v is None: return default
    return f"{v:>{nd+5}.{nd}f}"


def print_per_K_tables(rows: List[Row], splits: List[int], mapping_filter: Optional[str] = None):
    """Compact per-K terminal tables."""
    keys = sorted({(r.topk, r.mapping_name, r.dtype, r.pages, r.batch) for r in rows})
    by_k_map = {}
    for r in rows:
        if mapping_filter is not None and r.mapping_name != mapping_filter:
            continue
        by_k_map.setdefault((r.topk, r.mapping_name, r.dtype), []).append(r)
    for (K, mapping_name, dtype), kr in sorted(by_k_map.items()):
        print()
        print(f"=== K={K}  mapping={mapping_name}  dtype={dtype} ===")
        # Column header
        hdr = (f"{'pages':>6} {'B':>3} {'fused_us':>9} {'cub_us':>8}  "
               + " ".join(f"{'s='+str(s):>9}" for s in splits)
               + f"  {'best_us':>8} {'split':>5} {'sp_vs_fused':>11}")
        print(hdr)
        print("-" * len(hdr))
        cells = {}
        for r in kr:
            cells.setdefault((r.pages, r.batch), {})
            kk = (r.pages, r.batch)
            if r.method == "topk_sglang_fused":
                cells[kk]["fused"] = r.mean_us
            elif r.method == "topk_cu":
                cells[kk]["cub"] = r.mean_us
            elif r.method == "adaptive_merge":
                cells[kk].setdefault("adapt", {})[r.requested_split] = (r.mean_us, r.actual_path)
        for (pages, B), c in sorted(cells.items()):
            adapt = c.get("adapt", {})
            adapt_us = {s: (adapt.get(s, (None, ""))[0]) for s in splits}
            valid = [(s, adapt[s][0]) for s in splits if s in adapt
                     and adapt[s][0] is not None
                     and adapt[s][1].startswith("adaptive_")]
            if valid:
                best_split, best_us = min(valid, key=lambda kv: kv[1])
            else:
                best_split, best_us = None, None
            fused_us = c.get("fused")
            sp = (fused_us / best_us) if (fused_us and best_us) else None
            print(
                f"{pages:>6d} {B:>3d} {_fmt(fused_us):>9} {_fmt(c.get('cub')):>8}  "
                + " ".join(f"{_fmt(adapt_us[s]):>9}" for s in splits)
                + f"  {_fmt(best_us):>8} {str(best_split) if best_split else '-':>5} "
                + (f"{sp:>10.3f}x" if sp else f"{'-':>11}")
            )


def write_markdown_report(rows: List[Row], device_info: dict, args, path: Path,
                          splits: List[int], best_csv: Path, advantage_csv: Path,
                          raw_csv: Path, merge_csv: Optional[Path] = None,
                          merge_rows: Optional[List[dict]] = None,
                          adversarial_csv: Optional[Path] = None,
                          adversarial_rows: Optional[List[dict]] = None):
    failed = [r for r in rows if r.correctness is False]
    n_total = len(rows)
    with path.open("w") as f:
        f.write(f"# TopK Setting Sweep Report\n\n")
        f.write(f"- Device: **{device_info['name']}**  (SMs: {device_info['sm_count']})\n")
        f.write(f"- torch: {torch.__version__}  CUDA: {torch.version.cuda}\n")
        f.write(f"- Pages: {args.pages}\n")
        f.write(f"- K: {args.ks}\n")
        f.write(f"- Batches: {args.batches}\n")
        f.write(f"- Adaptive splits: {splits}\n")
        f.write(f"- Mappings: {args.mappings}\n")
        f.write(f"- Local mode: {'BLOCK_FULL_SORT' if args.local_mode == LOCAL_BLOCK_FULL_SORT else 'SELECT32_SORT32'}\n")
        f.write(f"- warmup={args.warmup}, repeat={args.repeat}\n")
        f.write(f"- Total measurements: {n_total}, correctness failures: {len(failed)}\n\n")

        # Per-K compact tables.
        f.write("## Per-K latency tables (us)\n\n")
        by_k = {}
        for r in rows:
            by_k.setdefault((r.topk, r.mapping_name, r.dtype), []).append(r)
        for (K, mapping_name, dtype), kr in sorted(by_k.items()):
            f.write(f"### K={K}, mapping={mapping_name}, dtype={dtype}\n\n")
            cells = {}
            for r in kr:
                kk = (r.pages, r.batch)
                cells.setdefault(kk, {})
                if r.method == "topk_sglang_fused": cells[kk]["fused"] = r.mean_us
                elif r.method == "topk_cu":         cells[kk]["cub"]   = r.mean_us
                elif r.method == "adaptive_merge":
                    cells[kk].setdefault("adapt", {})[r.requested_split] = (r.mean_us, r.actual_path)
            head = (["pages", "B", "fused_us", "cub_us"]
                    + [f"adapt_s{s}_us" for s in splits]
                    + ["best_us", "best_split", "actual_path", "speedup_vs_fused"])
            f.write("| " + " | ".join(head) + " |\n")
            f.write("|" + "|".join("---:" for _ in head) + "|\n")
            for (pages, B), c in sorted(cells.items()):
                adapt = c.get("adapt", {})
                row = [str(pages), str(B),
                       f"{c.get('fused'):.2f}" if c.get('fused') else "-",
                       f"{c.get('cub'):.2f}"   if c.get('cub')   else "-"]
                for s in splits:
                    val = adapt.get(s, (None, ""))[0]
                    row.append(f"{val:.2f}" if val else "-")
                valid = [(s, adapt[s][0], adapt[s][1]) for s in splits
                         if s in adapt and adapt[s][0] is not None
                         and adapt[s][1].startswith("adaptive_")]
                if valid:
                    best_split, best_us, best_path = min(valid, key=lambda x: x[1])
                else:
                    best_split, best_us, best_path = None, None, "-"
                fused_us = c.get('fused')
                sp = (fused_us / best_us) if (fused_us and best_us) else None
                # If everything was fused-fallback, fall back to noting that.
                if best_us is None:
                    fb = next((v for v in adapt.values() if v[0] is not None), (None, "-"))
                    row += ["-", "-", fb[1], "-"]
                else:
                    row += [f"{best_us:.2f}", str(best_split), best_path,
                            f"{sp:.3f}x" if sp else "-"]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

        # Merge-mode ablation (K=30 only).
        if merge_csv is not None and merge_rows:
            f.write("## Merge-mode ablation (K=30, merge stage in isolation)\n\n")
            f.write("Source: `topk_output_adaptive_workspace_ablation` modes 5/6/7/11.\n\n")
            with merge_csv.open() as g:
                f.write("```\n" + g.read() + "```\n\n")

        # Region analysis.
        f.write("## Parallel-advantage region analysis\n\n")
        f.write(f"Win threshold: speedup_vs_sglang >= {WIN_THRESHOLD}.\n\n")
        with advantage_csv.open() as g:
            f.write("```\n" + g.read() + "```\n\n")

        # Adversarial correctness.
        if adversarial_rows is not None:
            f.write("## Adversarial correctness cases\n\n")
            f.write("| case | fused | adapt s=1 | adapt s=4 |\n")
            f.write("|---|:-:|:-:|:-:|\n")
            for r in adversarial_rows:
                f.write(f"| {r['case']} | {'PASS' if r['fused_ok'] else 'FAIL'} | "
                        f"{'PASS' if r['adapt_split1_ok'] else 'FAIL'} | "
                        f"{'PASS' if r['adapt_split4_ok'] else 'FAIL'} |\n")
            f.write("\n")

        # Recommended dispatch policy
        f.write("## Recommended production dispatch policy\n\n")
        # Compute best split per (K bucket, pages) by majority best_split.
        best_by_bucket = {}
        for r in rows:
            if not (r.method == "adaptive_merge" and r.correctness
                    and r.mean_us is not None
                    and r.actual_path.startswith("adaptive_")):
                continue
            key = (k_bucket(r.topk), r.pages, r.batch, r.mapping_name)
            ent = best_by_bucket.setdefault(key, {"best": (float('inf'), None)})
            if r.mean_us < ent["best"][0]:
                ent["best"] = (r.mean_us, r.requested_split)
        bucket_splits = {}
        for (bucket, pages, B, mapping), v in best_by_bucket.items():
            bucket_splits.setdefault((bucket, pages, B, mapping), []).append(v["best"][1])
        f.write("| K_bucket | pages | B | mapping | recommended_split |\n")
        f.write("|---|---:|---:|---|---:|\n")
        for key, splits_list in sorted(bucket_splits.items()):
            bucket, pages, B, mapping = key
            try:
                rec = statistics.mode(splits_list)
            except statistics.StatisticsError:
                rec = sorted(splits_list)[0]
            f.write(f"| {bucket} | {pages} | {B} | {mapping} | {rec} |\n")
        f.write("\n- For `large_K(>=1024)` adaptive entry routes to fused (zero-overhead).\n")
        f.write("- For `mid_K(64-512)` adaptive entry currently routes to fused; ")
        f.write("a dedicated mid-K kernel is future work.\n")

        # Failures.
        if failed:
            f.write("\n## Correctness failures\n\n")
            f.write("| pages | K | B | mapping | method | split | actual_path | notes |\n")
            f.write("|---:|---:|---:|---|---|---:|---|---|\n")
            for r in failed:
                f.write(f"| {r.pages} | {r.topk} | {r.batch} | {r.mapping_name} | "
                        f"{r.method} | {r.requested_split} | {r.actual_path} | "
                        f"{r.notes} |\n")
        f.write(f"\n## Files\n- raw csv: `{raw_csv}`\n")
        f.write(f"- best adaptive csv: `{best_csv}`\n")
        f.write(f"- advantage summary csv: `{advantage_csv}`\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pages",    type=int, nargs="+", default=DEFAULT_PAGES)
    p.add_argument("--ks",       type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--batches",  type=int, nargs="+", default=DEFAULT_BATCHES)
    p.add_argument("--splits",   type=int, nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--mappings", type=str, nargs="+", default=DEFAULT_MAPPING_NAMES,
                   help="Mapping mode names (NONE, TRUNC8, POWER, LOG, ASINH, LOG1P, ERF, TANH).")
    p.add_argument("--mapping-power", type=float, default=0.5)
    p.add_argument("--dtypes",   type=str, nargs="+", default=DEFAULT_DTYPES)
    p.add_argument("--local-mode", type=int, default=LOCAL_SELECT32_SORT32,
                   choices=[LOCAL_BLOCK_FULL_SORT, LOCAL_SELECT32_SORT32],
                   help="0=BLOCK_FULL_SORT, 1=SELECT32_SORT32 (default).")
    p.add_argument("--warmup",   type=int, default=20)
    p.add_argument("--repeat",   type=int, default=200)
    p.add_argument("--output-dir", type=Path,
                   default=Path("bench_results") / time.strftime("setting_sweep_%Y%m%d_%H%M%S"))
    p.add_argument("--print-tables", action="store_true",
                   help="Also print per-K latency tables to stdout (large output).")
    p.add_argument("--merge-ablation", action="store_true", default=True,
                   help="Run merge-mode ablation sub-sweep (K=30 only). Default: on.")
    p.add_argument("--no-merge-ablation", dest="merge_ablation", action="store_false",
                   help="Skip the merge-mode ablation sub-sweep.")
    p.add_argument("--adversarial", action="store_true", default=True,
                   help="Run adversarial correctness cases (default: on).")
    p.add_argument("--no-adversarial", dest="adversarial", action="store_false",
                   help="Skip adversarial correctness cases.")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_info = dict(
        name=torch.cuda.get_device_name(0),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    print(f"Device: {device_info['name']}  SMs={device_info['sm_count']}")
    print(f"Output dir: {args.output_dir.resolve()}")

    # Validate mappings.
    mapping_modes = []
    for name in args.mappings:
        if name not in MAPPING_BY_NAME:
            sys.exit(f"unknown mapping name: {name} (valid: {list(MAPPING_BY_NAME)})")
        mapping_modes.append(MAPPING_BY_NAME[name])

    # Pre-allocate workspace large enough for the largest configuration.
    B_max = max(args.batches)
    ws = make_workspace(B_max=B_max, max_split=max(args.splits), K_local=32)

    configs = [(pages, K, B, mode, dtype)
               for pages in args.pages
               for K     in args.ks
               for B     in args.batches
               for mode  in mapping_modes
               for dtype in args.dtypes]
    print(f"Configs: {len(configs)}  (each runs fused + cub + {len(args.splits)} adaptive)")
    print(f"Splits: {args.splits}  warmup={args.warmup} repeat={args.repeat}")

    rows: List[Row] = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        pages, K, B, mode, dtype = cfg
        if i % 5 == 0 or i == 1:
            print(f"[{i:3d}/{len(configs)}] pages={pages} K={K} B={B} "
                  f"mapping={MAPPING_NAMES[mode]} dtype={dtype}  "
                  f"(elapsed {time.time()-t0:.1f}s)")
        rows.extend(run_one_setting(
            pages, K, B, mode, args.mapping_power, dtype,
            args.splits, args.local_mode,
            args.warmup, args.repeat, ws,
            device_info["name"], device_info["sm_count"],
        ))
    print(f"\nSweep complete in {time.time()-t0:.1f}s.  rows={len(rows)}")

    # Output files.
    raw_csv         = args.output_dir / "topk_setting_sweep_raw.csv"
    best_csv        = args.output_dir / "topk_setting_sweep_best_adaptive.csv"
    advantage_csv   = args.output_dir / "topk_parallel_advantage_summary.csv"
    merge_csv       = args.output_dir / "topk_merge_mode_summary.csv"
    adversarial_csv = args.output_dir / "topk_adversarial_correctness.csv"
    report_md       = args.output_dir / "topk_setting_sweep_report.md"

    write_raw_csv(rows, raw_csv)
    write_best_adaptive_csv(rows, best_csv)
    write_advantage_summary_csv(rows, advantage_csv)

    # Merge-mode ablation (K=30 only).
    merge_rows = []
    if args.merge_ablation:
        print("\nMerge-mode ablation (K=30, ablation harness):")
        merge_rows = run_merge_ablation(
            pages_list=args.pages, batches=args.batches,
            splits=args.splits, warmup=args.warmup, repeat=args.repeat,
            ws=ws, device_name=device_info["name"], sm_count=device_info["sm_count"],
        )
        print(f"  collected {len(merge_rows)} merge-only timings")
    write_merge_mode_csv(merge_rows, merge_csv)

    # Adversarial correctness check.
    adv_rows = []
    if args.adversarial:
        print("\nAdversarial correctness check (K=30, MAPPING_NONE, B=2, pages=4096):")
        adv_rows = adversarial_correctness_test(args.local_mode)
        for r in adv_rows:
            print(f"  case={r['case']:>22}  fused={r['fused_ok']}  "
                  f"adapt_s1={r['adapt_split1_ok']}  adapt_s4={r['adapt_split4_ok']}")
    with adversarial_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "fused_ok", "fused_note",
                    "adapt_split1_ok", "adapt_split1_note",
                    "adapt_split4_ok", "adapt_split4_note"])
        for r in adv_rows:
            w.writerow([r["case"], r["fused_ok"], r["fused_note"],
                        r["adapt_split1_ok"], r["adapt_split1_note"],
                        r["adapt_split4_ok"], r["adapt_split4_note"]])
    print(f"wrote {adversarial_csv}")

    write_markdown_report(rows, device_info, args, report_md,
                          args.splits, best_csv, advantage_csv, raw_csv,
                          merge_csv=merge_csv, merge_rows=merge_rows,
                          adversarial_csv=adversarial_csv, adversarial_rows=adv_rows)

    # Optional terminal tables.
    if args.print_tables:
        print_per_K_tables(rows, args.splits)

    # Short summary.
    print("\n" + "=" * 70)
    print(f"Files written under: {args.output_dir.resolve()}")
    print(f"  raw       : {raw_csv.name}")
    print(f"  best_adapt: {best_csv.name}")
    print(f"  advantage : {advantage_csv.name}")
    print(f"  report    : {report_md.name}")
    failed = [r for r in rows if r.correctness is False]
    print(f"Correctness failures: {len(failed)}")
    if failed:
        for r in failed[:10]:
            print(f"  - pages={r.pages} K={r.topk} B={r.batch} "
                  f"map={r.mapping_name} method={r.method} split={r.requested_split} "
                  f"path={r.actual_path}: {r.notes}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more (see raw csv)")
    # Quick win-region rollup.
    n_adaptive = sum(1 for r in rows if r.method == "adaptive_merge"
                     and r.actual_path.startswith("adaptive_") and r.correctness)
    n_wins = sum(1 for r in rows if r.method == "adaptive_merge"
                 and r.actual_path.startswith("adaptive_") and r.correctness
                 and r.speedup_vs_sglang_fused is not None
                 and r.speedup_vs_sglang_fused >= WIN_THRESHOLD)
    print(f"Adaptive-rows (correct, real adaptive path): {n_adaptive}")
    print(f"  -> wins vs fused (>= {WIN_THRESHOLD}x): {n_wins} "
          f"({100.0 * n_wins / max(n_adaptive,1):.1f}%)")


if __name__ == "__main__":
    main()
