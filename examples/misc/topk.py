"""Latency + recall comparison: best JIT top-k kernel vs CUB baseline.

- Baseline (v1): ``vortex_torch/kernels/topk/configs/sort_default.json``
  (CUB ``BlockRadixSort``, fp32 keys).
- Best (v2):     ``vortex_torch/kernels/topk/k_256/configs/claude_opus_4_7/batch_24_id1.json``
  (regcache-unrolled 8-bit radix + ``x*|x|`` transform, T=512, smem=8KB).
  Geomean ~1.48x over baseline at R@253 >= 0.9952 — see
  ``vortex_torch/kernels/topk/k_256/reports/claude_opus_4_7/batch_24_id1.md``.

Both kernels share the entry-point signature

    topk(x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
         sparse_kv_indices, eff_batch_size, reserved_bos, reserved_eos,
         max_num_pages)
"""

from pathlib import Path

import torch
import triton
from tqdm import tqdm

import vortex_torch.kernels.topk as _vt_topk
from vortex_torch.kernels.topk.dispatcher import load_submission


_TOPK_ROOT = Path(_vt_topk.__file__).resolve().parent
BASELINE_CONFIG = _TOPK_ROOT / "configs" / "sort_default.json"
BEST_CONFIG = _TOPK_ROOT / "k_256" / "configs" / "claude_opus_4_7" / "batch_24_id1.json"


SEQ_LENS = [1024, 1536, 2048, 4096]
BATCH_SIZES = [16, 32, 64, 128]

K = 253
EVAL_KS = [5, 10, 16, 32, 64, 96, 125, 157, 189, 253]

RESERVE_BOS = 0
RESERVE_EOS = 0
DEVICE = "cuda"
NUM_RUNS = 10


def make_inputs(batch_size, seq_len, k, reserve_bos, reserve_eos, device="cuda"):
    dense_kv_indptr = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device
    )

    dense_kv_indices = torch.arange(
        0, batch_size * seq_len, dtype=torch.int32, device=device
    )

    scores = torch.randn(
        batch_size * seq_len, dtype=torch.bfloat16, device=device
    )

    sparse_kv_indptr = torch.arange(
        0, batch_size * k + 1, k, dtype=torch.int32, device=device
    )

    sparse_kv_indices_v1 = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )

    sparse_kv_indices_v2 = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )

    return (
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices_v1,
        sparse_kv_indices_v2,
    )


def run_kernel(
    module,
    scores,
    dense_kv_indptr,
    sparse_kv_indptr,
    dense_kv_indices,
    sparse_kv_indices,
    batch_size,
    reserve_bos,
    reserve_eos,
    seq_len,
):
    module.topk(
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        sparse_kv_indices,
        batch_size,
        reserve_bos,
        reserve_eos,
        seq_len,
    )


def compute_recall_at_ks(ref_indices, pred_indices, batch_size, k, eval_ks):
    """
    ref_indices: v1 output, ordered top-k, shape [batch_size * k]
    pred_indices: v2 output, unordered k candidates, shape [batch_size * k]

    recall@r = fraction of ref top-r recovered in pred top-k
    """
    ref = ref_indices.view(batch_size, k).to(torch.int64)
    pred = pred_indices.view(batch_size, k).to(torch.int64)

    recalls = {}

    for r in eval_ks:
        ref_r = ref[:, :r]
        match = (ref_r.unsqueeze(2) == pred.unsqueeze(1))
        hit = match.any(dim=2)
        recall = hit.float().sum(dim=1) / r
        recalls[r] = recall.mean().item()

    return recalls


def bench_latency(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        fn,
        warmup=100,
        rep=1000,
        return_mode="mean",
    )
    return ms


def eval_one(baseline_mod, best_mod, batch_size, seq_len, k, reserve_bos, reserve_eos):
    (
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices_v1,
        sparse_kv_indices_v2,
    ) = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        k=k,
        reserve_bos=reserve_bos,
        reserve_eos=reserve_eos,
        device=DEVICE,
    )

    def fn_v1():
        run_kernel(
            baseline_mod,
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices_v1,
            batch_size,
            reserve_bos,
            reserve_eos,
            seq_len,
        )

    def fn_v2():
        run_kernel(
            best_mod,
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices_v2,
            batch_size,
            reserve_bos,
            reserve_eos,
            seq_len,
        )

    fn_v1()
    fn_v2()
    torch.cuda.synchronize()
    recalls = compute_recall_at_ks(
        sparse_kv_indices_v1,
        sparse_kv_indices_v2,
        batch_size=batch_size,
        k=k,
        eval_ks=EVAL_KS,
    )

    ms_v1 = bench_latency(fn_v1)
    ms_v2 = bench_latency(fn_v2)

    return ms_v1, ms_v2, recalls


def main():
    torch.cuda.init()

    print(f"Baseline (v1): {BASELINE_CONFIG}")
    print(f"Best     (v2): {BEST_CONFIG}")
    print("Compiling kernels (JIT, cached)...")
    baseline_mod = load_submission(str(BASELINE_CONFIG))
    best_mod = load_submission(str(BEST_CONFIG))

    results = {}

    for bs in BATCH_SIZES:
        results[bs] = {}
        for seq_len in SEQ_LENS:
            agg = {
                "v1_ms": [],
                "v2_ms": [],
                5: [],
                10: [],
                16: [],
                32: [],
                64: [],
                96: [],
                125: [],
                157: [],
                189: [],
                253: []
            }

            for seed in tqdm(range(NUM_RUNS)):
                torch.cuda.manual_seed(seed)
                ms_v1, ms_v2, recalls = eval_one(
                    baseline_mod,
                    best_mod,
                    batch_size=bs,
                    seq_len=seq_len,
                    k=K,
                    reserve_bos=RESERVE_BOS,
                    reserve_eos=RESERVE_EOS,
                )
                agg["v1_ms"].append(ms_v1)
                agg["v2_ms"].append(ms_v2)
                for kk in EVAL_KS:
                    agg[kk].append(recalls[kk])

            result = {
                "v1_ms_mean": sum(agg["v1_ms"]) / len(agg["v1_ms"]),
                "v2_ms_mean": sum(agg["v2_ms"]) / len(agg["v2_ms"]),
                "speedup": (sum(agg["v1_ms"]) / len(agg["v1_ms"])) / (sum(agg["v2_ms"]) / len(agg["v2_ms"])),
                "recall@5": sum(agg[5]) / len(agg[5]),
                "recall@10": sum(agg[10]) / len(agg[10]),
                "recall@16": sum(agg[16]) / len(agg[16]),
                "recall@32": sum(agg[32]) / len(agg[32]),
                "recall@64": sum(agg[64]) / len(agg[64]),
                "recall@96": sum(agg[96]) / len(agg[96]),
                "recall@125": sum(agg[125]) / len(agg[125]),
                "recall@157": sum(agg[157]) / len(agg[157]),
                "recall@189": sum(agg[189]) / len(agg[189]),
                "recall@253": sum(agg[253]) / len(agg[253]),
            }
            results[bs][seq_len] = result

            print(
                f"bs={bs:>4}, seq_len={seq_len:>6} | "
                f"v1={result['v1_ms_mean']:.6f} ms, "
                f"v2={result['v2_ms_mean']:.6f} ms, "
                f"speedup={result['speedup']:.4f}x | "
                f"R@5={result['recall@5']:.6f}, "
                f"R@10={result['recall@10']:.6f}, "
                f"R@16={result['recall@16']:.6f}, "
                f"R@32={result['recall@32']:.6f}, "
                f"R@64={result['recall@64']:.6f}, "
                f"R@96={result['recall@96']:.6f}, "
                f"R@125={result['recall@125']:.6f}, "
                f"R@157={result['recall@157']:.6f}, "
                f"R@189={result['recall@189']:.6f}, "
                f"R@253={result['recall@253']:.6f}"
            )

    print("\nSummary:")
    for bs in BATCH_SIZES:
        for seq_len in SEQ_LENS:
            r = results[bs][seq_len]
            print(
                f"[bs={bs}, seq={seq_len}] "
                f"v1={r['v1_ms_mean']:.6f} ms, "
                f"v2={r['v2_ms_mean']:.6f} ms, "
                f"speedup={r['speedup']:.4f}x, "
                f"R@5={r['recall@5']:.6f}, "
                f"R@10={r['recall@10']:.6f}, "
                f"R@16={r['recall@16']:.6f}, "
                f"R@32={r['recall@32']:.6f}, "
                f"R@64={r['recall@64']:.6f}, "
                f"R@96={r['recall@96']:.6f}, "
                f"R@125={r['recall@125']:.6f}, "
                f"R@157={r['recall@157']:.6f}, "
                f"R@189={r['recall@189']:.6f}, "
                f"R@253={r['recall@253']:.6f}"
            )


if __name__ == "__main__":
    main()
