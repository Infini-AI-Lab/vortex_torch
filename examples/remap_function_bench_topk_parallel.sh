#!/usr/bin/env bash
# ============================================================
# Three-way TopK kernel latency comparison for K=30.
#
# Compares (per (batch_size, pages)):
#   topk.cu              -> topk_output                (CUB BlockRadixSort full sort)
#   topk_sglang.cu       -> topk_output_sglang        +
#                           topk_output_sglang_fused  (2-stage radix select)
#   topk_sglang_merge.cu -> topk_output_adaptive       (adaptive split SELECT32_SORT32)
#
# Pages are varied by --seq-lens (with --page-size 1: pages == seq_len).
# Default sweep is the matrix the user requested:
#   batch_sizes = {1, 2, 4, 8, 16}
#   pages       = {4096, 8192, 16384}
#   topk        = 30
#
# No calibration, no remap autotune, no model download — purely synthetic
# scores so the only variable is the kernel itself.
#
# Usage:
#   bash examples/remap_function_bench_topk_parallel.sh --gpu 0
#   bash examples/remap_function_bench_topk_parallel.sh --gpu 0 \
#       --batch-sizes "1 2 4 8 16" \
#       --seq-lens    "4096 8192 16384 32768"
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults (matrix the brief calls out) ─────────────────────
GPU_ID=0
TOPK_VALS="30"
BATCH_SIZES="1 2 4 8 16"
SEQ_LENS="4096 8192 16384"     # pages-per-seg when page-size=1
NUM_KV_HEADS=8
PAGE_SIZE=1
RESERVED_BOS=1
RESERVED_EOS=2
DISTRIBUTIONS="normal"
WARMUP=20
REPEAT=200

# ── Arg parsing ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)             GPU_ID="$2"; shift 2 ;;
    --topk-vals)       TOPK_VALS="$2"; shift 2 ;;
    --batch-sizes)     BATCH_SIZES="$2"; shift 2 ;;
    --seq-lens)        SEQ_LENS="$2"; shift 2 ;;
    --page-size)       PAGE_SIZE="$2"; shift 2 ;;
    --num-kv-heads)    NUM_KV_HEADS="$2"; shift 2 ;;
    --distributions)   DISTRIBUTIONS="$2"; shift 2 ;;
    --reserved-bos)    RESERVED_BOS="$2"; shift 2 ;;
    --reserved-eos)    RESERVED_EOS="$2"; shift 2 ;;
    --warmup)          WARMUP="$2"; shift 2 ;;
    --repeat)          REPEAT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOPK_TAG="$(echo ${TOPK_VALS} | tr ' ' '-')"
RUN_DIR="${RESULTS_DIR}/three_way_topk${TOPK_TAG}_bs${PAGE_SIZE}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"
JSON_PATH="${RUN_DIR}/three_way.json"
CSV_PATH="${RUN_DIR}/summary.csv"

echo "============================================================"
echo "Three-way TopK kernel comparison"
echo "  TopK sweep:    ${TOPK_VALS}"
echo "  Batch sizes:   ${BATCH_SIZES}"
echo "  Seq lengths:   ${SEQ_LENS}    (page_size=${PAGE_SIZE})"
echo "  KV heads:      ${NUM_KV_HEADS}"
echo "  Distributions: ${DISTRIBUTIONS}"
echo "  GPU:           ${GPU_ID}"
echo "  Warmup/repeat: ${WARMUP}/${REPEAT}"
echo "  Output dir:    ${RUN_DIR}"
echo "============================================================"

# ── Run bench_topk.py with all (B, seq_len, K) combos in one shot ──
# --mapping-modes 0 = MAPPING_NONE → no remap, no autotune needed.
# --remap-bench    = drives the per-config table that includes baseline
#                    (topk_sglang) + naive (topk.cu) + sglang_ori rows.
# --bench-parallel = adds the topk_sglang_merge adaptive measurement
#                    into each row (under "parallel_ms").
PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
  --remap-bench \
  --bench-parallel \
  --batch-sizes ${BATCH_SIZES} \
  --num-kv-heads "${NUM_KV_HEADS}" \
  --seq-lens ${SEQ_LENS} \
  --topk-vals ${TOPK_VALS} \
  --page-size "${PAGE_SIZE}" \
  --reserved-bos "${RESERVED_BOS}" \
  --reserved-eos "${RESERVED_EOS}" \
  --distributions ${DISTRIBUTIONS} \
  --mapping-modes 0 \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --output-json "${JSON_PATH}" \
  2>&1 | tee "${RUN_DIR}/bench_topk.log"

# ── Aggregate to a clean CSV: one row per (B, pages, K, dist) ─────
python - "${JSON_PATH}" "${CSV_PATH}" <<'PY'
import csv, json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    data = json.load(f)
rows = data["results"] if isinstance(data, dict) and "results" in data else data
# Header. Latencies in microseconds.
hdr = [
    "topk", "batch_size", "pages", "distribution",
    "cub_topk_us",                   # topk.cu / topk_output (None when pages > 8192)
    "sglang_baseline_us",            # topk_sglang.cu / topk_output_sglang
    "sglang_fused_us",               # topk_sglang.cu / topk_output_sglang_fused (==baseline @ MAPPING_NONE)
    "adaptive_us",                   # topk_sglang_merge.cu / topk_output_adaptive
    "speedup_adaptive_vs_fused",
    "speedup_adaptive_vs_cub",
]
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for r in rows:
        B = r["batch_size"]; pg = r["pages_per_seg"]; K = r["topk_val"]; dist = r["distribution"]
        cub = r.get("naive_ms")
        baseline = r.get("baseline_ms")
        none_mode = next((m for m in r["modes"] if m.get("mode_name") == "None"), None)
        adaptive = none_mode.get("parallel_ms") if none_mode else None
        # At MAPPING_NONE the fused kernel == baseline kernel (no remap branch),
        # so report baseline as the fused number too for clarity.
        fused = baseline
        def us(x): return f"{x*1000:.3f}" if x is not None else ""
        sp_f = f"{baseline/adaptive:.3f}" if (adaptive and baseline) else ""
        sp_c = f"{cub/adaptive:.3f}"      if (adaptive and cub) else ""
        w.writerow([K, B, pg, dist, us(cub), us(baseline), us(fused), us(adaptive), sp_f, sp_c])
print(f"wrote {dst}")
PY

# ── Print human-readable summary table ──────────────────────────
echo ""
echo "============================================================"
echo "Summary (us per kernel call; speedup = fused_us / adaptive_us)"
echo "============================================================"
column -t -s, "${CSV_PATH}" || cat "${CSV_PATH}"

echo ""
echo "Done. Results:"
echo "  raw JSON: ${JSON_PATH}"
echo "  summary:  ${CSV_PATH}"
echo "  log:      ${RUN_DIR}/bench_topk.log"
