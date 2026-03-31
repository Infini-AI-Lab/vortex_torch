#!/usr/bin/env bash
# ============================================================
# Bucket Distribution Profiling Pipeline
#
# Profiles the SGLang TopK kernel's first-pass bucket distribution
# to identify hotspot buckets causing tail latency.
#
# Three steps:
#   1. Calibrate — collect real-data histograms
#                  (skippable via --real-histograms PATH)
#   2. Bench     — histogram profiling (bucket_uniform + normal)
#   3. Analyze   — comparison plots + bucket count tables
#
# All outputs (JSON, plots, CSV tables, logs) are written to a
# single timestamped folder under examples/results/dist_analysis_*.
#
# Usage:
#   bash run_distribution_analysis.sh --gpu 5
#   bash run_distribution_analysis.sh --gpu 5 \
#       --real-histograms /path/to/calibration_dir/raw_histograms.npy
# ============================================================

# Mapping functions:
# 0: None           — original fp16 bit-pattern bucketing
# 1: LUT CDF        — LUT-based CDF equalization (calibrated)
# 2: Quantile       — piecewise-linear quantile mapping (calibrated)
# 3: Power          — y = sign(x) * |x|^p
# 4: Log            — y = sign(x) * log(|x| + 1)
# 5: Index Cache    — reuse previous layer's indices
# 6: Asinh          — y = asinh(beta * x)
# 7: Log1p          — y = sign(x) * log1p(alpha * |x|)
# 8: Trunc8         — bf16 upper-8-bit bucketing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=5
MODEL_NAME="Qwen/Qwen3-1.7B"
TOPK_VAL=30
MEM=0.7
ALGO="block_sparse_attention"
# The path to the raw_histograms.npy file (set to skip calibration)
# REAL_HISTOGRAMS="/scr/dataset/yuke/xinrui/new/vortex_torch/examples/calibration/raw_histograms.npy"
REAL_HISTOGRAMS=""
# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)       MODEL_NAME="$2"; shift 2 ;;
    --topk-val)         TOPK_VAL="$2"; shift 2 ;;
    --mem)              MEM="$2"; shift 2 ;;
    --gpu)              GPU_ID="$2"; shift 2 ;;
    --algo)             ALGO="$2"; shift 2 ;;
    --real-histograms)  REAL_HISTOGRAMS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/dist_analysis_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "Bucket Distribution Profiling Pipeline"
echo "  Model:           ${MODEL_NAME}"
echo "  Algorithm:       ${ALGO}"
echo "  TopK:            ${TOPK_VAL}"
echo "  GPU:             ${GPU_ID}"
echo "  Real histograms: ${REAL_HISTOGRAMS:-<will calibrate>}"
echo "  Output:          ${RUN_DIR}"
echo "============================================================"

# ── Step 1: Calibrate — collect real-data histograms + LUT/quantiles ──
if [ -n "${REAL_HISTOGRAMS}" ]; then
  echo ""
  echo ">>> Step 1: SKIPPED (using provided --real-histograms ${REAL_HISTOGRAMS})"
  REAL_HIST_PATH="${REAL_HISTOGRAMS}"
else
  echo ""
  echo ">>> Step 1: Calibrating — collecting real-inference histograms"
  CALIBRATION_DIR="${RUN_DIR}/calibration"
  mkdir -p "${CALIBRATION_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --mem "${MEM}" \
    --vortex-module-name "${ALGO}" \
    --output-dir "${CALIBRATION_DIR}" \
    2>&1 | tee "${RUN_DIR}/step1_calibrate.log"
  REAL_HIST_PATH="${CALIBRATION_DIR}/raw_histograms.npy"
  echo ">>> Step 1: Done. Calibration saved to ${CALIBRATION_DIR}"
fi

# ── Step 2: Histogram profiling (bucket_uniform + normal) ─────
echo ""
echo ">>> Step 2: Kernel-level histogram profiling (bucket_uniform + normal)"

BENCH_JSON="${RUN_DIR}/bench_distribution.json"

python "${BENCH_DIR}/bench_topk.py" \
  --batch-sizes 4 \
  --seq-lens 4096 \
  --topk-vals "${TOPK_VAL}" \
  --num-kv-heads 2 \
  --distributions bucket_uniform normal \
  --histogram \
  --filter-kernels sglang_m0 sglang_m1 sglang_m2 sglang_m3 sglang_m4 sglang_m6 sglang_m7 sglang_m8 \
  --repeat 20 \
  --output-json "${BENCH_JSON}" \
  2>&1 | tee "${RUN_DIR}/step2_bench.log"

echo ">>> Step 2: Done. Results saved to ${BENCH_JSON}"

# ── Step 3: Analyze — comparison plots + tables ───────────────
echo ""
echo ">>> Step 3: Generating distribution comparison plots + tables"

python "${BENCH_DIR}/analyze_topk_distribution.py" \
  --bench-json "${BENCH_JSON}" \
  --real-histograms "${REAL_HIST_PATH}" \
  --output-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/step3_analyze.log"

echo ">>> Step 3: Done."

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Bucket Distribution Profiling Complete"
echo "  All outputs in: ${RUN_DIR}/"
echo "    bench_distribution.json     — raw benchmark data"
echo "    distribution_comparison.png — bucket dist plots"
echo "    bucket_counts.csv           — per-bucket count table"
echo "    step{1,2,3}_*.log           — pipeline logs"
echo "============================================================"
