#!/usr/bin/env bash
# ============================================================
# Bucket Distribution / Remap Latency Pipeline (parametric modes)
#
# Tests the surviving parametric mapping modes after the lean
# refactor:
#   Mode 3 (Power):       y = sign(x) * |x|^p
#   Mode 6 (Asinh):       y = asinh(beta * x)
#   Mode 7 (Log1p):       y = sign(x) * log1p(alpha * |x|)
#   Mode 9 (Erf):         y = erf(alpha * x)
#   Mode 10 (Tanh):       y = tanh(alpha * x)
#   Mode 13 (ExpStretch): y = exp(alpha * x)
#
# Pipeline:
#   1. Calibrate  — collect real-distribution histograms from the
#                   chosen model (skippable via --real-histograms).
#   2. Autotune   — rank per-mode hparams by measured fused-topk
#                   kernel latency (lowest wins).
#   3. Remap bench— bench_topk.py --remap-bench fed with the
#                   autotune JSON. Reports per-mode remap / topk /
#                   fused / baseline latencies and threshold stats.
#
# Usage:
#   bash run_distribution_analysis_new.sh --gpu 5
#   bash run_distribution_analysis_new.sh --gpu 5 \
#       --model-name Qwen/Qwen3-8B --block-size 32
#   bash run_distribution_analysis_new.sh --gpu 5 \
#       --real-histograms /path/to/raw_histograms.npy
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=4
MODEL_NAME="Qwen/Qwen3-1.7B"
TOPK_VAL=2048
MEM=0.7
ALGO="block_sparse_attention"
SEQ_LEN=65536
BLOCK_SIZE=16
BATCH_SIZE=4
NUM_KV_HEADS=8
DISTRIBUTIONS="bucket_uniform normal"
# LUT_CDF (1) / QUANTILE (2) are evaluated only when calibration produces
# lut.npy / quantiles.npy. 0 baseline is always included by --remap-bench.
MAPPING_MODES="1 2 3 6 7 8 9 10 11 13"
REPEAT=100
WARMUP=20
REAL_HISTOGRAMS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)      MODEL_NAME="$2"; shift 2 ;;
    --topk-val)        TOPK_VAL="$2"; shift 2 ;;
    --mem)             MEM="$2"; shift 2 ;;
    --gpu)             GPU_ID="$2"; shift 2 ;;
    --algo)            ALGO="$2"; shift 2 ;;
    --real-histograms) REAL_HISTOGRAMS="$2"; shift 2 ;;
    --seq-len)         SEQ_LEN="$2"; shift 2 ;;
    --block-size|--page-size) BLOCK_SIZE="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-kv-heads)    NUM_KV_HEADS="$2"; shift 2 ;;
    --distributions)   DISTRIBUTIONS="$2"; shift 2 ;;
    --modes)           MAPPING_MODES="$2"; shift 2 ;;
    --repeat)          REPEAT="$2"; shift 2 ;;
    --warmup)          WARMUP="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

MIN_SEQ_LEN=$(( (TOPK_VAL + 4) * BLOCK_SIZE ))
if [ "${SEQ_LEN}" -lt "${MIN_SEQ_LEN}" ]; then
  echo "ERROR: --seq-len ${SEQ_LEN} too small for --topk-val ${TOPK_VAL} @ --block-size ${BLOCK_SIZE}."
  echo "  Minimum: ${MIN_SEQ_LEN}"
  exit 1
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="$(echo "${MODEL_NAME}" | tr '/' '_')"
RUN_DIR="${RESULTS_DIR}/dist_analysis_${MODEL_SLUG}_topk${TOPK_VAL}_bs${BLOCK_SIZE}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "Bucket Distribution / Remap Latency Pipeline (parametric modes)"
echo "  Model:           ${MODEL_NAME}"
echo "  Algorithm:       ${ALGO}"
echo "  TopK:            ${TOPK_VAL}"
echo "  Block size:      ${BLOCK_SIZE}"
echo "  Seq len:         ${SEQ_LEN} ($(( SEQ_LEN / BLOCK_SIZE )) pages/seg)"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  KV heads:        ${NUM_KV_HEADS}"
echo "  Distributions:   ${DISTRIBUTIONS}"
echo "  Mapping modes:   ${MAPPING_MODES}"
echo "  GPU:             ${GPU_ID}"
echo "  Real histograms: ${REAL_HISTOGRAMS:-<will calibrate from ${MODEL_NAME}>}"
echo "  Output:          ${RUN_DIR}"
echo "============================================================"

# ── Step 1: Calibrate ───────────────────────────────────────────
if [ -n "${REAL_HISTOGRAMS}" ]; then
  echo ""
  echo ">>> Step 1: SKIPPED (using provided --real-histograms ${REAL_HISTOGRAMS})"
  REAL_HIST_PATH="${REAL_HISTOGRAMS}"
else
  echo ""
  echo ">>> Step 1: Calibrating ${MODEL_NAME} — collecting real topk histograms"
  CALIBRATION_DIR="${RUN_DIR}/calibration"
  mkdir -p "${CALIBRATION_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --mem "${MEM}" \
    --vortex-module-name "${ALGO}" \
    --page-size "${BLOCK_SIZE}" \
    --output-dir "${CALIBRATION_DIR}" \
    2>&1 | tee "${RUN_DIR}/step1_calibrate.log"
  REAL_HIST_PATH="${CALIBRATION_DIR}/raw_histograms.npy"
  echo ">>> Step 1: Done. Calibration saved to ${CALIBRATION_DIR}"
fi

# Pick up lut.npy / quantiles.npy if calibration produced them.
CALIB_DIR="$(dirname "${REAL_HIST_PATH}")"
LUT_PATH=""
Q_PATH=""
[ -f "${CALIB_DIR}/lut.npy" ]       && LUT_PATH="${CALIB_DIR}/lut.npy"
[ -f "${CALIB_DIR}/quantiles.npy" ] && Q_PATH="${CALIB_DIR}/quantiles.npy"

# ── Step 2: Autotune (latency-ranked) ───────────────────────────
echo ""
echo ">>> Step 2: Auto-tuning hyperparameters by fused-topk kernel latency"
AUTOTUNE_JSON="${RUN_DIR}/autotune_results.json"
AUTOTUNE_EXTRA=()
[ -n "${LUT_PATH}" ] && AUTOTUNE_EXTRA+=(--lut-path "${LUT_PATH}")
[ -n "${Q_PATH}" ]   && AUTOTUNE_EXTRA+=(--quantiles-path "${Q_PATH}")
PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
  --topk-val "${TOPK_VAL}" \
  --batch-size "${BATCH_SIZE}" \
  --num-kv-heads "${NUM_KV_HEADS}" \
  --seq-len "${SEQ_LEN}" \
  --page-size "${BLOCK_SIZE}" \
  --real-histograms "${REAL_HIST_PATH}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --collect-stats \
  "${AUTOTUNE_EXTRA[@]}" \
  --output-json "${AUTOTUNE_JSON}" \
  2>&1 | tee "${RUN_DIR}/step2_autotune.log"
echo ">>> Step 2: Done. Autotune results saved to ${AUTOTUNE_JSON}"

# ── Step 3: Remap bench with autotuned hparams ──────────────────
echo ""
echo ">>> Step 3: Remap benchmark (baseline / fused / remap / split) with autotuned hparams"
BENCH_JSON="${RUN_DIR}/remap_bench.json"
BENCH_EXTRA=()
[ -n "${LUT_PATH}" ] && BENCH_EXTRA+=(--lut-path "${LUT_PATH}")
[ -n "${Q_PATH}" ]   && BENCH_EXTRA+=(--quantiles-path "${Q_PATH}")
PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
  --remap-bench \
  --batch-sizes "${BATCH_SIZE}" \
  --num-kv-heads "${NUM_KV_HEADS}" \
  --seq-lens "${SEQ_LEN}" \
  --topk-vals "${TOPK_VAL}" \
  --page-size "${BLOCK_SIZE}" \
  --distributions ${DISTRIBUTIONS} \
  --mapping-modes ${MAPPING_MODES} \
  --autotune-json "${AUTOTUNE_JSON}" \
  "${BENCH_EXTRA[@]}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --output-json "${BENCH_JSON}" \
  2>&1 | tee "${RUN_DIR}/step3_bench.log"
echo ">>> Step 3: Done. Remap bench saved to ${BENCH_JSON}"

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Bucket Distribution / Remap Latency Pipeline Complete"
echo "  Model:          ${MODEL_NAME}"
echo "  Block size:     ${BLOCK_SIZE}"
echo "  All outputs in: ${RUN_DIR}/"
echo "    calibration/raw_histograms.npy  — real topk distribution"
echo "    autotune_results.json           — latency-ranked hparams"
echo "    remap_bench.json                — remap/topk/fused/baseline latencies"
echo "    step{1,2,3}_*.log               — pipeline logs"
echo "============================================================"
