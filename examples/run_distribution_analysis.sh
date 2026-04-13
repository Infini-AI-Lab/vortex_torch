#!/usr/bin/env bash
# ============================================================
# Bucket Distribution Profiling Pipeline
#
# Profiles the SGLang TopK kernel's first-pass bucket distribution
# to identify hotspot buckets causing tail latency.
#
# Four steps:
#   1. Calibrate  — collect real-data histograms
#                   (skippable via --real-histograms PATH)
#   2. Auto-tune  — sweep hyperparameters to find best per-mode power
#   3. Bench      — histogram profiling (bucket_uniform + normal)
#                   noscale kernels use the same autotuned power
#   4. Analyze    — comparison plots + bucket count tables
#
# All outputs (JSON, plots, CSV tables, logs) are written to a
# single timestamped folder under examples/results/dist_analysis_*.
#
# Usage:
#   bash run_distribution_analysis.sh --gpu 5
#   bash run_distribution_analysis.sh --gpu 5 \
#       --real-histograms /path/to/calibration_dir/raw_histograms.npy
#   bash run_distribution_analysis.sh --gpu 5 --block-size 16
#   bash run_distribution_analysis.sh --watchdog-timeout 0   # disable calibrate watchdog (fork)
# Models (default: 1.7B + 4B). Override with repeated --model-name:
#   bash run_distribution_analysis.sh --model-name Qwen/Qwen3-1.7B --model-name Qwen/Qwen3-4B
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
# 9: Erf            — y = erf(alpha * x)
# 10: Tanh          — y = tanh(alpha * x)
# 11: Subtract      — x - pivot (RadiK-style scatter)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=7
# Models to run (full pipeline per model). Override with one or more --model-name.
MODEL_NAMES=( "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" )
MODEL_NAMES_USER_SET=0
TOPK_VAL=30
MEM=0.7
ALGO="block_sparse_attention"
RADIX_BITS=8
SAMPLE_STRIDE=1
SEQ_LEN=32768
# KV page / block size (passed to benchmarks as --page-size)
BLOCK_SIZE=16
# The path to the raw_histograms.npy file (set to skip calibration)
REAL_HISTOGRAMS="/data/datasets/xinrui/My_Projects/vortex_torch/examples/calibration/raw_histograms.npy"
REAL_HISTOGRAMS=""
HAS_WATCHDOG_TIMEOUT=0
WATCHDOG_TIMEOUT=""
# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      if [ "${MODEL_NAMES_USER_SET}" -eq 0 ]; then
        MODEL_NAMES=()
        MODEL_NAMES_USER_SET=1
      fi
      MODEL_NAMES+=("$2")
      shift 2
      ;;
    --topk-val)         TOPK_VAL="$2"; shift 2 ;;
    --mem)              MEM="$2"; shift 2 ;;
    --gpu)              GPU_ID="$2"; shift 2 ;;
    --algo)             ALGO="$2"; shift 2 ;;
    --real-histograms)  REAL_HISTOGRAMS="$2"; shift 2 ;;
    --radix-bits)       RADIX_BITS="$2"; shift 2 ;;
    --sample-stride)    SAMPLE_STRIDE="$2"; shift 2 ;;
    --seq-len)          SEQ_LEN="$2"; shift 2 ;;
    --block-size)       BLOCK_SIZE="$2"; shift 2 ;;
    --watchdog-timeout) HAS_WATCHDOG_TIMEOUT=1; WATCHDOG_TIMEOUT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
  echo "ERROR: No models in MODEL_NAMES; pass at least one --model-name."
  exit 1
fi

# Validate seq_len: need pages/seg > topk_val (reserved=3 pages + slack)
MIN_SEQ_LEN=$(( (TOPK_VAL + 4) * BLOCK_SIZE ))
if [ "${SEQ_LEN}" -lt "${MIN_SEQ_LEN}" ]; then
  echo "ERROR: --seq-len ${SEQ_LEN} too small for --topk-val ${TOPK_VAL}."
  echo "  Minimum: ${MIN_SEQ_LEN} (pages/seg must exceed topk_val + 3 reserved pages)"
  exit 1
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Bucket Distribution Profiling Pipeline"
echo "  Models (${#MODEL_NAMES[@]}):  ${MODEL_NAMES[*]}"
echo "  Algorithm:       ${ALGO}"
echo "  TopK:            ${TOPK_VAL}"
echo "  Seq len:         ${SEQ_LEN} ($(( SEQ_LEN / BLOCK_SIZE )) pages/seg)"
echo "  Block size:      ${BLOCK_SIZE} (--page-size in benchmarks)"
echo "  GPU:             ${GPU_ID}"
echo "  Radix bits:      ${RADIX_BITS} ($(( 1 << RADIX_BITS )) bins)"
echo "  Sample stride:   ${SAMPLE_STRIDE}"
if [ "${HAS_WATCHDOG_TIMEOUT}" -eq 1 ]; then
  echo "  Watchdog (cal):  ${WATCHDOG_TIMEOUT}s (0 = off, vortex SGLang fork)"
else
  echo "  Watchdog (cal):  <default 300s>"
fi
echo "  Real histograms: ${REAL_HISTOGRAMS:-<will calibrate per model>}"
echo "  Run id:          ${TIMESTAMP}"
echo "  Output root:     ${RESULTS_DIR}/dist_analysis_<model>_${TIMESTAMP}/"
echo "============================================================"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  MODEL_SLUG="${MODEL_NAME//\//_}"
  RUN_DIR="${RESULTS_DIR}/dist_analysis_${MODEL_SLUG}_${TIMESTAMP}"
  mkdir -p "${RUN_DIR}"

  echo ""
  echo "############################ MODEL: ${MODEL_NAME} ############################"
  echo "  Output: ${RUN_DIR}"

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
    CALIB_EXTRA_ARGS=()
    if [ "${HAS_WATCHDOG_TIMEOUT}" -eq 1 ]; then
      CALIB_EXTRA_ARGS+=(--watchdog-timeout "${WATCHDOG_TIMEOUT}")
    fi
    python "${BENCH_DIR}/calibrate_topk.py" \
      --model-name "${MODEL_NAME}" \
      --topk-val "${TOPK_VAL}" \
      --mem "${MEM}" \
      --vortex-module-name "${ALGO}" \
      --page-size "${BLOCK_SIZE}" \
      --output-dir "${CALIBRATION_DIR}" \
      "${CALIB_EXTRA_ARGS[@]}" \
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
  [ -n "${LUT_PATH}" ] && echo "  Calibration LUT:      ${LUT_PATH}"
  [ -n "${Q_PATH}" ]   && echo "  Calibration quantile: ${Q_PATH}"

  # ── Step 2: Auto-tune — rank by fused-topk kernel latency ──────
  echo ""
  echo ">>> Step 2: Auto-tuning hyperparameters by fused-topk kernel latency"

  AUTOTUNE_JSON="${RUN_DIR}/autotune_results.json"
  AUTOTUNE_EXTRA=(--real-histograms "${REAL_HIST_PATH}")
  [ -n "${LUT_PATH}" ] && AUTOTUNE_EXTRA+=(--lut-path "${LUT_PATH}")
  [ -n "${Q_PATH}" ]   && AUTOTUNE_EXTRA+=(--quantiles-path "${Q_PATH}")

  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --topk-val "${TOPK_VAL}" \
    --batch-size 4 \
    --seq-len ${SEQ_LEN} \
    --page-size "${BLOCK_SIZE}" \
    --num-kv-heads 2 \
    --collect-stats \
    "${AUTOTUNE_EXTRA[@]}" \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2_autotune.log"

  echo ">>> Step 2: Done. Autotune results saved to ${AUTOTUNE_JSON}"

  # ── Step 3: Remap benchmark with autotuned hparams ──────────────
  echo ""
  echo ">>> Step 3: Remap benchmark (baseline / fused / remap / split) with autotuned hparams"

  BENCH_JSON="${RUN_DIR}/remap_bench.json"
  BENCH_EXTRA=()
  [ -n "${LUT_PATH}" ] && BENCH_EXTRA+=(--lut-path "${LUT_PATH}")
  [ -n "${Q_PATH}" ]   && BENCH_EXTRA+=(--quantiles-path "${Q_PATH}")

  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
    --remap-bench \
    --batch-sizes 4 \
    --num-kv-heads 8 \
    --seq-lens ${SEQ_LEN} \
    --topk-vals "${TOPK_VAL}" \
    --page-size "${BLOCK_SIZE}" \
    --distributions bucket_uniform normal \
    --mapping-modes 0 1 2 3 6 7 8 9 10 11 13 \
    --autotune-json "${AUTOTUNE_JSON}" \
    "${BENCH_EXTRA[@]}" \
    --repeat 20 \
    --output-json "${BENCH_JSON}" \
    2>&1 | tee "${RUN_DIR}/step3_bench.log"

  echo ">>> Step 3: Done. Remap bench saved to ${BENCH_JSON}"
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Bucket Distribution Profiling Complete"
echo "  Per-model outputs under ${RESULTS_DIR}/ (run id ${TIMESTAMP}):"
echo "    dist_analysis_<model>_${TIMESTAMP}/"
echo "      autotune_results.json, bench_distribution.json, plots, CSV, logs"
echo "============================================================"
