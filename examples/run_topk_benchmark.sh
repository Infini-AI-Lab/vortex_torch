#!/usr/bin/env bash
# ============================================================
# TopK Benchmark
#
# Compares ALL TopK kernel variants under controlled conditions:
#   Step 1: Calibrate (for modes 1/2)
#   Step 2: Kernel-level latency (bench_topk.py, all 6 modes)
#   Step 3: E2E accuracy (verify_algo.py)
#           - Full-attention baseline first
#           - Then naive, sglang mode 0/1/2/3/4
#           - Same model, same prompts, deterministic sampling
#
# Fairness improvements over verify_algo_topk_mapping.sh:
#   - Full-attention baseline for absolute reference
#   - All modes in one sweep (including calibrated 1/2)
#   - Sequential runs on same CUDA device minimize interference
#   - Deterministic sampling (temperature=0) for reproducibility
#   - Results saved to a single timestamped directory
#
# Usage:
#   bash run_topk_benchmark.sh [OPTIONS]
#
# Options:
#   --model-name NAME   HuggingFace model (default: Qwen/Qwen3-1.7B)
#   --topk-val K        Top-k value (default: 30)
#   --trials N          E2E trial count (default: 8)
#   --mem FRAC          GPU memory fraction (default: 0.7)
#   --gpu GPU_ID        CUDA device (default: 0)
#   --algo NAME         Sparse attention algorithm (default: block_sparse_attention)
#   --skip-calibrate    Reuse existing calibration data
#   --skip-kernel       Skip kernel-level benchmark (step 2)
#   --skip-e2e          Skip E2E accuracy benchmark (step 3)
# ============================================================
set -euo pipefail

# use GPU_ID to set the GPU id you want to use
GPU_ID=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-1.7B"
TOPK_VAL=30
TRIALS=8
MEM=0.7
ALGO="block_sparse_attention"
SKIP_CALIBRATE=false
SKIP_KERNEL=false
SKIP_E2E=true
BENCHMARKS="amc23"    # space-separated list, e.g. "amc23 aime24"

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)      MODEL_NAME="$2"; shift 2 ;;
    --topk-val)        TOPK_VAL="$2"; shift 2 ;;
    --trials)          TRIALS="$2"; shift 2 ;;
    --mem)             MEM="$2"; shift 2 ;;
    --gpu)             GPU_ID="$2"; shift 2 ;;
    --algo)            ALGO="$2"; shift 2 ;;
    --benchmark)       BENCHMARKS="$2"; shift 2 ;;
    --skip-calibrate)  SKIP_CALIBRATE=true; shift ;;
    --skip-kernel)     SKIP_KERNEL=true; shift ;;
    --skip-e2e)        SKIP_E2E=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCH_LABEL=$(echo "${BENCHMARKS}" | tr ' ' '_')
RUN_DIR="${RESULTS_DIR}/topk_benchmark_${BENCH_LABEL}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "Fair Unified TopK Benchmark"
echo "  Model:     ${MODEL_NAME}"
echo "  Algorithm: ${ALGO}"
echo "  TopK:      ${TOPK_VAL}"
echo "  Trials:    ${TRIALS}"
echo "  GPU:       ${GPU_ID}"
echo "  Output:    ${RUN_DIR}"
echo "============================================================"

# ── Step 1: Calibrate (for modes 1/2) ────────────────────────
CALIBRATION_DIR="${RUN_DIR}/calibration"
if [ "${SKIP_CALIBRATE}" = true ] && [ -d "${CALIBRATION_DIR}" ]; then
  echo ""
  echo ">>> Step 1: SKIPPED (--skip-calibrate)"
else
  echo ""
  echo ">>> Step 1: Calibrating — collecting histograms for LUT/quantile modes"
  mkdir -p "${CALIBRATION_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --mem "${MEM}" \
    --vortex-module-name "${ALGO}" \
    --output-dir "${CALIBRATION_DIR}" \
    2>&1 | tee "${RUN_DIR}/step1_calibrate.log"
  echo ">>> Step 1: Done."
fi

# ── Step 2: Kernel-level latency benchmark ────────────────────
if [ "${SKIP_KERNEL}" = true ]; then
  echo ""
  echo ">>> Step 2: SKIPPED (--skip-kernel)"
else
  # Step 2a: Auto-tune parametric mapping modes (must run before bench)
  echo ""
  echo ">>> Step 2a: Auto-tuning parametric mapping hyperparameters"
  AUTOTUNE_JSON="${RUN_DIR}/autotune_results.json"
  REAL_HIST_ARGS=""
  if [ -f "${CALIBRATION_DIR}/raw_histograms.npy" ]; then
    REAL_HIST_ARGS="--real-histograms ${CALIBRATION_DIR}/raw_histograms.npy"
  fi
  python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --topk-val "${TOPK_VAL}" \
    --batch-size 4 \
    --seq-len 32768 \
    --num-kv-heads 2 \
    ${REAL_HIST_ARGS} \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2a_autotune.log"
  echo ">>> Step 2a: Done. Autotune results saved to ${AUTOTUNE_JSON}"

  # Step 2b: Kernel-level latency + histogram benchmark (using autotune params)
  echo ""
  echo ">>> Step 2b: Kernel-level latency benchmark (all modes)"

  BENCH_JSON="${RUN_DIR}/kernel_latency.json"

  # Build calibration args
  LUT_ARGS=""
  if [ -f "${CALIBRATION_DIR}/lut.npy" ]; then
    LUT_ARGS="--lut-path ${CALIBRATION_DIR}/lut.npy"
  fi
  QUANTILES_ARGS=""
  if [ -f "${CALIBRATION_DIR}/quantiles.npy" ]; then
    QUANTILES_ARGS="--quantiles-path ${CALIBRATION_DIR}/quantiles.npy"
  fi

  python "${BENCH_DIR}/bench_topk.py" \
    --batch-sizes 4 8 16 32 \
    --seq-lens 2048 4096 8192 16384 32768 \
    --topk-vals "${TOPK_VAL}" \
    --num-kv-heads 2 4 \
    --distributions normal lognormal uniform \
    --histogram \
    --hit-rate \
    --warmup 20 \
    --repeat 100 \
    ${LUT_ARGS} \
    ${QUANTILES_ARGS} \
    --autotune-json "${AUTOTUNE_JSON}" \
    --output-json "${BENCH_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2b_kernel_bench.log"

  echo ">>> Step 2b: Done. Results saved to ${BENCH_JSON}"

  # Step 2c: Per-mode distribution analysis
  echo ""
  echo ">>> Step 2c: Generating per-mode distribution analysis"

  python "${BENCH_DIR}/analyze_topk_distribution.py" \
    --bench-json "${BENCH_JSON}" \
    ${REAL_HIST_ARGS} \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/step2c_analyze.log"

  echo ">>> Step 2c: Done. Per-mode plots saved to ${RUN_DIR}"
fi

# ── Step 3: E2E accuracy comparison ──────────────────────────
if [ "${SKIP_E2E}" = true ]; then
  echo ""
  echo ">>> Step 3: SKIPPED (--skip-e2e)"
else
  echo ""
  echo ">>> Step 3: E2E accuracy comparison"

  E2E_DIR="${RUN_DIR}/e2e"
  mkdir -p "${E2E_DIR}"

  # Helper: run verify_algo.py with common args and save output
  run_e2e() {
    local label="$1"
    shift
    local logfile="${E2E_DIR}/${label}.log"
    echo ""
    echo "  --- ${label} ---"
    { time python "${SCRIPT_DIR}/verify_algo.py" \
      --trials "${TRIALS}" \
      --topk-val "${TOPK_VAL}" \
      --model-name "${MODEL_NAME}" \
      --benchmark ${BENCHMARKS} \
      --mem "${MEM}" \
      "$@" ; } \
      2>&1 | tee "${logfile}"
  }

  # 3a. Full-attention baseline (oracle)
  run_e2e "full_attention_baseline" \
    --full-attention

  # 3b. Naive TopK
  run_e2e "naive_mode0" \
    --vortex-module-name "${ALGO}" \
    --topk-type naive

  # 3c. SGLang mode 0 (no mapping)
  run_e2e "sglang_mode0_none" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 0

  # 3d. SGLang mode 1 (LUT CDF) — requires calibration
  if [ -f "${CALIBRATION_DIR}/lut.npy" ]; then
    run_e2e "sglang_mode1_lut_cdf" \
      --vortex-module-name "${ALGO}" \
      --topk-type sglang \
      --topk-mapping-mode 1 \
      --topk-mapping-lut-path "${CALIBRATION_DIR}/lut.npy"
  else
    echo "  --- sglang_mode1_lut_cdf: SKIPPED (no lut.npy) ---"
  fi

  # 3e. SGLang mode 2 (quantile) — requires calibration
  if [ -f "${CALIBRATION_DIR}/quantiles.npy" ]; then
    run_e2e "sglang_mode2_quantile" \
      --vortex-module-name "${ALGO}" \
      --topk-type sglang \
      --topk-mapping-mode 2 \
      --topk-mapping-quantiles-path "${CALIBRATION_DIR}/quantiles.npy"
  else
    echo "  --- sglang_mode2_quantile: SKIPPED (no quantiles.npy) ---"
  fi

  # 3f. SGLang mode 3 (power)
  run_e2e "sglang_mode3_power" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 3 \
    --topk-mapping-power 0.5

  # 3g. SGLang mode 4 (log)
  run_e2e "sglang_mode4_log" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 4

  # 3h. SGLang mode 6 (asinh)
  run_e2e "sglang_mode6_asinh" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 6 \
    --topk-mapping-power 1.0

  # 3i. SGLang mode 7 (log1p)
  run_e2e "sglang_mode7_log1p" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 7 \
    --topk-mapping-power 1.0

  # 3j. SGLang mode 8 (Trunc8)
  run_e2e "sglang_mode8_trunc8" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 8

  # 3k. SGLang mode 9 (Erf)
  run_e2e "sglang_mode9_erf" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 9 \
    --topk-mapping-power 1.0

  # 3l. SGLang mode 10 (Tanh)
  run_e2e "sglang_mode10_tanh" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 10 \
    --topk-mapping-power 1.0

  # 3m. SGLang mode 11 (Subtract)
  run_e2e "sglang_mode11_subtract" \
    --vortex-module-name "${ALGO}" \
    --topk-type sglang \
    --topk-mapping-mode 11

  echo ""
  echo ">>> Step 3: Done. E2E logs saved to ${E2E_DIR}/"

  # ── Summary table: extract pass@N from each log ─────────────
  echo ""
  echo "============================================================"
  echo "E2E Accuracy Summary"
  echo "============================================================"
  printf "%-35s  %s\n" "Configuration" "Result"
  printf "%-35s  %s\n" "-----------------------------------" "------"
  for logfile in "${E2E_DIR}"/*.log; do
    label=$(basename "${logfile}" .log)
    # Extract the last line matching pass@ pattern
    result=$(grep -oP 'pass@\d+\s*[=:]\s*[\d.]+' "${logfile}" | tail -1 || echo "N/A")
    printf "%-35s  %s\n" "${label}" "${result}"
  done
  echo "============================================================"
fi

# ── Final Summary ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "TopK Benchmark Complete"
echo "  All results: ${RUN_DIR}"
echo "  Calibration: ${CALIBRATION_DIR}"
[ "${SKIP_KERNEL}" != true ] && echo "  Kernel JSON: ${RUN_DIR}/kernel_latency.json"
[ "${SKIP_KERNEL}" != true ] && echo "  Per-mode:    ${RUN_DIR}/distribution_comparison_m*.png, bucket_counts_m*.csv"
[ "${SKIP_E2E}" != true ] && echo "  E2E logs:    ${RUN_DIR}/e2e/"
echo "============================================================"
