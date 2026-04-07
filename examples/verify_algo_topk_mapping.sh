#!/usr/bin/env bash
set -e
# use CUDA_VISIBLE_DEVICES to set the GPU id you want to use
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
GPU_ID=0
BENCHMARKS="amc23"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU_ID="$2"; shift 2 ;;
    --benchmark)  BENCHMARKS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

sparse_algos=(
  "block_sparse_attention"
)

BENCH_LABEL=$(echo "${BENCHMARKS}" | tr ' ' '_')
RESULTS_DIR="results/${BENCH_LABEL}"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Set this to an existing calibration directory to skip re-running calibration.
# It must contain lut.npy and quantiles.npy (output of calibrate_topk.py).
CALIBRATION_DIR="/data/datasets/xinrui/My_Projects/vortex_torch/examples/calibration"
# ============================================================
# Baseline: naive topk (mode 0)
# ============================================================
for algo in "${sparse_algos[@]}"; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_naive_${TIMESTAMP}.log"
    echo ">>> Running verify_algo.py with --vortex-module-name ${algo} --topk-type naive --topk-mapping-mode 0"
    echo ">>> Saving results to ${OUTFILE}"
    { time python verify_algo.py \
      --trials 8 \
      --topk-val 30 \
      --vortex-module-name "${algo}" \
      --model-name Qwen/Qwen3-1.7B \
      --topk-type naive \
      --topk-mapping-mode 0 \
      --benchmark ${BENCHMARKS} \
      --mem 0.7 ; } \
     2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Calibration: collect histograms for LUT/quantile generation
# Skipped if CALIBRATION_DIR already has lut.npy + quantiles.npy
# ============================================================
if [ -f "${CALIBRATION_DIR}/lut.npy" ] && [ -f "${CALIBRATION_DIR}/quantiles.npy" ]; then
    echo ">>> Calibration SKIPPED (using existing ${CALIBRATION_DIR})"
else
    CALIBRATION_DIR="${RESULTS_DIR}/calibration_${TIMESTAMP}"
    for algo in "${sparse_algos[@]}"; do
        echo ">>> Calibrating for ${algo}..."
        python "${BENCH_DIR}/calibrate_topk.py" \
          --model-name Qwen/Qwen3-1.7B \
          --topk-val 30 \
          --mem 0.7 \
          --vortex-module-name "${algo}" \
          --output-dir "${CALIBRATION_DIR}" \
          2>&1 | tee "${RESULTS_DIR}/calibration_${algo}_${TIMESTAMP}.log"
    done
fi

# ============================================================
# Auto-tune: find best hyperparameters per mode
# Uses topk_profile_histogram kernel on real calibration data
# ============================================================
REAL_HISTOGRAMS="${CALIBRATION_DIR}/raw_histograms.npy"
if [ -f "${REAL_HISTOGRAMS}" ]; then
  echo "============================================================"
  echo "Auto-tuning hyperparameters (real calibration data)"
  echo "============================================================"
  AUTOTUNE_JSON="${RESULTS_DIR}/autotune_${TIMESTAMP}.json"
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --topk-val 30 \
    --batch-size 4 \
    --seq-len 32768 \
    --num-kv-heads 2 \
    --real-histograms "${REAL_HISTOGRAMS}" \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RESULTS_DIR}/autotune_${TIMESTAMP}.log"
  echo ">>> Auto-tune results saved to ${AUTOTUNE_JSON}"
  echo ""

  # Extract best per-mode hyperparameters from autotune JSON
  eval "$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
best = {}
for r in data:
    m = r.get('mode')
    if m in (3, 6, 7, 9, 10):
        if m not in best or r['gini'] < best[m]['gini']:
            best[m] = r
for m in (3, 6, 7, 9, 10):
    print(f'BEST_HPARAM_{m}={best[m][\"param\"]}' if m in best else f'BEST_HPARAM_{m}=0.5')
" "${AUTOTUNE_JSON}")"
  echo ">>> Autotuned best powers: mode3=${BEST_HPARAM_3} mode6=${BEST_HPARAM_6} mode7=${BEST_HPARAM_7} mode9=${BEST_HPARAM_9} mode10=${BEST_HPARAM_10}"
  echo ""
else
  echo ">>> WARNING: ${REAL_HISTOGRAMS} not found, using default power=0.5 for all modes"
  BEST_HPARAM_3=0.5
  BEST_HPARAM_6=0.5
  BEST_HPARAM_7=0.5
  BEST_HPARAM_9=0.5
  BEST_HPARAM_10=0.5
fi

# ============================================================
# Mode 1: LUT CDF with calibrated LUT
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_1_calibrated_${TIMESTAMP}.log"
  echo ">>> Running mode 1 (LUT CDF) with calibrated LUT for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 1 \
    --topk-mapping-lut-path "${CALIBRATION_DIR}/lut.npy" \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Mode 2: Quantile with calibrated quantiles
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_2_calibrated_${TIMESTAMP}.log"
  echo ">>> Running mode 2 (quantile) with calibrated quantiles for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 2 \
    --topk-mapping-quantiles-path "${CALIBRATION_DIR}/quantiles.npy" \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# sglang topk: non-parametric modes (0, 4, 8, 11)
# ============================================================
for algo in "${sparse_algos[@]}"; do
  for topk_mapping_mode in 0 4 8 11; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_${topk_mapping_mode}_${TIMESTAMP}.log"
    echo ">>> Running verify_algo.py with --vortex-module-name ${algo} --topk-type sglang --topk-mapping-mode ${topk_mapping_mode}"
    echo ">>> Saving results to ${OUTFILE}"

    { time python verify_algo.py \
      --trials 8 \
      --topk-val 30 \
      --vortex-module-name "${algo}" \
      --model-name Qwen/Qwen3-1.7B \
      --topk-type sglang \
      --topk-mapping-mode ${topk_mapping_mode} \
      --benchmark ${BENCHMARKS} \
      --mem 0.7 ; } \
      2>&1 | tee "${OUTFILE}"
  done
done

# ============================================================
# Mode 3: power — autotuned best p
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_3_p${BEST_HPARAM_3}_${TIMESTAMP}.log"
  echo ">>> Running mode 3 (power) p=${BEST_HPARAM_3} (autotuned) for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 3 \
    --topk-mapping-hparam ${BEST_HPARAM_3} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Mode 6: asinh — autotuned best beta
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_6_beta${BEST_HPARAM_6}_${TIMESTAMP}.log"
  echo ">>> Running mode 6 (asinh) beta=${BEST_HPARAM_6} (autotuned) for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 6 \
    --topk-mapping-hparam ${BEST_HPARAM_6} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Mode 7: log1p — autotuned best alpha
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_7_alpha${BEST_HPARAM_7}_${TIMESTAMP}.log"
  echo ">>> Running mode 7 (log1p) alpha=${BEST_HPARAM_7} (autotuned) for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 7 \
    --topk-mapping-hparam ${BEST_HPARAM_7} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Mode 9: erf — autotuned best alpha
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_9_alpha${BEST_HPARAM_9}_${TIMESTAMP}.log"
  echo ">>> Running mode 9 (erf) alpha=${BEST_HPARAM_9} (autotuned) for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 9 \
    --topk-mapping-hparam ${BEST_HPARAM_9} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Mode 10: tanh — autotuned best alpha
# ============================================================
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_10_alpha${BEST_HPARAM_10}_${TIMESTAMP}.log"
  echo ">>> Running mode 10 (tanh) alpha=${BEST_HPARAM_10} (autotuned) for ${algo}"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 10 \
    --topk-mapping-hparam ${BEST_HPARAM_10} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Counter profiling: collect COUNTER_NUM_EQUAL for all modes
# ============================================================
echo ""
echo "============================================================"
echo "Counter profiling: COUNTER_NUM_EQUAL per mode (topk=30)"
echo "============================================================"
COUNTER_JSON="${RESULTS_DIR}/counters_${TIMESTAMP}.json"
PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
  --batch-sizes 4 \
  --seq-lens 4096 \
  --topk-vals 30 \
  --num-kv-heads 2 \
  --distributions normal \
  --counters \
  --filter-kernels sglang_ori sglang_m0 sglang_m3 sglang_m6 sglang_m7 sglang_m8 sglang_m9 sglang_m10 sglang_m11 \
  --repeat 5 \
  --output-json "${COUNTER_JSON}" \
  2>&1 | tee "${RESULTS_DIR}/counters_${TIMESTAMP}.log"
echo ">>> Counters saved to ${COUNTER_JSON}"