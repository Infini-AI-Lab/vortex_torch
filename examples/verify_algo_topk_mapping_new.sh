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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=5
TOPK_VAL=30
BENCHMARKS="amc23"    # space-separated list, e.g. "amc23 aime24"

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topk-val)   TOPK_VAL="$2"; shift 2 ;;
    --gpu)        GPU_ID="$2"; shift 2 ;;
    --benchmark)  BENCHMARKS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

sparse_algos=(
  "block_sparse_attention"
)

# Path to real-data histograms from calibration (for auto-tuning)
REAL_HISTOGRAMS="/data/datasets/xinrui/My_Projects/vortex_torch/examples/calibration/raw_histograms.npy"

BENCH_LABEL=$(echo "${BENCHMARKS}" | tr ' ' '_')
RESULTS_DIR="results/topk${TOPK_VAL}_${BENCH_LABEL}"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Step 0: Auto-tune — find best hyperparameters per mode
# Uses topk_profile_histogram kernel on synthetic data (fast, no model)
# ============================================================
echo "============================================================"
echo "Step 0: Auto-tuning hyperparameters (synthetic data)"
echo "============================================================"
AUTOTUNE_JSON="${RESULTS_DIR}/autotune_${TIMESTAMP}.json"
PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
  --topk-val ${TOPK_VAL} \
  --batch-size 4 \
  --seq-len 32768 \
  --num-kv-heads 2 \
  --real-histograms "${REAL_HISTOGRAMS}" \
  --output-json "${AUTOTUNE_JSON}" \
  2>&1 | tee "${RESULTS_DIR}/autotune_${TIMESTAMP}.log"
echo ">>> Auto-tune results saved to ${AUTOTUNE_JSON}"
echo ""

# ============================================================
# Extract best per-mode hyperparameters from autotune JSON
# ============================================================
eval "$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
best = {}
for r in data:
    m = r.get('mode')
    if m in (3, 6, 7, 9, 10, 13, 14):
        if m not in best or r['gini'] < best[m]['gini']:
            best[m] = r
for m in (3, 6, 7, 9, 10, 13, 14):
    print(f'BEST_POWER_{m}={best[m][\"param\"]}' if m in best else f'BEST_POWER_{m}=0.5')
" "${AUTOTUNE_JSON}")"
echo ">>> Autotuned best powers: mode3=${BEST_POWER_3} mode6=${BEST_POWER_6} mode7=${BEST_POWER_7} mode9=${BEST_POWER_9} mode10=${BEST_POWER_10} mode13=${BEST_POWER_13} mode14=${BEST_POWER_14}"
echo ""

# ============================================================
# Baseline: Original sglang kernel (no remap)
# ============================================================
echo "============================================================"
echo "Baseline: sglang_ori (no remap)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_ori_${TIMESTAMP}.log"
  echo ">>> sglang_ori algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang_ori \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 1: Mode 3 (power) — autotuned best p
# ============================================================
echo "============================================================"
echo "Step 1: Mode 3 (power) — p=${BEST_POWER_3} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_3_p${BEST_POWER_3}_${TIMESTAMP}.log"
  echo ">>> Mode 3 (power) p=${BEST_POWER_3} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 3 \
    --topk-mapping-power ${BEST_POWER_3} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 2: Mode 6 (asinh) — autotuned best beta
# ============================================================
echo "============================================================"
echo "Step 2: Mode 6 (asinh) — beta=${BEST_POWER_6} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_6_beta${BEST_POWER_6}_${TIMESTAMP}.log"
  echo ">>> Mode 6 (asinh) beta=${BEST_POWER_6} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 6 \
    --topk-mapping-power ${BEST_POWER_6} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 3: Mode 7 (log1p) — autotuned best alpha
# ============================================================
echo "============================================================"
echo "Step 3: Mode 7 (log1p) — alpha=${BEST_POWER_7} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_7_alpha${BEST_POWER_7}_${TIMESTAMP}.log"
  echo ">>> Mode 7 (log1p) alpha=${BEST_POWER_7} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 7 \
    --topk-mapping-power ${BEST_POWER_7} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 4: Mode 8 (trunc8) — fixed parameter
# ============================================================
echo "============================================================"
echo "Step 4: Mode 8 (trunc8)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_8_${TIMESTAMP}.log"
  echo ">>> Mode 8 (trunc8) algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 8 \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 5: Mode 9 (erf) — autotuned best alpha
# ============================================================
echo "============================================================"
echo "Step 5: Mode 9 (erf) — alpha=${BEST_POWER_9} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_9_alpha${BEST_POWER_9}_${TIMESTAMP}.log"
  echo ">>> Mode 9 (erf) alpha=${BEST_POWER_9} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 9 \
    --topk-mapping-power ${BEST_POWER_9} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 6: Mode 10 (tanh) — autotuned best alpha
# ============================================================
echo "============================================================"
echo "Step 6: Mode 10 (tanh) — alpha=${BEST_POWER_10} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_10_alpha${BEST_POWER_10}_${TIMESTAMP}.log"
  echo ">>> Mode 10 (tanh) alpha=${BEST_POWER_10} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 10 \
    --topk-mapping-power ${BEST_POWER_10} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 7: Mode 11 (subtract) — fixed parameter
# ============================================================
echo "============================================================"
echo "Step 7: Mode 11 (subtract)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_11_${TIMESTAMP}.log"
  echo ">>> Mode 11 (subtract) algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 11 \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 8: Mode 12 (adaptive_tail_window), rho=4.0
# ============================================================
echo ""
echo "============================================================"
echo "Step 8: Mode 12 (adaptive_tail_window), rho=4.0"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_12_${TIMESTAMP}.log"
  echo ">>> Mode 12 (adaptive_tail_window) algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 12 \
    --topk-mapping-power 4.0 \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 9: Mode 13 (exp_stretch) — autotuned best alpha
# ============================================================
echo ""
echo "============================================================"
echo "Step 9: Mode 13 (exp_stretch) — alpha=${BEST_POWER_13} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_13_alpha${BEST_POWER_13}_${TIMESTAMP}.log"
  echo ">>> Mode 13 (exp_stretch) alpha=${BEST_POWER_13} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 13 \
    --topk-mapping-power ${BEST_POWER_13} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Step 10: Mode 14 (topk_window) — autotuned best rho
# ============================================================
echo ""
echo "============================================================"
echo "Step 10: Mode 14 (topk_window) — rho=${BEST_POWER_14} (autotuned)"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_14_rho${BEST_POWER_14}_${TIMESTAMP}.log"
  echo ">>> Mode 14 (topk_window) rho=${BEST_POWER_14} algo=${algo}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val ${TOPK_VAL} \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 14 \
    --topk-mapping-power ${BEST_POWER_14} \
    --benchmark ${BENCHMARKS} \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Counter profiling: collect COUNTER_NUM_EQUAL for all modes
# (single extra kernel call per mode, no overhead on accuracy runs)
# ============================================================
echo ""
echo "============================================================"
echo "Counter profiling: COUNTER_NUM_EQUAL per mode (topk=${TOPK_VAL})"
echo "============================================================"
COUNTER_JSON="${RESULTS_DIR}/counters_${TIMESTAMP}.json"
PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
  --batch-sizes 4 \
  --seq-lens 4096 \
  --topk-vals ${TOPK_VAL} \
  --num-kv-heads 2 \
  --distributions normal \
  --counters \
  --real-histograms "${REAL_HISTOGRAMS}" \
  --autotune-json "${AUTOTUNE_JSON}" \
  --filter-kernels sglang_ori sglang_m0 sglang_m3 sglang_m6 sglang_m7 sglang_m8 sglang_m9 sglang_m10 sglang_m11 sglang_m13 sglang_m14 \
  --mapping-power-13 ${BEST_POWER_13} --mapping-power-14 ${BEST_POWER_14} \
  --repeat 5 \
  --output-json "${COUNTER_JSON}" \
  2>&1 | tee "${RESULTS_DIR}/counters_${TIMESTAMP}.log"
echo ">>> Counters saved to ${COUNTER_JSON}"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All runs complete. Results in ${RESULTS_DIR}/"
echo "  Auto-tune:   ${AUTOTUNE_JSON}"
echo "  Counters:    ${COUNTER_JSON}"
echo "  Mode 3 (power):       p     = ${BEST_POWER_3} (autotuned)"
echo "  Mode 6 (asinh):       beta  = ${BEST_POWER_6} (autotuned)"
echo "  Mode 7 (log1p):       alpha = ${BEST_POWER_7} (autotuned)"
echo "  Mode 8 (trunc8):      (fixed)"
echo "  Mode 9 (erf):         alpha = ${BEST_POWER_9} (autotuned)"
echo "  Mode 10 (tanh):       alpha = ${BEST_POWER_10} (autotuned)"
echo "  Mode 11 (subtract):   (fixed)"
echo "  Mode 12 (tail_win):   rho   = 4.0"
echo "  Mode 13 (exp_stretch):alpha = ${BEST_POWER_13} (autotuned)"
echo "  Mode 14 (topk_window):rho   = ${BEST_POWER_14} (autotuned)"
echo "============================================================"
