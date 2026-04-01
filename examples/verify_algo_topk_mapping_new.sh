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
export CUDA_VISIBLE_DEVICES=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

sparse_algos=(
  "block_sparse_attention"
)

# Path to real-data histograms from calibration (for auto-tuning)
REAL_HISTOGRAMS="/scr/dataset/yuke/xinrui/new/vortex_torch/examples/calibration/raw_histograms.npy"

RESULTS_DIR="results"
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
  --topk-val 30 \
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
    if m in (3, 6, 7, 9, 10):
        if m not in best or r['gini'] < best[m]['gini']:
            best[m] = r
for m in (3, 6, 7, 9, 10):
    print(f'BEST_POWER_{m}={best[m][\"param\"]}' if m in best else f'BEST_POWER_{m}=0.5')
" "${AUTOTUNE_JSON}")"
echo ">>> Autotuned best powers: mode3=${BEST_POWER_3} mode6=${BEST_POWER_6} mode7=${BEST_POWER_7} mode9=${BEST_POWER_9} mode10=${BEST_POWER_10}"
echo ""

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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 3 \
    --topk-mapping-power ${BEST_POWER_3} \
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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 6 \
    --topk-mapping-power ${BEST_POWER_6} \
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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 7 \
    --topk-mapping-power ${BEST_POWER_7} \
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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 8 \
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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 9 \
    --topk-mapping-power ${BEST_POWER_9} \
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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 10 \
    --topk-mapping-power ${BEST_POWER_10} \
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
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 11 \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All runs complete. Results in ${RESULTS_DIR}/"
echo "  Auto-tune:   ${AUTOTUNE_JSON}"
echo "  Mode 3 (power):    p     = ${BEST_POWER_3} (autotuned)"
echo "  Mode 6 (asinh):    beta  = ${BEST_POWER_6} (autotuned)"
echo "  Mode 7 (log1p):    alpha = ${BEST_POWER_7} (autotuned)"
echo "  Mode 8 (trunc8):   (fixed)"
echo "  Mode 9 (erf):      alpha = ${BEST_POWER_9} (autotuned)"
echo "  Mode 10 (tanh):    alpha = ${BEST_POWER_10} (autotuned)"
echo "  Mode 11 (subtract): (fixed)"
echo "============================================================"
