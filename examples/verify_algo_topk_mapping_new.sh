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
  --seq-len 4096 \
  --num-kv-heads 2 \
  --real-histograms "${REAL_HISTOGRAMS}" \
  --output-json "${AUTOTUNE_JSON}" \
  2>&1 | tee "${RESULTS_DIR}/autotune_${TIMESTAMP}.log"
echo ">>> Auto-tune results saved to ${AUTOTUNE_JSON}"
echo ""

# ============================================================
# Step 1: Mode 3 (power) — sweep p values
# ============================================================
echo "============================================================"
echo "Step 1: Mode 3 (power) — sweeping p"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  for p in 0.1 0.25 0.75 0.9; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_3_p${p}_${TIMESTAMP}.log"
    echo ">>> Mode 3 (power) p=${p} algo=${algo}"
    { time python verify_algo.py \
      --trials 8 \
      --topk-val 30 \
      --vortex-module-name "${algo}" \
      --model-name Qwen/Qwen3-1.7B \
      --topk-type sglang \
      --topk-mapping-mode 3 \
      --topk-mapping-power ${p} \
      --mem 0.7 ; } \
      2>&1 | tee "${OUTFILE}"
  done
done

# ============================================================
# Step 2: Mode 6 (asinh) — sweep beta values
# ============================================================
echo "============================================================"
echo "Step 2: Mode 6 (asinh) — sweeping beta"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  for beta in 0.1 0.5 1.0 2.0 4.0; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_6_beta${beta}_${TIMESTAMP}.log"
    echo ">>> Mode 6 (asinh) beta=${beta} algo=${algo}"
    { time python verify_algo.py \
      --trials 8 \
      --topk-val 30 \
      --vortex-module-name "${algo}" \
      --model-name Qwen/Qwen3-1.7B \
      --topk-type sglang \
      --topk-mapping-mode 6 \
      --topk-mapping-power ${beta} \
      --mem 0.7 ; } \
      2>&1 | tee "${OUTFILE}"
  done
done

# ============================================================
# Step 3: Mode 7 (log1p) — sweep alpha values
# ============================================================
echo "============================================================"
echo "Step 3: Mode 7 (log1p) — sweeping alpha"
echo "============================================================"
for algo in "${sparse_algos[@]}"; do
  for alpha in 0.1 0.5 0.75 1.0 2.0 4.0 8.0; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_7_alpha${alpha}_${TIMESTAMP}.log"
    echo ">>> Mode 7 (log1p) alpha=${alpha} algo=${algo}"
    { time python verify_algo.py \
      --trials 8 \
      --topk-val 30 \
      --vortex-module-name "${algo}" \
      --model-name Qwen/Qwen3-1.7B \
      --topk-type sglang \
      --topk-mapping-mode 7 \
      --topk-mapping-power ${alpha} \
      --mem 0.7 ; } \
      2>&1 | tee "${OUTFILE}"
  done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All sweeps complete. Results in ${RESULTS_DIR}/"
echo "  Auto-tune:  ${AUTOTUNE_JSON}"
echo "  Mode 3 (power):  p   = [0.1, 0.25, 0.75, 0.9]"
echo "  Mode 6 (asinh):  beta  = [0.1, 0.5, 1.0, 2.0, 4.0]"
echo "  Mode 7 (log1p):  alpha = [0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]"
echo "============================================================"
