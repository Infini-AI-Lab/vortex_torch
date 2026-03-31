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
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

sparse_algos=(
  "block_sparse_attention"
)

topk_mapping_modes=(
  0 # none
  3 # power
  4 # log
)
RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Set this to an existing calibration directory to skip re-running calibration.
# It must contain lut.npy and quantiles.npy (output of calibrate_topk.py).
CALIBRATION_DIR="/scr/dataset/yuke/xinrui/new/vortex_torch/examples/calibration"

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
# sglang topk: modes that don't need calibration (0, 3, 4)
# ============================================================
for algo in "${sparse_algos[@]}"; do
  for topk_mapping_mode in "${topk_mapping_modes[@]}"; do
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
      --topk-mapping-power 0.5 \
      --mem 0.7 ; } \
      2>&1 | tee "${OUTFILE}"
  done
done

# ============================================================
# Mode 6: asinh — sweep beta values
# ============================================================
for algo in "${sparse_algos[@]}"; do
  for beta in 0.5 1.0 2.0; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_6_beta${beta}_${TIMESTAMP}.log"
    echo ">>> Running mode 6 (asinh) beta=${beta} for ${algo}"
    echo ">>> Saving results to ${OUTFILE}"
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
# Mode 7: log1p — sweep alpha values
# ============================================================
for algo in "${sparse_algos[@]}"; do
  for alpha in 0.5 1.0 2.0; do
    OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_7_alpha${alpha}_${TIMESTAMP}.log"
    echo ">>> Running mode 7 (log1p) alpha=${alpha} for ${algo}"
    echo ">>> Saving results to ${OUTFILE}"
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