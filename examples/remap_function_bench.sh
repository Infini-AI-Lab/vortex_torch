#!/usr/bin/env bash
# ============================================================
# Remap Function Benchmark
#
# Compares four kernel configurations for TopK page selection:
#   1. baseline                    — unmapped topk (topk_output_sglang)
#   2. fused remap + topk          — topk_output_sglang_fused
#   3. remap only                  — topk_remap_only (standalone kernel)
#   4. unmapped topk on remapped   — topk_output_sglang on the output
#                                    buffer of step 3
#
# Per configuration the script also reports the threshold-bin
# position, the threshold-bin size, and how many values are
# selected from the threshold bin (derived from
# topk_profile_counters — collected after all timing measurements,
# never interleaved with latency measurements).
#
# Pipeline:
#   1. Calibrate  — run `calibrate_topk.py` on the chosen model to
#                   collect the REAL per-segment topk distribution
#                   (raw_histograms.npy). Skippable via
#                   --real-histograms /path/to/raw_histograms.npy.
#   2. Autotune   — run `autotune_topk_mapping.py` on those real
#                   histograms and pick the per-mode hyperparameter
#                   with the LOWEST measured topk kernel latency.
#   3. Remap bench— run `bench_topk.py --remap-bench` with the
#                   autotune-selected per-mode hyperparameters.
#
# Argument layout mirrors run_distribution_analysis_new.sh.
#
# Usage:
#   # Default (Qwen/Qwen3-1.7B, block_size=16):
#   bash remap_function_bench.sh --gpu 5
#
#   # Larger model + larger page/block size:
#   bash remap_function_bench.sh --gpu 0 \
#       --model-name Qwen/Qwen3-8B \
#       --block-size 32 \
#       --seq-len 16384 --topk-val 512 \
#       --modes "0 3 6 7"
#
#   # Reuse an existing calibration:
#   bash remap_function_bench.sh --gpu 0 \
#       --model-name Qwen/Qwen3-8B \
#       --real-histograms /path/to/calibration/raw_histograms.npy
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
SAMPLE_STRIDE=1
SEQ_LEN=65536
BLOCK_SIZE=16
BATCH_SIZE=4
NUM_KV_HEADS=8
DISTRIBUTIONS="normal bucket_uniform"
# Modes 1 (LUT_CDF) and 2 (Quantile) are evaluated only if calibration
# produces lut.npy / quantiles.npy. The shell script detects that below.
MAPPING_MODES="0 1 2 3 6 7 8 9 10 11 13"
# Fallback hparam used only if autotune is explicitly skipped.
MAPPING_HPARAM=0.5
REPEAT=100
WARMUP=20
# Empty by default — Step 1 will calibrate on the selected model.
# Pass --real-histograms /path/to/raw_histograms.npy to skip calibration.
REAL_HISTOGRAMS=""
SKIP_AUTOTUNE=0

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)       MODEL_NAME="$2"; shift 2 ;;
    --topk-val)         TOPK_VAL="$2"; shift 2 ;;
    --mem)              MEM="$2"; shift 2 ;;
    --gpu)              GPU_ID="$2"; shift 2 ;;
    --algo)             ALGO="$2"; shift 2 ;;
    --real-histograms)  REAL_HISTOGRAMS="$2"; shift 2 ;;
    --sample-stride)    SAMPLE_STRIDE="$2"; shift 2 ;;
    --seq-len)          SEQ_LEN="$2"; shift 2 ;;
    --block-size|--page-size) BLOCK_SIZE="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    --num-kv-heads)     NUM_KV_HEADS="$2"; shift 2 ;;
    --distributions)    DISTRIBUTIONS="$2"; shift 2 ;;
    --modes)            MAPPING_MODES="$2"; shift 2 ;;
    --mapping-hparam)   MAPPING_HPARAM="$2"; shift 2 ;;
    --repeat)           REPEAT="$2"; shift 2 ;;
    --warmup)           WARMUP="$2"; shift 2 ;;
    --skip-autotune)    SKIP_AUTOTUNE=1; shift 1 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Validate seq_len: need pages/seg > topk_val (3 reserved pages)
MIN_SEQ_LEN=$(( (TOPK_VAL + 4) * BLOCK_SIZE ))
if [ "${SEQ_LEN}" -lt "${MIN_SEQ_LEN}" ]; then
  echo "ERROR: --seq-len ${SEQ_LEN} too small for --topk-val ${TOPK_VAL} @ --block-size ${BLOCK_SIZE}."
  echo "  Minimum: ${MIN_SEQ_LEN} (pages/seg must exceed topk_val + 3 reserved pages)"
  exit 1
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="$(echo "${MODEL_NAME}" | tr '/' '_')"
RUN_DIR="${RESULTS_DIR}/remap_bench_${MODEL_SLUG}_topk${TOPK_VAL}_bs${BLOCK_SIZE}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "Remap Function Benchmark"
echo "  Model:           ${MODEL_NAME}"
echo "  Algorithm:       ${ALGO}"
echo "  TopK:            ${TOPK_VAL}"
echo "  Block size:      ${BLOCK_SIZE}"
echo "  Seq len:         ${SEQ_LEN} ($(( SEQ_LEN / BLOCK_SIZE )) pages/seg)"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  KV heads:        ${NUM_KV_HEADS}"
echo "  Distributions:   ${DISTRIBUTIONS}"
echo "  Mapping modes:   ${MAPPING_MODES}"
echo "  Fallback hparam: ${MAPPING_HPARAM}  (used only when --skip-autotune)"
echo "  GPU:             ${GPU_ID}"
echo "  Sample stride:   ${SAMPLE_STRIDE}"
echo "  Real histograms: ${REAL_HISTOGRAMS:-<will calibrate from ${MODEL_NAME}>}"
echo "  Output:          ${RUN_DIR}"
echo "============================================================"

# ── Step 1: Calibrate — collect real-distribution topk histograms ──
# calibrate_topk.py runs the model end-to-end with histogram profiling
# enabled and writes per-segment raw_histograms.npy. The histograms are
# aggregated over every layer and every decode/prefill step so the
# autotune in Step 2 sees the true attention-score distribution.
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
    --output-dir "${CALIBRATION_DIR}" \
    2>&1 | tee "${RUN_DIR}/step1_calibrate.log"
  REAL_HIST_PATH="${CALIBRATION_DIR}/raw_histograms.npy"
  echo ">>> Step 1: Done. Calibration saved to ${CALIBRATION_DIR}"
fi

# Calibration may have produced lut.npy / quantiles.npy for modes 1 and 2.
CALIB_DIR="$(dirname "${REAL_HIST_PATH}")"
LUT_PATH=""
Q_PATH=""
[ -f "${CALIB_DIR}/lut.npy" ]       && LUT_PATH="${CALIB_DIR}/lut.npy"
[ -f "${CALIB_DIR}/quantiles.npy" ] && Q_PATH="${CALIB_DIR}/quantiles.npy"
[ -n "${LUT_PATH}" ] && echo "  Calibration LUT:      ${LUT_PATH}"
[ -n "${Q_PATH}" ]   && echo "  Calibration quantile: ${Q_PATH}"

# ── Step 2: Auto-tune hyperparameters by profiled fused-topk latency ──
# For every (mode, hparam) combo in the sweep grid, the autotune runs the
# fused remap+topk kernel on the real histogram and measures end-to-end
# kernel latency with CUDA events. The per-mode hparam with the lowest
# measured topk kernel latency wins.
AUTOTUNE_JSON="${RUN_DIR}/autotune_results.json"
if [ "${SKIP_AUTOTUNE}" -eq 1 ]; then
  echo ""
  echo ">>> Step 2: SKIPPED (using fallback --mapping-hparam ${MAPPING_HPARAM})"
  AUTOTUNE_ARGS=""
else
  echo ""
  echo ">>> Step 2: Auto-tuning hyperparameters by profiled topk kernel latency"
  AUTOTUNE_EXTRA=()
  [ -n "${LUT_PATH}" ] && AUTOTUNE_EXTRA+=(--lut-path "${LUT_PATH}")
  [ -n "${Q_PATH}" ]   && AUTOTUNE_EXTRA+=(--quantiles-path "${Q_PATH}")
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --batch-size "${BATCH_SIZE}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --seq-len "${SEQ_LEN}" \
    --topk-val "${TOPK_VAL}" \
    --page-size "${BLOCK_SIZE}" \
    --real-histograms "${REAL_HIST_PATH}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --collect-stats \
    "${AUTOTUNE_EXTRA[@]}" \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2_autotune.log"
  echo ">>> Step 2: Done. Autotune results saved to ${AUTOTUNE_JSON}"
  AUTOTUNE_ARGS="--autotune-json ${AUTOTUNE_JSON}"
fi

# ── Step 3: Remap benchmark (baseline / fused / remap / split) ──
echo ""
echo ">>> Step 3: Timing remap / topk / fused / baseline with autotuned hparams"
REMAP_JSON="${RUN_DIR}/remap_bench.json"
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
  --mapping-hparam "${MAPPING_HPARAM}" \
  ${AUTOTUNE_ARGS} \
  "${BENCH_EXTRA[@]}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --output-json "${REMAP_JSON}" \
  2>&1 | tee "${RUN_DIR}/step3_remap_bench.log"
echo ">>> Step 3: Done. Remap bench saved to ${REMAP_JSON}"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Remap Function Benchmark Complete"
echo "  Model:            ${MODEL_NAME}"
echo "  Block size:       ${BLOCK_SIZE}"
echo "  All outputs in:   ${RUN_DIR}/"
echo "    calibration/raw_histograms.npy  — real topk distribution (per layer)"
echo "    autotune_results.json           — latency-ranked mapping hparams"
echo "    remap_bench.json                — per-config remap/topk/fused/baseline latencies"
echo "    step{1,2,3}_*.log               — pipeline logs"
echo "============================================================"
