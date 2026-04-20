#!/usr/bin/env bash
# ============================================================
# Remap Function Benchmark — Parallel TopK variant.
#
# Wraps bench_topk.py --remap-bench with --bench-parallel so the
# output table includes a "par_ms" column comparing the split+merge
# kernel (topk_output_sglang_parallel) against the single-CTA
# fused kernel. Also sweeps batch size and num_splits so the
# occupancy-vs-merge-overhead curve is visible.
#
# Pipeline mirrors remap_function_bench_topk2028.sh:
#   Step 1 — calibrate (can be skipped with --real-histograms)
#   Step 2 — autotune per-mode hparams by fused-kernel latency
#   Step 3 — remap bench, looped over NUM_SPLITS_SWEEP values
#
# Usage:
#   bash remap_function_bench_topk_parallel.sh --gpu 4
#
#   # Explicit batch-size sweep:
#   bash remap_function_bench_topk_parallel.sh --gpu 4 \
#       --batch-sizes "1 2 4 8" --num-splits-sweep "auto 2 4 8"
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=7
MODEL_NAME="Qwen/Qwen3-1.7B"
TOPK_VAL=2048
MEM=0.7
MAX_TOTAL_TOKENS=64768
MIN_FREE_DISK_GB=22
ALGO="block_sparse_attention"
SAMPLE_STRIDE=1
SEQ_LEN=32768
BLOCK_SIZE=1
BATCH_SIZES="1 2 4 8 16"
NUM_KV_HEADS=8
DISTRIBUTIONS="normal bucket_uniform"
# Modes excluding 1 (LUT_CDF) and 2 (Quantile) which are discarded.
MAPPING_MODES="0 3 6 7 9 10 11 13 15 16 17 18 19"
MAPPING_HPARAM=0.5
REPEAT=100
WARMUP=20
# "auto" lets bench_topk.py pick via sqrt(pages/topk). Explicit ints
# pin a split count for A/B comparisons.
NUM_SPLITS_SWEEP="auto 2 4 8"
REAL_HISTOGRAMS="/var/tmp/zhuominc/vortex_torch/calibration/raw_histograms_qwen3-1.7B.npy"
SKIP_AUTOTUNE=0
PINNED_AUTOTUNE_JSON=""

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)       MODEL_NAME="$2"; shift 2 ;;
    --topk-val)         TOPK_VAL="$2"; shift 2 ;;
    --mem)              MEM="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    --gpu)              GPU_ID="$2"; shift 2 ;;
    --algo)             ALGO="$2"; shift 2 ;;
    --real-histograms)  REAL_HISTOGRAMS="$2"; shift 2 ;;
    --sample-stride)    SAMPLE_STRIDE="$2"; shift 2 ;;
    --seq-len)          SEQ_LEN="$2"; shift 2 ;;
    --block-size|--page-size) BLOCK_SIZE="$2"; shift 2 ;;
    --batch-sizes)      BATCH_SIZES="$2"; shift 2 ;;
    --num-kv-heads)     NUM_KV_HEADS="$2"; shift 2 ;;
    --distributions)    DISTRIBUTIONS="$2"; shift 2 ;;
    --modes)            MAPPING_MODES="$2"; shift 2 ;;
    --mapping-hparam)   MAPPING_HPARAM="$2"; shift 2 ;;
    --repeat)           REPEAT="$2"; shift 2 ;;
    --warmup)           WARMUP="$2"; shift 2 ;;
    --num-splits-sweep) NUM_SPLITS_SWEEP="$2"; shift 2 ;;
    --skip-autotune)    SKIP_AUTOTUNE=1; shift 1 ;;
    --pinned-autotune-json) PINNED_AUTOTUNE_JSON="$2"; SKIP_AUTOTUNE=1; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGL_ENABLE_JIT_DEEPGEMM="${SGL_ENABLE_JIT_DEEPGEMM:-true}"

if [ -z "${DG_JIT_NVCC_COMPILER:-}" ]; then
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export DG_JIT_NVCC_COMPILER="${CUDA_HOME}/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then
    export DG_JIT_NVCC_COMPILER="$(command -v nvcc)"
  fi
fi

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
RUN_DIR="${RESULTS_DIR}/parallel_bench_${MODEL_SLUG}_topk${TOPK_VAL}_bs${BLOCK_SIZE}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

CALIBRATION_BASE="/var/tmp/zhuominc/vortex_torch/calibration"
MODEL_TAG="$(echo "${MODEL_NAME##*/}" | sed 's/^Q/q/')"
DEFAULT_REAL_HIST="${CALIBRATION_BASE}/raw_histograms_${MODEL_TAG}.npy"
mkdir -p "${CALIBRATION_BASE}"

if [ -z "${REAL_HISTOGRAMS}" ] && [ -f "${DEFAULT_REAL_HIST}" ]; then
  REAL_HISTOGRAMS="${DEFAULT_REAL_HIST}"
fi

echo "============================================================"
echo "Remap Function Benchmark (Parallel TopK variant)"
echo "  Model:           ${MODEL_NAME}"
echo "  Algorithm:       ${ALGO}"
echo "  TopK:            ${TOPK_VAL}"
echo "  Block size:      ${BLOCK_SIZE}"
echo "  Seq len:         ${SEQ_LEN} ($(( SEQ_LEN / BLOCK_SIZE )) pages/seg)"
echo "  Batch sizes:     ${BATCH_SIZES}"
echo "  KV heads:        ${NUM_KV_HEADS}"
echo "  Distributions:   ${DISTRIBUTIONS}"
echo "  Mapping modes:   ${MAPPING_MODES}"
echo "  num_splits sweep:${NUM_SPLITS_SWEEP}"
echo "  GPU:             ${GPU_ID}"
echo "  Real histograms: ${REAL_HISTOGRAMS:-<will calibrate from ${MODEL_NAME}>}"
echo "  Output:          ${RUN_DIR}"
echo "============================================================"

# ── Step 1: Calibrate ────────────────────────────────────────
if [ -n "${REAL_HISTOGRAMS}" ]; then
  echo ""
  echo ">>> Step 1: SKIPPED (using provided --real-histograms ${REAL_HISTOGRAMS})"
  REAL_HIST_PATH="${REAL_HISTOGRAMS}"
else
  echo ""
  echo ">>> Step 1: Calibrating ${MODEL_NAME} — collecting real topk histograms"
  CALIBRATION_DIR="${CALIBRATION_BASE}/staging_${MODEL_TAG}_topk${TOPK_VAL}_bs${BLOCK_SIZE}_${TIMESTAMP}"
  mkdir -p "${CALIBRATION_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --page-size "${BLOCK_SIZE}" \
    --mem "${MEM}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --min-free-disk-gb "${MIN_FREE_DISK_GB}" \
    --vortex-module-name "${ALGO}" \
    --output-dir "${CALIBRATION_DIR}" \
    2>&1 | tee "${RUN_DIR}/step1_calibrate.log"
  mv -f "${CALIBRATION_DIR}/raw_histograms.npy" "${DEFAULT_REAL_HIST}"
  REAL_HIST_PATH="${DEFAULT_REAL_HIST}"
  echo ">>> Step 1: Done. raw_histograms -> ${REAL_HIST_PATH}"
fi

# ── Step 2: Autotune ─────────────────────────────────────────
AUTOTUNE_JSON="${RUN_DIR}/autotune_results.json"
if [ "${SKIP_AUTOTUNE}" -eq 1 ]; then
  echo ""
  if [ -n "${PINNED_AUTOTUNE_JSON}" ]; then
    echo ">>> Step 2: SKIPPED (pinned hparams from ${PINNED_AUTOTUNE_JSON})"
    AUTOTUNE_ARGS="--autotune-json ${PINNED_AUTOTUNE_JSON}"
  else
    echo ">>> Step 2: SKIPPED (using fallback --mapping-hparam ${MAPPING_HPARAM})"
    AUTOTUNE_ARGS=""
  fi
else
  echo ""
  echo ">>> Step 2: Auto-tuning hyperparameters by profiled topk kernel latency"
  # Autotune on the largest batch size so the picked hparam matches realistic
  # decode conditions; the hparam itself is largely batch-invariant.
  FIRST_BS="$(echo ${BATCH_SIZES} | awk '{print $NF}')"
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --batch-size "${FIRST_BS}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --seq-len "${SEQ_LEN}" \
    --topk-val "${TOPK_VAL}" \
    --page-size "${BLOCK_SIZE}" \
    --real-histograms "${REAL_HIST_PATH}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --collect-stats \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2_autotune.log"
  echo ">>> Step 2: Done. Autotune results saved to ${AUTOTUNE_JSON}"
  AUTOTUNE_ARGS="--autotune-json ${AUTOTUNE_JSON}"
fi

# ── Step 3: Remap + Parallel bench, sweeping num_splits ──────
echo ""
echo ">>> Step 3: Timing baseline / fused / parallel with num_splits sweep"

for NS in ${NUM_SPLITS_SWEEP}; do
  if [ "${NS}" = "auto" ]; then
    NS_ARG="--num-splits -1"
    NS_TAG="auto"
  else
    NS_ARG="--num-splits ${NS}"
    NS_TAG="ns${NS}"
  fi
  REMAP_JSON="${RUN_DIR}/remap_bench_${NS_TAG}.json"
  LOG="${RUN_DIR}/step3_remap_bench_${NS_TAG}.log"
  BENCH_EXTRA=()
  [ -n "${REAL_HIST_PATH}" ] && BENCH_EXTRA+=(--real-histograms "${REAL_HIST_PATH}")
  echo ""
  echo "--- num_splits=${NS_TAG} ---"
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
    --remap-bench \
    --bench-parallel \
    ${NS_ARG} \
    --batch-sizes ${BATCH_SIZES} \
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
    2>&1 | tee "${LOG}"
  echo ">>> num_splits=${NS_TAG}: JSON -> ${REMAP_JSON}"
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Parallel TopK Benchmark Complete"
echo "  Model:            ${MODEL_NAME}"
echo "  Block size:       ${BLOCK_SIZE}"
echo "  Batch sizes:      ${BATCH_SIZES}"
echo "  num_splits sweep: ${NUM_SPLITS_SWEEP}"
echo "  All outputs in:   ${RUN_DIR}/"
echo "    autotune_results.json                  — latency-ranked mapping hparams"
echo "    remap_bench_<splits>.json              — per-config latencies including par_ms"
echo "    step{1,2,3}_*.log                      — pipeline logs"
echo "============================================================"
