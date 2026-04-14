#!/usr/bin/env bash
# ============================================================
# Unified TopK Benchmark
#
# Three-step pipeline on a single configurable model:
#   Step 1: Calibrate                — run the model to collect
#                                      real-distribution histograms
#                                      (raw_histograms.npy, lut.npy,
#                                      quantiles.npy).
#   Step 2: Latency autotune + bench — rank per-mode hparams by
#                                      measured fused-topk kernel
#                                      latency, then run the
#                                      remap / topk / fused / baseline
#                                      comparison.
#   Step 3: E2E accuracy             — verify_algo.py on the same
#                                      model for the unmapped baseline
#                                      plus each mapping mode, with
#                                      autotuned hparams.
#
# Usage:
#   bash run_topk_benchmark.sh --gpu 0
#   bash run_topk_benchmark.sh --gpu 0 --model-name Qwen/Qwen3-8B \
#        --block-size 32 --topk-val 512
#   bash run_topk_benchmark.sh --gpu 0 --max-total-tokens 1048576
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=4
MODEL_NAME="Qwen/Qwen3-1.7B"
TOPK_VAL=30
TRIALS=8
MEM=0.7
MAX_TOTAL_TOKENS=1048576
ALGO="block_sparse_attention"
BLOCK_SIZE=16
BATCH_SIZE=4
NUM_KV_HEADS=8
SEQ_LEN=32768
BENCHMARKS="amc23"
SKIP_CALIBRATE=false
SKIP_KERNEL=false
SKIP_E2E=true

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)      MODEL_NAME="$2"; shift 2 ;;
    --topk-val)        TOPK_VAL="$2"; shift 2 ;;
    --trials)          TRIALS="$2"; shift 2 ;;
    --mem)             MEM="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --gpu)             GPU_ID="$2"; shift 2 ;;
    --algo)            ALGO="$2"; shift 2 ;;
    --benchmark)       BENCHMARKS="$2"; shift 2 ;;
    --block-size|--page-size) BLOCK_SIZE="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-kv-heads)    NUM_KV_HEADS="$2"; shift 2 ;;
    --seq-len)         SEQ_LEN="$2"; shift 2 ;;
    --skip-calibrate)  SKIP_CALIBRATE=true; shift ;;
    --skip-kernel)     SKIP_KERNEL=true; shift ;;
    --skip-e2e)        SKIP_E2E=false; shift ;;  # --skip-e2e actually toggles it OFF (enables)
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCH_LABEL=$(echo "${BENCHMARKS}" | tr ' ' '_')
MODEL_SLUG="$(echo "${MODEL_NAME}" | tr '/' '_')"
RUN_DIR="${RESULTS_DIR}/topk_benchmark_${MODEL_SLUG}_${BENCH_LABEL}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "Unified TopK Benchmark"
echo "  Model:      ${MODEL_NAME}"
echo "  Algorithm:  ${ALGO}"
echo "  TopK:       ${TOPK_VAL}"
echo "  Block size: ${BLOCK_SIZE}"
echo "  Seq len:    ${SEQ_LEN}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  KV heads:   ${NUM_KV_HEADS}"
echo "  Trials:     ${TRIALS}"
echo "  Max total tokens: ${MAX_TOTAL_TOKENS}  (calibration KV / VTX buffer cap)"
echo "  GPU:        ${GPU_ID}"
echo "  Output:     ${RUN_DIR}"
echo "============================================================"

# ── Step 1: Calibrate ────────────────────────────────────────
CALIBRATION_DIR="${RUN_DIR}/calibration"
if [ "${SKIP_CALIBRATE}" = true ] && [ -d "${CALIBRATION_DIR}" ]; then
  echo ""
  echo ">>> Step 1: SKIPPED (--skip-calibrate)"
else
  echo ""
  echo ">>> Step 1: Calibrating ${MODEL_NAME} — real topk histograms + LUT/quantiles"
  mkdir -p "${CALIBRATION_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --mem "${MEM}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --vortex-module-name "${ALGO}" \
    --page-size "${BLOCK_SIZE}" \
    --output-dir "${CALIBRATION_DIR}" \
    2>&1 | tee "${RUN_DIR}/step1_calibrate.log"
  echo ">>> Step 1: Done."
fi

REAL_HIST_PATH="${CALIBRATION_DIR}/raw_histograms.npy"
LUT_PATH=""
Q_PATH=""
[ -f "${CALIBRATION_DIR}/lut.npy" ]       && LUT_PATH="${CALIBRATION_DIR}/lut.npy"
[ -f "${CALIBRATION_DIR}/quantiles.npy" ] && Q_PATH="${CALIBRATION_DIR}/quantiles.npy"
[ -n "${LUT_PATH}" ] && echo "  Calibration LUT:      ${LUT_PATH}"
[ -n "${Q_PATH}" ]   && echo "  Calibration quantile: ${Q_PATH}"

# ── Step 2: Latency autotune + remap bench ───────────────────
AUTOTUNE_JSON="${RUN_DIR}/autotune_results.json"
if [ "${SKIP_KERNEL}" = true ]; then
  echo ""
  echo ">>> Step 2: SKIPPED (--skip-kernel)"
else
  echo ""
  echo ">>> Step 2a: Auto-tuning per-mode hparams by fused-topk kernel latency"
  AUTOTUNE_EXTRA=()
  [ -f "${REAL_HIST_PATH}" ] && AUTOTUNE_EXTRA+=(--real-histograms "${REAL_HIST_PATH}")
  [ -n "${LUT_PATH}" ] && AUTOTUNE_EXTRA+=(--lut-path "${LUT_PATH}")
  [ -n "${Q_PATH}" ]   && AUTOTUNE_EXTRA+=(--quantiles-path "${Q_PATH}")
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --topk-val "${TOPK_VAL}" \
    --batch-size "${BATCH_SIZE}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --seq-len "${SEQ_LEN}" \
    --page-size "${BLOCK_SIZE}" \
    --warmup 20 --repeat 100 \
    --collect-stats \
    "${AUTOTUNE_EXTRA[@]}" \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2a_autotune.log"
  echo ">>> Step 2a: Done. Autotune saved to ${AUTOTUNE_JSON}"

  echo ""
  echo ">>> Step 2b: Remap benchmark (baseline / fused / remap / split) with autotuned hparams"
  BENCH_JSON="${RUN_DIR}/kernel_latency.json"
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
    --distributions normal bucket_uniform \
    --mapping-modes 0 1 2 3 6 7 8 9 10 11 13 \
    --autotune-json "${AUTOTUNE_JSON}" \
    "${BENCH_EXTRA[@]}" \
    --warmup 20 --repeat 100 \
    --output-json "${BENCH_JSON}" \
    2>&1 | tee "${RUN_DIR}/step2b_kernel_bench.log"
  echo ">>> Step 2b: Done. Results saved to ${BENCH_JSON}"
fi

# ── Step 3: E2E accuracy ─────────────────────────────────────
if [ "${SKIP_E2E}" = true ]; then
  echo ""
  echo ">>> Step 3: SKIPPED (default). Pass --skip-e2e to toggle it ON."
else
  echo ""
  echo ">>> Step 3: E2E accuracy comparison"

  # Extract autotuned hparams per mode.
  eval "$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
best = {}
for r in data:
    m = r.get('mode'); lat = r.get('latency_ms')
    if m is None or lat is None: continue
    if m not in best or lat < best[m]['latency_ms']:
        best[m] = r
for m in (3, 6, 7, 9, 10, 11, 13):
    print(f'BEST_HPARAM_{m}={best.get(m, {}).get(\"param\", 0.5)}')
" "${AUTOTUNE_JSON}")"

  E2E_DIR="${RUN_DIR}/e2e"
  mkdir -p "${E2E_DIR}"

  run_e2e() {
    # $1=label, remaining args passed to verify_algo.py
    local label="$1"; shift
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

  run_mapped() {
    # $1=mode $2=hparam $3=label
    local mode="$1"; local hp="$2"; local label="$3"
    local extra=(--vortex-module-name "${ALGO}")
    if [ "${mode}" -eq 0 ]; then
      extra+=(--topk-type sglang)
    else
      extra+=(--topk-type sglang_fused --topk-mapping-mode "${mode}" --topk-mapping-hparam "${hp}")
    fi
    run_e2e "${label}" "${extra[@]}"
  }

  run_e2e "full_attention_baseline" --full-attention
  run_e2e "naive_topk"               --vortex-module-name "${ALGO}" --topk-type naive
  run_mapped 0  0.5                "sglang_m0_none"
  run_mapped 3  "${BEST_HPARAM_3}"  "sglang_m3_power_p${BEST_HPARAM_3}"
  run_mapped 4  0.5                "sglang_m4_log"
  run_mapped 6  "${BEST_HPARAM_6}"  "sglang_m6_asinh_beta${BEST_HPARAM_6}"
  run_mapped 7  "${BEST_HPARAM_7}"  "sglang_m7_log1p_alpha${BEST_HPARAM_7}"
  run_mapped 8  0.5                "sglang_m8_trunc8"
  run_mapped 9  "${BEST_HPARAM_9}"  "sglang_m9_erf_alpha${BEST_HPARAM_9}"
  run_mapped 10 "${BEST_HPARAM_10}" "sglang_m10_tanh_alpha${BEST_HPARAM_10}"
  run_mapped 11 "${BEST_HPARAM_11}" "sglang_m11_subtract_pivot${BEST_HPARAM_11}"
  run_mapped 13 "${BEST_HPARAM_13}" "sglang_m13_expstretch_alpha${BEST_HPARAM_13}"

  echo ""
  echo ">>> Step 3: Done. E2E logs saved to ${E2E_DIR}/"
fi

# ── Final Summary ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "TopK Benchmark Complete"
echo "  Model:       ${MODEL_NAME}"
echo "  Block size:  ${BLOCK_SIZE}"
echo "  All results: ${RUN_DIR}"
echo "  Calibration: ${CALIBRATION_DIR}"
[ "${SKIP_KERNEL}" != true ] && echo "  Autotune:    ${AUTOTUNE_JSON}"
[ "${SKIP_KERNEL}" != true ] && echo "  Kernel JSON: ${RUN_DIR}/kernel_latency.json"
[ "${SKIP_E2E}"    != true ] && echo "  E2E logs:    ${RUN_DIR}/e2e/"
echo "============================================================"
