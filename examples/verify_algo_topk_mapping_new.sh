#!/usr/bin/env bash
# ============================================================
# E2E accuracy sweep over the surviving parametric mapping modes.
# Each mode runs verify_algo.py with the per-mode hyperparameter
# that autotune_topk_mapping.py picked as having the lowest
# measured fused-topk-kernel latency.
#
# Mapping modes (after the lean refactor):
#   0: None           — unmapped baseline (no remap)
#   3: Power          — y = sign(x) * |x|^p
#   4: Log            — y = sign(x) * log(|x| + 1)    [no knob]
#   6: Asinh          — y = asinh(beta * x)
#   7: Log1p          — y = sign(x) * log1p(alpha * |x|)
#   9: Erf            — y = erf(alpha * x)
#  10: Tanh           — y = tanh(alpha * x)
#  13: ExpStretch     — y = exp(alpha * x)
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=5
TOPK_VAL=30
BENCHMARKS="amc23"
MODEL_NAME="Qwen/Qwen3-1.7B"
BLOCK_SIZE=16
BATCH_SIZE=4
NUM_KV_HEADS=2
SEQ_LEN=32768
MAX_TOTAL_TOKENS=1048576
REAL_HISTOGRAMS=""
SKIP_AUTOTUNE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topk-val)        TOPK_VAL="$2"; shift 2 ;;
    --gpu)             GPU_ID="$2"; shift 2 ;;
    --benchmark)       BENCHMARKS="$2"; shift 2 ;;
    --model-name)      MODEL_NAME="$2"; shift 2 ;;
    --block-size|--page-size) BLOCK_SIZE="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-kv-heads)    NUM_KV_HEADS="$2"; shift 2 ;;
    --seq-len)         SEQ_LEN="$2"; shift 2 ;;
    --real-histograms) REAL_HISTOGRAMS="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --skip-autotune)   SKIP_AUTOTUNE=1; shift 1 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

sparse_algos=( "block_sparse_attention" )

BENCH_LABEL=$(echo "${BENCHMARKS}" | tr ' ' '_')
MODEL_SLUG="$(echo "${MODEL_NAME}" | tr '/' '_')"
RESULTS_DIR="results/topk_mapping_${MODEL_SLUG}_topk${TOPK_VAL}_${BENCH_LABEL}"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Step 0: Calibrate (optional) — real-distribution histograms
# ============================================================
if [ -z "${REAL_HISTOGRAMS}" ]; then
  echo "============================================================"
  echo "Step 0: Calibrating ${MODEL_NAME} for real-distribution histograms"
  echo "  Max total tokens (KV / VTX cap): ${MAX_TOTAL_TOKENS}"
  echo "============================================================"
  CAL_DIR="${RESULTS_DIR}/calibration_${TIMESTAMP}"
  mkdir -p "${CAL_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --mem 0.7 \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --vortex-module-name "${sparse_algos[0]}" \
    --page-size "${BLOCK_SIZE}" \
    --output-dir "${CAL_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/calibrate_${TIMESTAMP}.log"
  REAL_HISTOGRAMS="${CAL_DIR}/raw_histograms.npy"
fi

# Pick up lut.npy / quantiles.npy if calibration produced them.
CALIB_DIR="$(dirname "${REAL_HISTOGRAMS}")"
LUT_PATH=""
Q_PATH=""
[ -f "${CALIB_DIR}/lut.npy" ]       && LUT_PATH="${CALIB_DIR}/lut.npy"
[ -f "${CALIB_DIR}/quantiles.npy" ] && Q_PATH="${CALIB_DIR}/quantiles.npy"

# ============================================================
# Step 1: Auto-tune — rank by profiled fused-topk kernel latency
# ============================================================
AUTOTUNE_JSON="${RESULTS_DIR}/autotune_${TIMESTAMP}.json"
if [ "${SKIP_AUTOTUNE}" -eq 0 ]; then
  echo "============================================================"
  echo "Step 1: Auto-tuning hyperparameters by fused-topk kernel latency"
  echo "============================================================"
  AUTOTUNE_EXTRA=()
  [ -n "${LUT_PATH}" ] && AUTOTUNE_EXTRA+=(--lut-path "${LUT_PATH}")
  [ -n "${Q_PATH}" ]   && AUTOTUNE_EXTRA+=(--quantiles-path "${Q_PATH}")
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/autotune_topk_mapping.py" \
    --topk-val ${TOPK_VAL} \
    --batch-size ${BATCH_SIZE} \
    --seq-len ${SEQ_LEN} \
    --num-kv-heads ${NUM_KV_HEADS} \
    --page-size ${BLOCK_SIZE} \
    --real-histograms "${REAL_HISTOGRAMS}" \
    "${AUTOTUNE_EXTRA[@]}" \
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${RESULTS_DIR}/autotune_${TIMESTAMP}.log"
  echo ">>> Auto-tune results saved to ${AUTOTUNE_JSON}"
fi

# Extract best per-mode hparam (ranked by measured kernel latency, lowest wins)
eval "$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
best = {}
for r in data:
    m = r.get('mode')
    lat = r.get('latency_ms')
    if m is None or lat is None: continue
    if m not in best or lat < best[m]['latency_ms']:
        best[m] = r
for m in (3, 6, 7, 9, 10, 11, 13):
    v = best.get(m, {}).get('param', 0.5)
    print(f'BEST_HPARAM_{m}={v}')
" "${AUTOTUNE_JSON}")"
echo ">>> Autotuned hparams (lowest topk kernel latency):"
echo "    mode3=${BEST_HPARAM_3} mode6=${BEST_HPARAM_6} mode7=${BEST_HPARAM_7}"
echo "    mode9=${BEST_HPARAM_9} mode10=${BEST_HPARAM_10} mode11=${BEST_HPARAM_11} mode13=${BEST_HPARAM_13}"
echo ""

run_verify() {
  # $1=mode $2=hparam $3=label
  local mode="$1"; local hp="$2"; local label="$3"
  for algo in "${sparse_algos[@]}"; do
    local out="${RESULTS_DIR}/topk_mapping_${algo}_${label}_${TIMESTAMP}.log"
    echo ">>> ${label} algo=${algo}"
    local extra_args=()
    if [ "${mode}" -eq 0 ]; then
      extra_args+=(--topk-type sglang)
    else
      extra_args+=(--topk-type sglang_fused --topk-mapping-mode "${mode}" --topk-mapping-hparam "${hp}")
    fi
    { time python verify_algo.py \
      --trials 8 \
      --topk-val "${TOPK_VAL}" \
      --vortex-module-name "${algo}" \
      --model-name "${MODEL_NAME}" \
      --benchmark ${BENCHMARKS} \
      --mem 0.7 \
      "${extra_args[@]}" ; } \
      2>&1 | tee "${out}"
  done
}

echo "============================================================"
echo "Baseline: sglang (no remap)"
echo "============================================================"
run_verify 0 0.5 "sglang_m0"

echo "============================================================"
echo "Mode 3 (power) — p=${BEST_HPARAM_3} (autotuned)"
echo "============================================================"
run_verify 3 "${BEST_HPARAM_3}" "sglang_m3_p${BEST_HPARAM_3}"

echo "============================================================"
echo "Mode 4 (log)"
echo "============================================================"
run_verify 4 0.5 "sglang_m4"

echo "============================================================"
echo "Mode 6 (asinh) — beta=${BEST_HPARAM_6} (autotuned)"
echo "============================================================"
run_verify 6 "${BEST_HPARAM_6}" "sglang_m6_beta${BEST_HPARAM_6}"

echo "============================================================"
echo "Mode 7 (log1p) — alpha=${BEST_HPARAM_7} (autotuned)"
echo "============================================================"
run_verify 7 "${BEST_HPARAM_7}" "sglang_m7_alpha${BEST_HPARAM_7}"

echo "============================================================"
echo "Mode 9 (erf) — alpha=${BEST_HPARAM_9} (autotuned)"
echo "============================================================"
run_verify 9 "${BEST_HPARAM_9}" "sglang_m9_alpha${BEST_HPARAM_9}"

echo "============================================================"
echo "Mode 10 (tanh) — alpha=${BEST_HPARAM_10} (autotuned)"
echo "============================================================"
run_verify 10 "${BEST_HPARAM_10}" "sglang_m10_alpha${BEST_HPARAM_10}"

echo "============================================================"
echo "Mode 8 (trunc8)"
echo "============================================================"
run_verify 8 0.5 "sglang_m8"

echo "============================================================"
echo "Mode 11 (subtract) — pivot=${BEST_HPARAM_11} (autotuned)"
echo "============================================================"
run_verify 11 "${BEST_HPARAM_11}" "sglang_m11_pivot${BEST_HPARAM_11}"

echo "============================================================"
echo "Mode 13 (exp_stretch) — alpha=${BEST_HPARAM_13} (autotuned)"
echo "============================================================"
run_verify 13 "${BEST_HPARAM_13}" "sglang_m13_alpha${BEST_HPARAM_13}"

echo ""
echo "============================================================"
echo "All runs complete. Results in ${RESULTS_DIR}/"
echo "  Model:      ${MODEL_NAME}"
echo "  Block size: ${BLOCK_SIZE}"
echo "  Auto-tune:  ${AUTOTUNE_JSON}"
echo "  Mode 3 (power):       p     = ${BEST_HPARAM_3} (autotuned)"
echo "  Mode 6 (asinh):       beta  = ${BEST_HPARAM_6} (autotuned)"
echo "  Mode 7 (log1p):       alpha = ${BEST_HPARAM_7} (autotuned)"
echo "  Mode 9 (erf):         alpha = ${BEST_HPARAM_9} (autotuned)"
echo "  Mode 10 (tanh):       alpha = ${BEST_HPARAM_10} (autotuned)"
echo "  Mode 13 (exp_stretch):alpha = ${BEST_HPARAM_13} (autotuned)"
echo "============================================================"
