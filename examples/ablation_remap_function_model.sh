#!/usr/bin/env bash
# ============================================================
# Ablation: Remap function vs. model
#
# Sweeps MODEL_NAME across the Qwen3 family. For every model:
#   1. Calibrate (or reuse cached raw_histograms_<model>.npy)
#   2. Autotune the per-mode hparam on that model's histogram
#      (NOT hardcoded; freshly tuned per model)
#   3. Remap-bench across the autotuned hparams
#
# Mapping modes under test (matches the screenshot):
#   0 none, 3 power, 6 asinh, 7 log1p, 9 erf, 10 tanh,
#   11 subtract, 13 exp_stretch, 15 shift_pow2, 16 shift_pow3,
#   17 linear_steep
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=4
MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B"
TOPK_VAL=2048
BLOCK_SIZE=1
MEM=0.7
MIN_FREE_DISK_GB=22
ALGO="block_sparse_attention"
BATCH_SIZE=4
NUM_KV_HEADS=8
DISTRIBUTIONS="normal bucket_uniform"
MAPPING_MODES="0 3 6 7 9 10 11 13 15 16 17"
MAPPING_HPARAM=0.5
REPEAT=100
WARMUP=20

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)              GPU_ID="$2"; shift 2 ;;
    --models)           MODELS="$2"; shift 2 ;;
    --topk-val)         TOPK_VAL="$2"; shift 2 ;;
    --block-size|--page-size) BLOCK_SIZE="$2"; shift 2 ;;
    --mem)              MEM="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    --algo)             ALGO="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    --num-kv-heads)     NUM_KV_HEADS="$2"; shift 2 ;;
    --distributions)    DISTRIBUTIONS="$2"; shift 2 ;;
    --modes)            MAPPING_MODES="$2"; shift 2 ;;
    --mapping-hparam)   MAPPING_HPARAM="$2"; shift 2 ;;
    --repeat)           REPEAT="$2"; shift 2 ;;
    --warmup)           WARMUP="$2"; shift 2 ;;
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

# Per-model max-total-tokens (KV pool cap for calibration). Larger models
# need a smaller cap so they fit at MEM=0.7. Override by passing the env
# var MAX_TOTAL_TOKENS_<TAG>=N before invocation.
declare -A MAX_TOTAL_TOKENS_LUT
MAX_TOTAL_TOKENS_LUT["qwen3-0.6B"]=131072
MAX_TOTAL_TOKENS_LUT["qwen3-1.7B"]=64768
MAX_TOTAL_TOKENS_LUT["qwen3-4B"]=32768
MAX_TOTAL_TOKENS_LUT["qwen3-8B"]=16384

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${RESULTS_DIR}/ablation_remap_model_${TIMESTAMP}"
mkdir -p "${SWEEP_DIR}"

CALIBRATION_BASE="/var/tmp/zhuominc/vortex_torch/calibration"
mkdir -p "${CALIBRATION_BASE}"

echo "============================================================"
echo "Ablation: remap function vs model"
echo "  Models:          ${MODELS}"
echo "  TopK:            ${TOPK_VAL}"
echo "  Block size:      ${BLOCK_SIZE}"
echo "  Mapping modes:   ${MAPPING_MODES}"
echo "  GPU:             ${GPU_ID}"
echo "  Sweep dir:       ${SWEEP_DIR}"
echo "============================================================"

# ── Sweep ──────────────────────────────────────────────────────
SWEEP_INDEX="${SWEEP_DIR}/sweep_index.json"
{
  echo "{"
  echo "  \"axis_name\": \"model\","
  echo "  \"axis_type\": \"kernel\","
  echo "  \"topk_val\": ${TOPK_VAL},"
  echo "  \"block_size\": ${BLOCK_SIZE},"
  echo "  \"mapping_modes\": [${MAPPING_MODES// /, }],"
  echo "  \"cells\": ["
} > "${SWEEP_INDEX}"

# Pick a single seq_len that satisfies pages/seg > topk_val for all models.
MIN_SEQ_LEN=$(( (TOPK_VAL + 4) * BLOCK_SIZE ))
SEQ_LEN=${MIN_SEQ_LEN}
if [ "${SEQ_LEN}" -lt 8192 ]; then SEQ_LEN=8192; fi

FIRST_CELL=1
for MODEL_NAME in ${MODELS}; do
  MODEL_TAG="$(echo "${MODEL_NAME##*/}" | sed 's/^Q/q/')"
  MODEL_SLUG="$(echo "${MODEL_NAME}" | tr '/' '_')"
  DEFAULT_REAL_HIST="${CALIBRATION_BASE}/raw_histograms_${MODEL_TAG}.npy"

  # Per-model max-total-tokens (override-able via env).
  MTT_DEFAULT="${MAX_TOTAL_TOKENS_LUT[${MODEL_TAG}]:-32768}"
  ENV_KEY="MAX_TOTAL_TOKENS_$(echo "${MODEL_TAG}" | tr '.-' '__')"
  MAX_TOTAL_TOKENS="${!ENV_KEY:-${MTT_DEFAULT}}"

  CELL_DIR="${SWEEP_DIR}/${MODEL_SLUG}"
  mkdir -p "${CELL_DIR}"
  AUTOTUNE_JSON="${CELL_DIR}/autotune_results.json"
  REMAP_JSON="${CELL_DIR}/remap_bench.json"

  echo ""
  echo "============================================================"
  echo ">>> Cell: model=${MODEL_NAME} (max_total_tokens=${MAX_TOTAL_TOKENS})"
  echo "============================================================"

  # Step 1: calibrate (cached per-model)
  if [ -f "${DEFAULT_REAL_HIST}" ]; then
    echo ">>> Calibration cache hit: ${DEFAULT_REAL_HIST}"
    REAL_HIST_PATH="${DEFAULT_REAL_HIST}"
  else
    echo ">>> Calibrating ${MODEL_NAME}"
    STAGING_DIR="${CALIBRATION_BASE}/staging_${MODEL_TAG}_${TIMESTAMP}"
    mkdir -p "${STAGING_DIR}"
    python "${BENCH_DIR}/calibrate_topk.py" \
      --model-name "${MODEL_NAME}" \
      --topk-val "${TOPK_VAL}" \
      --page-size "${BLOCK_SIZE}" \
      --mem "${MEM}" \
      --max-total-tokens "${MAX_TOTAL_TOKENS}" \
      --min-free-disk-gb "${MIN_FREE_DISK_GB}" \
      --vortex-module-name "${ALGO}" \
      --output-dir "${STAGING_DIR}" \
      2>&1 | tee "${CELL_DIR}/step1_calibrate.log"
    mv -f "${STAGING_DIR}/raw_histograms.npy" "${DEFAULT_REAL_HIST}"
    REAL_HIST_PATH="${DEFAULT_REAL_HIST}"
  fi

  # Step 2: autotune
  echo ">>> Autotuning hparams for ${MODEL_NAME}"
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
    --output-json "${AUTOTUNE_JSON}" \
    2>&1 | tee "${CELL_DIR}/step2_autotune.log"

  # Step 3: remap bench
  echo ">>> Remap bench for ${MODEL_NAME}"
  PYTHONPATH="${SCRIPT_DIR}/.." python "${BENCH_DIR}/bench_topk.py" \
    --remap-bench \
    --per-head-bench \
    --batch-sizes "${BATCH_SIZE}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --seq-lens "${SEQ_LEN}" \
    --topk-vals "${TOPK_VAL}" \
    --page-size "${BLOCK_SIZE}" \
    --distributions ${DISTRIBUTIONS} \
    --mapping-modes ${MAPPING_MODES} \
    --mapping-hparam "${MAPPING_HPARAM}" \
    --autotune-json "${AUTOTUNE_JSON}" \
    --real-histograms "${REAL_HIST_PATH}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --output-json "${REMAP_JSON}" \
    2>&1 | tee "${CELL_DIR}/step3_remap_bench.log"

  if [ "${FIRST_CELL}" -eq 1 ]; then
    FIRST_CELL=0
  else
    echo "    ," >> "${SWEEP_INDEX}"
  fi
  cat >> "${SWEEP_INDEX}" <<EOF
    {
      "axis_value": "${MODEL_NAME}",
      "axis_label": "${MODEL_TAG}",
      "model_tag": "${MODEL_TAG}",
      "max_total_tokens": ${MAX_TOTAL_TOKENS},
      "cell_dir": "${CELL_DIR}",
      "autotune_json": "${AUTOTUNE_JSON}",
      "remap_bench_json": "${REMAP_JSON}"
    }
EOF
done

echo "  ]" >> "${SWEEP_INDEX}"
echo "}" >> "${SWEEP_INDEX}"

# ── Per-cell screenshot-style hparam summary ──────────────────
SELECTED_TXT="${SWEEP_DIR}/selected_hparams.txt"
PYTHONPATH="${SCRIPT_DIR}/.." python3 - "${SWEEP_INDEX}" "${SELECTED_TXT}" "model" <<'PY'
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
try:
    from autotune_topk_mapping import PARAM_NAME
except Exception:
    PARAM_NAME = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha",
                  11: "pivot", 13: "alpha", 15: "pivot", 16: "pivot", 17: "k"}
DISPLAY = {3: "Power", 6: "Asinh", 7: "Log1p", 9: "Erf", 10: "Tanh",
           11: "Subtract", 13: "ExpStretch", 15: "ShiftPow2",
           16: "ShiftPow3", 17: "LinearSteep"}

idx_path, out_path, axis_name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(idx_path) as f:
    idx = json.load(f)

lines = [f"== Selected mapping functions (autotuned, {axis_name} sweep) =="]
for cell in idx["cells"]:
    with open(cell["autotune_json"]) as f:
        results = json.load(f)
    best = {}
    for r in results:
        m = r["mode"]
        if m not in best or r["latency_ms"] < best[m]["latency_ms"]:
            best[m] = r
    parts = []
    for m in sorted(DISPLAY):
        if m in best:
            parts.append(f"{DISPLAY[m]}({PARAM_NAME.get(m,'p')}={best[m].get('param',0.0)})")
    lines.append(f"[{axis_name}={cell['axis_value']}] " + "  ".join(parts))

txt = "\n".join(lines) + "\n"
print(txt)
with open(out_path, "w") as f:
    f.write(txt)
PY

echo ""
echo "============================================================"
echo "Model ablation complete."
echo "  Sweep dir:        ${SWEEP_DIR}"
echo "  Sweep index:      ${SWEEP_INDEX}"
echo "  Selected hparams: ${SELECTED_TXT}"
echo "Run analyze with:"
echo "  python examples/analyze_ablation_remap.py --sweep-dir ${SWEEP_DIR}"
echo "============================================================"
