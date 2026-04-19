#!/usr/bin/env bash
# ============================================================
# Ablation: Remap function vs. block (page) size
#
# Sweeps BLOCK_SIZE and, for every cell, runs the full
#   calibrate -> autotune -> remap-bench
# pipeline so the per-mode hyperparameter is freshly chosen by
# autotune for that block size (NOT hardcoded).
#
# Mapping modes under test (matches the screenshot):
#   0  none          — unmapped baseline
#   3  power         — p
#   6  asinh         — beta
#   7  log1p         — alpha
#   9  erf           — alpha
#  10  tanh          — alpha
#  11  subtract      — pivot
#  13  exp_stretch   — alpha
#  15  shift_pow2    — pivot
#  16  shift_pow3    — pivot
#  17  linear_steep  — k
#
# Output:
#   results/ablation_remap_block_size_<TS>/
#     bs<N>/{autotune_results.json, remap_bench.json, step{1,2,3}_*.log}
#     sweep_index.json
#     selected_hparams.txt   — per-cell screenshot-style summary
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/../benchmarks"

# ── Defaults ──────────────────────────────────────────────────
GPU_ID=4
MODEL_NAME="Qwen/Qwen3-1.7B"
TOPK_VAL=2048
MEM=0.7
MAX_TOTAL_TOKENS=64768
MIN_FREE_DISK_GB=22
ALGO="block_sparse_attention"
BATCH_SIZE=4
NUM_KV_HEADS=8
DISTRIBUTIONS="normal bucket_uniform"
MAPPING_MODES="0 3 6 7 9 10 11 13 15 16 17"
MAPPING_HPARAM=0.5
REPEAT=100
WARMUP=20
BLOCK_SIZES="1 2 4 8 16 32 64"
REAL_HISTOGRAMS=""

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)              GPU_ID="$2"; shift 2 ;;
    --model-name)       MODEL_NAME="$2"; shift 2 ;;
    --topk-val)         TOPK_VAL="$2"; shift 2 ;;
    --mem)              MEM="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    --algo)             ALGO="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    --num-kv-heads)     NUM_KV_HEADS="$2"; shift 2 ;;
    --distributions)    DISTRIBUTIONS="$2"; shift 2 ;;
    --modes)            MAPPING_MODES="$2"; shift 2 ;;
    --mapping-hparam)   MAPPING_HPARAM="$2"; shift 2 ;;
    --repeat)           REPEAT="$2"; shift 2 ;;
    --warmup)           WARMUP="$2"; shift 2 ;;
    --block-sizes)      BLOCK_SIZES="$2"; shift 2 ;;
    --real-histograms)  REAL_HISTOGRAMS="$2"; shift 2 ;;
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

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="$(echo "${MODEL_NAME}" | tr '/' '_')"
SWEEP_DIR="${RESULTS_DIR}/ablation_remap_block_size_${MODEL_SLUG}_${TIMESTAMP}"
mkdir -p "${SWEEP_DIR}"

# Per-model calibration cache (reused across block_size cells: page size
# does not change the per-segment score distribution).
CALIBRATION_BASE="/var/tmp/zhuominc/vortex_torch/calibration"
MODEL_TAG="$(echo "${MODEL_NAME##*/}" | sed 's/^Q/q/')"
DEFAULT_REAL_HIST="${CALIBRATION_BASE}/raw_histograms_${MODEL_TAG}.npy"
mkdir -p "${CALIBRATION_BASE}"

if [ -z "${REAL_HISTOGRAMS}" ] && [ -f "${DEFAULT_REAL_HIST}" ]; then
  REAL_HISTOGRAMS="${DEFAULT_REAL_HIST}"
fi

echo "============================================================"
echo "Ablation: remap function vs block_size"
echo "  Model:           ${MODEL_NAME}"
echo "  Algorithm:       ${ALGO}"
echo "  TopK:            ${TOPK_VAL}"
echo "  Block sizes:     ${BLOCK_SIZES}"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  KV heads:        ${NUM_KV_HEADS}"
echo "  Distributions:   ${DISTRIBUTIONS}"
echo "  Mapping modes:   ${MAPPING_MODES}"
echo "  GPU:             ${GPU_ID}"
echo "  Sweep dir:       ${SWEEP_DIR}"
echo "============================================================"

# ── Step 0: Calibrate once for this model ──────────────────────
if [ -n "${REAL_HISTOGRAMS}" ]; then
  echo ">>> Step 0: SKIPPED calibration (using ${REAL_HISTOGRAMS})"
  REAL_HIST_PATH="${REAL_HISTOGRAMS}"
else
  echo ">>> Step 0: Calibrating ${MODEL_NAME} for raw_histograms.npy"
  STAGING_DIR="${CALIBRATION_BASE}/staging_${MODEL_TAG}_${TIMESTAMP}"
  mkdir -p "${STAGING_DIR}"
  python "${BENCH_DIR}/calibrate_topk.py" \
    --model-name "${MODEL_NAME}" \
    --topk-val "${TOPK_VAL}" \
    --page-size 1 \
    --mem "${MEM}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --min-free-disk-gb "${MIN_FREE_DISK_GB}" \
    --vortex-module-name "${ALGO}" \
    --output-dir "${STAGING_DIR}" \
    2>&1 | tee "${SWEEP_DIR}/step0_calibrate.log"
  mv -f "${STAGING_DIR}/raw_histograms.npy" "${DEFAULT_REAL_HIST}"
  REAL_HIST_PATH="${DEFAULT_REAL_HIST}"
  echo ">>> Step 0: Done. raw_histograms -> ${REAL_HIST_PATH}"
fi

# ── Sweep ──────────────────────────────────────────────────────
SWEEP_INDEX="${SWEEP_DIR}/sweep_index.json"
echo "{" > "${SWEEP_INDEX}"
echo "  \"axis_name\": \"block_size\"," >> "${SWEEP_INDEX}"
echo "  \"axis_type\": \"kernel\"," >> "${SWEEP_INDEX}"
echo "  \"model_name\": \"${MODEL_NAME}\"," >> "${SWEEP_INDEX}"
echo "  \"topk_val\": ${TOPK_VAL}," >> "${SWEEP_INDEX}"
echo "  \"mapping_modes\": [${MAPPING_MODES// /, }]," >> "${SWEEP_INDEX}"
echo "  \"cells\": [" >> "${SWEEP_INDEX}"

FIRST_CELL=1
for BLOCK_SIZE in ${BLOCK_SIZES}; do
  # Pick a seq_len that satisfies pages/seg > topk_val + 3 reserved.
  MIN_SEQ_LEN=$(( (TOPK_VAL + 4) * BLOCK_SIZE ))
  SEQ_LEN=${MIN_SEQ_LEN}
  # Round up to next power-of-two-ish multiple of 1024 for stable timing.
  if [ "${SEQ_LEN}" -lt 8192 ]; then SEQ_LEN=8192; fi

  CELL_DIR="${SWEEP_DIR}/bs${BLOCK_SIZE}"
  mkdir -p "${CELL_DIR}"
  AUTOTUNE_JSON="${CELL_DIR}/autotune_results.json"
  REMAP_JSON="${CELL_DIR}/remap_bench.json"

  echo ""
  echo "============================================================"
  echo ">>> Cell: block_size=${BLOCK_SIZE}  seq_len=${SEQ_LEN}"
  echo "============================================================"

  echo ">>> Autotuning hparams for block_size=${BLOCK_SIZE}"
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

  echo ">>> Remap bench for block_size=${BLOCK_SIZE}"
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
      "axis_value": ${BLOCK_SIZE},
      "axis_label": "bs${BLOCK_SIZE}",
      "seq_len": ${SEQ_LEN},
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
PYTHONPATH="${SCRIPT_DIR}/.." python3 - "${SWEEP_INDEX}" "${SELECTED_TXT}" <<'PY'
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
try:
    from autotune_topk_mapping import MODE_NAMES, PARAM_NAME
except Exception:
    MODE_NAMES = {0: "none", 3: "power", 6: "asinh", 7: "log1p", 9: "erf",
                  10: "tanh", 11: "subtract", 13: "exp_stretch",
                  15: "shift_pow2", 16: "shift_pow3", 17: "linear_steep"}
    PARAM_NAME = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha",
                  11: "pivot", 13: "alpha", 15: "pivot", 16: "pivot", 17: "k"}

DISPLAY = {3: "Power", 6: "Asinh", 7: "Log1p", 9: "Erf", 10: "Tanh",
           11: "Subtract", 13: "ExpStretch", 15: "ShiftPow2",
           16: "ShiftPow3", 17: "LinearSteep"}

idx_path, out_path = sys.argv[1], sys.argv[2]
with open(idx_path) as f:
    idx = json.load(f)

lines = ["== Selected mapping functions (autotuned, block_size sweep) =="]
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
        if m not in best:
            continue
        pname = PARAM_NAME.get(m, "p")
        pval = best[m].get("param", 0.0)
        parts.append(f"{DISPLAY[m]}({pname}={pval})")
    lines.append(f"[block_size={cell['axis_value']}] " + "  ".join(parts))

txt = "\n".join(lines) + "\n"
print(txt)
with open(out_path, "w") as f:
    f.write(txt)
PY

echo ""
echo "============================================================"
echo "Block-size ablation complete."
echo "  Sweep dir:        ${SWEEP_DIR}"
echo "  Per-cell results: ${SWEEP_DIR}/bs<N>/"
echo "  Sweep index:      ${SWEEP_INDEX}"
echo "  Selected hparams: ${SELECTED_TXT}"
echo "Run analyze with:"
echo "  python examples/analyze_ablation_remap.py --sweep-dir ${SWEEP_DIR}"
echo "============================================================"
