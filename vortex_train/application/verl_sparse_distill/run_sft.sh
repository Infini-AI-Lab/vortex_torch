#!/usr/bin/env bash
# Sparse (or dense-control) SFT on Nemotron-SFT-Math-v4 with verl.
#
# Usage:
#   run_sft.sh sparse            # vortex sparse attention in training
#   run_sft.sh dense             # identical run, flash_attention_2 -- the control
#
# The two differ in exactly two settings (the attention implementation and whether the
# hook installs), so a quality difference is attributable to attention and nothing else.
#
# Why the geometry travels in the ENVIRONMENT: verl trains in Ray worker processes, so
# registering the attention function in this shell's python would not reach them. Ray
# propagates the launcher's env, and `sparse_hook.install_from_env()` runs on import in
# each worker. `--config-path` points hydra at this directory.
set -euo pipefail

MODE="${1:-sparse}"
REPO="${REPO:-/scratch/zhuominc/vortex_train}"
VERL="${VERL:-/scratch/zhuominc/verl}"
PY="${PY:-/scratch/zhuominc/venv-0516/bin/python}"
DATA="${DATA:-/scratch/zhuominc/data/nemotron_math}"
NGPU="${NGPU:-8}"
export HF_HOME="${HF_HOME:-/scratch/zhuominc/hf}"

# The training budget MUST match what vortex_torch serves, or the evaluation compares a
# model trained for one context restriction against another. These defaults mirror the
# serving config used elsewhere in this repo (topk 16 + bos 1 + local 1, block_kv 64,
# block_q 1 = per-token selection).
export VORTEX_SPARSE_ALGO="${VORTEX_SPARSE_ALGO:-block_topk}"
export VORTEX_SPARSE_TOPK="${VORTEX_SPARSE_TOPK:-16}"
export VORTEX_SPARSE_BLOCK_Q="${VORTEX_SPARSE_BLOCK_Q:-1}"
export VORTEX_SPARSE_BLOCK_KV="${VORTEX_SPARSE_BLOCK_KV:-64}"
export VORTEX_SPARSE_RESERVE_BOS="${VORTEX_SPARSE_RESERVE_BOS:-1}"
export VORTEX_SPARSE_RESERVE_LOCAL="${VORTEX_SPARSE_RESERVE_LOCAL:-1}"

case "$MODE" in
  sparse)
    export VORTEX_SPARSE_ENABLE=1
    ATTN=vortex_sparse
    EXP=qwen3_4b_nemotron_math_sparse
    ;;
  dense)
    export VORTEX_SPARSE_ENABLE=0
    ATTN=flash_attention_2
    EXP=qwen3_4b_nemotron_math_dense
    ;;
  *) echo "usage: $0 {sparse|dense}" >&2; exit 2 ;;
esac

# Both the trainer package and this repo must be importable in the workers: verl for the
# trainer, $REPO for `application.verl_sparse_distill.sparse_hook`.
export PYTHONPATH="${VERL}:${REPO}:${PYTHONPATH:-}"
# Interpolated by sft_sparse.yaml to locate the custom dataset class.
export VORTEX_TRAIN_REPO="${REPO}"

# Make every worker install the hook without touching verl's source.
#
# `sitecustomize.py` is imported automatically by EVERY interpreter that has its directory
# on PYTHONPATH -- including torchrun's and Ray's workers -- and it runs before user code.
# That ordering is required, not merely convenient: transformers validates
# `attn_implementation` against ALL_ATTENTION_FUNCTIONS inside PreTrainedModel.__init__
# and raises `Specified attn_implementation="vortex_sparse" is not supported` if the name
# is not registered by then. Registering in this shell, or after model construction, is
# too late.
#
# The directory is a real path in the repo (application/verl_sparse_distill/hookpath), not
# a mktemp one, so the mechanism is discoverable and cannot vanish between launcher and
# worker startup.
export PYTHONPATH="${REPO}/application/verl_sparse_distill/hookpath:${PYTHONPATH}"

echo "=== mode=$MODE attn=$ATTN gpus=$NGPU ==="
echo "=== budget: topk=$VORTEX_SPARSE_TOPK + bos=$VORTEX_SPARSE_RESERVE_BOS +"\
     "local=$VORTEX_SPARSE_RESERVE_LOCAL, block_kv=$VORTEX_SPARSE_BLOCK_KV,"\
     "block_q=$VORTEX_SPARSE_BLOCK_Q ==="

cd "$VERL"
# `$PY -m torch.distributed.run`, not bare `torchrun`: the bare command resolves to
# whichever torchrun is first on PATH (the system python3.10 here), and its workers then
# import from THAT interpreter -- failing on `No module named 'ray'` even though the venv
# has it. Going through the venv's python guarantees launcher and workers share it.
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" \
  -m verl.trainer.sft_trainer \
  --config-dir "${REPO}/application/verl_sparse_distill" \
  --config-name sft_sparse \
  data.train_files="${DATA}/train.parquet" \
  data.val_files="${DATA}/val.parquet" \
  model.override_config.attn_implementation="${ATTN}" \
  trainer.experiment_name="${EXP}" \
  "${@:2}"
