#!/usr/bin/env bash
#
# DEPRECATED under sglang 0.5.16 — use install_vortex.sh instead.
#
# This script existed solely to work around a transformers version split: GLM-4.7-Flash
# (HF model type `glm4_moe_lite`) REQUIRES transformers >= 5.0, while sglang 0.5.9 and
# vortex_torch both pinned `transformers==4.57.1`. So `vortex_glm` installed the 4.57.1
# stack and then OVERRODE transformers with a GLM-supporting git commit.
#
# sglang 0.5.16 pins `transformers==5.12.1` outright, so the default `vortex_v1` env
# built by install_vortex.sh already loads GLM-4.7-Flash. There is no split left and
# no override step to perform — this script is kept only for reproducing the old
# 0.5.9-era environment, and it still points at the vendored v0.5.9 tree.
#
# Captured from the working env: python 3.12, torch 2.9.1+cu128, flashinfer 0.6.3,
# sglang (editable, vendored v0.5.9), transformers @ 76732b4 (5.0.0.dev0).
#
# Usage:
#   bash install_vortex_glm.sh            # create the env
#   FORCE=1 bash install_vortex_glm.sh    # remove an existing env first
#   ENV_NAME=glm2 bash install_vortex_glm.sh   # build under a different name
#
# Install only needs CPU (all kernels are prebuilt wheels or JIT-compiled at
# runtime), so it works even while the GPUs are busy.

set -euo pipefail

ENV_NAME="${ENV_NAME:-vortex_glm}"
PY_VER="${PY_VER:-3.12}"
# transformers commit that ships glm4_moe_lite support (== 5.0.0.dev0 in the env).
TRANSFORMERS_COMMIT="${TRANSFORMERS_COMMIT:-76732b4e7120808ff989edbd16401f61fa6a0afa}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$REPO_ROOT/third_party/sglang/v0.5.9/sglang/python"
[ -d "$SGLANG_DIR" ] || { echo "ERROR: vendored sglang not found at $SGLANG_DIR" >&2; exit 1; }

# ---- conda bootstrap -------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo ">>> removing existing env '$ENV_NAME' (FORCE=1)"
        conda env remove -y -n "$ENV_NAME"
    else
        echo "ERROR: conda env '$ENV_NAME' already exists. Re-run with FORCE=1 to recreate." >&2
        exit 1
    fi
fi

echo ">>> [1/5] creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# ---- 2. torch (CUDA 12.8 build) pinned first ------------------------------
# Pinned before sglang sees `torch==2.9.1` so the exact CUDA build is locked in
# and torchvision/torchaudio match. (Default PyPI torch 2.9.1 is the cu128 wheel.)
echo ">>> [2/5] installing torch 2.9.1 + torchvision 0.24.1 + torchaudio 2.9.1 (cu128)"
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# ---- 3. sglang (editable, vendored) ---------------------------------------
# Pulls the runtime tree: flashinfer_python/cubin 0.6.3, sgl-kernel, xgrammar,
# outlines, cuda-python, etc. (and transformers 4.57.1, overridden in step 5).
echo ">>> [3/5] installing vendored sglang (editable) from $SGLANG_DIR"
pip install -e "$SGLANG_DIR"

# ---- 4. vortex_torch (editable) -------------------------------------------
echo ">>> [4/5] installing vortex_torch (editable) from $REPO_ROOT"
pip install -e "$REPO_ROOT"

# ---- 5. transformers override (GLM-4.7 / glm4_moe_lite) -------------------
# LAST so it wins over the transformers==4.57.1 pins above. pip will print a
# dependency-conflict warning about that pin — expected and harmless here.
echo ">>> [5/5] overriding transformers with glm4_moe_lite commit ${TRANSFORMERS_COMMIT:0:12}"
pip install --upgrade "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_COMMIT}"

# ---- verify ----------------------------------------------------------------
echo ">>> verifying the environment"
python - <<'PY'
import torch, transformers, sglang, vortex_torch
import flashinfer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print(f"  python        : {__import__('sys').version.split()[0]}")
print(f"  torch         : {torch.__version__}  (cuda {torch.version.cuda})")
print(f"  transformers  : {transformers.__version__}")
print(f"  flashinfer    : {flashinfer.__version__}")
print(f"  sglang        : {sglang.__version__}")
print(f"  vortex_torch  : {getattr(vortex_torch, '__version__', '?')}")
ok = "glm4_moe_lite" in CONFIG_MAPPING_NAMES
print(f"  glm4_moe_lite recognized: {ok}")
assert ok, "transformers does not recognize glm4_moe_lite — GLM-4.7-Flash will NOT load"
PY

echo ""
echo ">>> done. Activate with:  conda activate $ENV_NAME"
echo ">>> sanity run (needs a free GPU):"
echo "    conda activate $ENV_NAME"
echo "    CUDA_VISIBLE_DEVICES=<gpu> python examples/ruler/run_ruler_mla.py --n 20 --dump"
