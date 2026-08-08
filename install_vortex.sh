#!/usr/bin/env bash
#
# Reproducible build of the `vortex_v1` conda environment — the default env for
# this project (all the slash commands, RULER, and AIME24 runners expect it).
#
# sglang 0.5.16 pins transformers==5.12.1, so this single env ALSO loads
# GLM-4.7-Flash (`glm4_moe_lite`, which needs transformers >= 5). The separate
# `vortex_glm` env that install_vortex_glm.sh built for the transformers-4/5
# split is therefore obsolete under 0.5.16 — use this script for every model.
#
# Dependency policy: sglang, sglang-kernel, torch, flashinfer, transformers and
# everything else are installed by ONE pip resolve, driven by the vendored
# sglang's own `python/pyproject.toml`. Do NOT pre-pin torch or upgrade
# individual packages afterwards — 0.5.16's set is mutually version-locked
# (torch 2.11.0 / flashinfer 0.6.14[cu13] / sglang-kernel 0.4.5 / CUDA 13) and
# piecewise installs silently produce an ABI-mismatched env.
#
# Resulting env: python 3.12, torch 2.11.0, flashinfer_python 0.6.14,
# sglang-kernel 0.4.5, transformers 5.12.1, sglang (editable, vendored
# v0.5.16), vortex_torch (editable).
#
# Usage:
#   bash install_vortex.sh            # create the env
#   FORCE=1 bash install_vortex.sh    # remove an existing env first
#   ENV_NAME=vortex2 bash install_vortex.sh   # build under a different name
#
# Install only needs CPU (all kernels are prebuilt wheels or JIT-compiled at
# runtime), so it works even while the GPUs are busy. Note the runtime needs a
# CUDA 13-capable driver.

set -euo pipefail

ENV_NAME="${ENV_NAME:-vortex_v1}"
PY_VER="${PY_VER:-3.12}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$REPO_ROOT/third_party/sglang/v0.5.16/sglang/python"
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

echo ">>> [1/3] creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# ---- 2. sglang + vortex_torch in ONE resolve ------------------------------
# Both editable, installed together so pip solves sglang's pinned stack
# (torch 2.11.0, flashinfer_python[cu13] 0.6.14, sglang-kernel 0.4.5,
# transformers 5.12.1, cuda-python>=13, ...) and vortex_torch's requirements
# simultaneously. A conflict surfaces here as a resolver error rather than as a
# silently broken env.
echo ">>> [2/3] installing vendored sglang + vortex_torch (single resolve)"
pip install -e "$SGLANG_DIR" -e "$REPO_ROOT"

# ---- 3. verify -------------------------------------------------------------
echo ">>> [3/3] verifying the environment"
python - <<'PY'
import sys

import torch, transformers, sglang, vortex_torch
import flashinfer
print(f"  python        : {sys.version.split()[0]}")
print(f"  torch         : {torch.__version__}  (cuda {torch.version.cuda})")
print(f"  transformers  : {transformers.__version__}")
print(f"  flashinfer    : {flashinfer.__version__}")
print(f"  sglang        : {sglang.__version__}")
print(f"  vortex_torch  : {getattr(vortex_torch, '__version__', '?')}")

# sglang 0.5.16 pins transformers 5.x; GLM-4.7-Flash needs >= 5.
ok = int(transformers.__version__.split(".")[0]) >= 5
print(f"  transformers >= 5 (sglang 0.5.16 pin, GLM-capable): {ok}")
assert ok, f"expected transformers >= 5, got {transformers.__version__}"

# The vortex hooks must be live in the vendored tree.
from sglang.srt.server_args import ServerArgs
assert hasattr(ServerArgs, "_VORTEX_LEGACY_DEFAULTS"), (
    "sglang is installed but the vortex patch is missing — check that "
    "SGLANG_DIR points at third_party/sglang/v0.5.16/sglang/python"
)
print("  vortex sglang hooks: present")
PY

echo ""
echo ">>> done. Activate with:  conda activate $ENV_NAME"
echo ">>> sanity run (needs a free GPU):"
echo "    conda activate $ENV_NAME"
echo "    CUDA_VISIBLE_DEVICES=<gpu> python algorithm_scientist/run_ruler.py --config <submission>.json"
