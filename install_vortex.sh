#!/usr/bin/env bash
# Build a reproducible Vortex + official SGLang 0.5.12 environment.

set -euo pipefail

ENV_NAME="${ENV_NAME:-vortex_v1}"
PY_VER="${PY_VER:-3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "${FORCE:-0}" = "1" ]; then
        conda env remove -y -n "$ENV_NAME"
    else
        echo "ERROR: conda env '$ENV_NAME' already exists. Re-run with FORCE=1." >&2
        exit 1
    fi
fi

echo ">>> [1/2] creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel

echo ">>> [2/2] installing Vortex, official SGLang, and research tools"
python -m pip install -e "$REPO_ROOT[sglang,research]"

python - <<'PY'
from importlib.metadata import entry_points, version

import vortex_torch

plugins = {ep.name: ep.value for ep in entry_points(group="sglang.srt.plugins")}
assert plugins.get("vortex") == "vortex_torch.engine.sgl.plugin:register", plugins
assert version("sglang") == "0.5.12.post1"
assert version("vortex-torch") == vortex_torch.__version__
print(f"sglang       : {version('sglang')}")
print(f"vortex_torch : {vortex_torch.__version__}")
print(f"plugin       : {plugins['vortex']}")
PY

echo ">>> done. Activate with: conda activate $ENV_NAME"
