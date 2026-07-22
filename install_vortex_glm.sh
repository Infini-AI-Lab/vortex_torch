#!/usr/bin/env bash
# Compatibility wrapper around the shared installer, with a GLM capability
# check so dependency resolution cannot silently select an incompatible
# Transformers build.

set -euo pipefail

export ENV_NAME="${ENV_NAME:-vortex_glm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/install_vortex.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python - <<'PY'
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

supported = "glm4_moe_lite" in CONFIG_MAPPING_NAMES
print(f"transformers          : {transformers.__version__}")
print(f"glm4_moe_lite support : {supported}")
assert supported, (
    "Transformers does not recognize glm4_moe_lite; "
    "the GLM-4.7-Flash environment is incomplete."
)
PY
