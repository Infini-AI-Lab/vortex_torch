#!/usr/bin/env bash
# Compatibility wrapper: SGLang 0.5.12 already carries Transformers 5.x.

set -euo pipefail

export ENV_NAME="${ENV_NAME:-vortex_glm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/install_vortex.sh"
