# Installation

Vortex plugs into a **vendored SGLang** (under `third_party/`). Install SGLang
in editable mode first, then Vortex. Installation is CPU-only — all kernels are
prebuilt wheels or JIT-compiled at runtime — so it works even while the GPUs are
busy.

## From source

```bash
git clone --recursive https://github.com/Infini-AI-Lab/vortex_torch.git
cd vortex_torch

# 1. SGLang dependency (vendored, editable)
cd third_party/sglang/v0.5.9/sglang
pip install -e "python"
cd ../../../../

# 2. Vortex (editable)
pip install -e .
```

If you cloned without `--recursive`, pull the submodules first:

```bash
git submodule update --init --recursive
```

## Reproducible conda environment (recommended)

The repo ships a one-shot script that builds the exact tested environment —
Python 3.12, torch 2.9.1+cu128, flashinfer 0.6.3, transformers 4.57.1, plus
editable SGLang and Vortex:

```bash
bash install_vortex.sh          # creates the `vortex_v1` conda env
conda activate vortex_v1
```

```{note}
For **MLA models** such as GLM-4.7-Flash (HF type `glm4_moe_lite`, which
requires `transformers >= 5.0`), use `install_vortex_glm.sh` instead — it builds
a separate `vortex_glm` env that overrides transformers with a GLM-supporting
build.
```

## Verify

```bash
python -c "import torch, sglang, vortex_torch, flashinfer; print('vortex ok')"
```

You should see `vortex ok` with no import errors. You're ready for the
[Quick Start](quickstart.md).
