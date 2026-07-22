# Installation

Vortex is an out-of-tree plugin for the official `sglang==0.5.12.post1`
package. Installation does not require a GPU; Vortex's custom kernels are
JIT-compiled when a GPU server starts.

## From source

```bash
git clone https://github.com/Infini-AI-Lab/vortex_torch.git
cd vortex_torch
pip install -e ".[sglang]"
```

SGLang discovers the installed Vortex hook through the
`sglang.srt.plugins` entry-point group. The old `third_party/sglang/v0.5.9`
tree remains only as historical reference material. It is not installed,
tested, or supported by the current runtime.

Install the benchmark and dataset dependencies as a separate extra:

```bash
pip install -e ".[sglang,research]"
```

## Reproducible conda environment (recommended)

The repo ships a one-shot script that creates a Python 3.12 environment and
installs the same official SGLang/Vortex plugin stack:

```bash
bash install_vortex.sh          # creates the `vortex_v1` conda env
conda activate vortex_v1
```

`install_vortex_glm.sh` now delegates to the same stack under the historical
`vortex_glm` environment name; SGLang 0.5.12 already uses Transformers 5.x.

## Verify

```bash
python -c "from importlib.metadata import entry_points; assert any(e.name == 'vortex' for e in entry_points(group='sglang.srt.plugins')); print('vortex plugin ok')"
```

You should see `vortex plugin ok`. You're ready for the
[Quick Start](quickstart.md).
