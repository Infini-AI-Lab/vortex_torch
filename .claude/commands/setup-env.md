---
description: Establish a working python environment for vortex_torch BEFORE any other work — detect what's on the host (conda/uv/venv/docker), pick or build one where `import vortex_torch` works, and return the run prefix to use all session. Handles the GLM (transformers 5) split.
argument-hint: [--model <hf-id>]   (so it can also ensure the GLM env when needed)
---

**Run this first.** Do not assume `conda activate vortex_v1` exists — the env may
be a different conda env, a uv/venv, or docker, and **GLM models need a newer
transformers**. Resolve the environment yourself, then use the resulting **run
prefix** for every python call this session.

## Step 0 — detect

```bash
python algorithm_scientist/detect_env.py
```
It probes conda envs, uv, venvs, docker, and the current interpreter for where
`import vortex_torch` works (with torch/transformers/sglang versions), and prints
a **RECOMMENDED** run prefix for the default vortex env and (separately) a GLM env
(transformers ≥ 5). `--no-import` for a fast tooling-only scan; `--json` for
machine output.

## Step 1 — adopt the prefix

Pick `RUN` from the recommendation, and use it as a **prefix on every python
call** this session (robust in non-interactive subshells):
```bash
RUN="conda run -n vortex_v1 python"     # or: ".venv/bin/python", "uv run python",
                                        #     "docker run --gpus all <img> python", etc.
$RUN -c "import sys,vortex_torch; print(sys.executable, vortex_torch.__version__)"
```
If the target `--model` is GLM-family, also set `RUN_GLM` to the GLM env's prefix
(transformers ≥ 5; see [AI/workflows/support_model.md](../../AI/workflows/support_model.md)).
The per-command `conda activate vortex_v1` snippets elsewhere are the *default* —
substitute your detected prefix when it differs.

## Step 2 — build one if none works

If `detect_env.py` finds no working vortex env, build it from the repo specs it
reports (`pyproject.toml`/`setup.py`/`environment.yml`/`requirements.txt`/
`Dockerfile`), preferring the tooling that's present:

- **conda**: `conda create -n vortex_v1 python=3.12 -y && conda run -n vortex_v1 pip install -e .`
- **uv**: `uv venv && uv pip install -e .`  (then `RUN="uv run python"`)
- **venv**: `python -m venv .venv && .venv/bin/pip install -e .`
- **docker**: build/run the repo `Dockerfile` (or a known image), `RUN="docker run --gpus all -v $PWD:/w -w /w <img> python"`.

Building the C extension may need a CUDA toolchain (`nvcc`) and a torch matching
the local CUDA — if the build fails, that's `/debug` territory (wrong CUDA/torch,
missing compiler). For **GLM**, build a *separate* env and upgrade transformers:
`conda create -n vortex_glm … && conda run -n vortex_glm pip install -e . && conda run -n vortex_glm pip install -U "transformers>=5"`.

## Step 3 — verify + report

```bash
$RUN -c "import vortex_torch, torch; print('ok', torch.__version__)"
```
Report: `tooling found | chosen RUN prefix | (RUN_GLM if built) | versions
(torch/transformers/sglang) | built? what`. Hand the `RUN` prefix to the rest of
the session; if `import vortex_torch` still fails after a build attempt, surface
the exact error (don't proceed to GPU work on a broken env).
