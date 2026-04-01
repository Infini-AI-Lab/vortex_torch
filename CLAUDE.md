# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vortex is a lightweight, modular framework for building custom sparse attention algorithms for LLM inference. It provides a PyTorch-like frontend that abstracts away batching, caching, and paged attention, running on optimized backends (FlashInfer, CUDA Graph) via SGLang integration.

## Build & Install

```bash
# Clone with submodules
git clone -b v1 --recursive <repo-url>

# Install SGLang dependency (custom fork in third_party/, supports v0.4.9)
cd third_party/sglang && bash install.sh && cd ../../

# Install Vortex (editable mode, compiles CUDA extensions for SM_86/SM_89/SM_90)
pip install -e .
```

Requires Python >=3.10, torch>=2.7, lighteval[math]==0.12.2. CUDA extensions (`vortex_torch_C`) are built from `csrc/` (register.cc, utils_sglang.cu, topk.cu, topk_sglang.cu).

## Testing & Verification

There is no formal test suite (no pytest). Verification is done by running algorithms against SGLang reference output and comparing accuracy on math benchmarks.

```bash
# Single algorithm verification (from examples/ directory)
python examples/verify_algo.py --trials 2 --topk-val 30 --vortex-module-name block_sparse_attention

# Full options
python examples/verify_algo.py \
  --trials 8 --topk-val 30 \
  --vortex-module-name block_sparse_attention \
  --model-name Qwen/Qwen3-1.7B \
  --topk-type naive \
  --mem 0.7

# Batch test (outputs timestamped logs to examples/results/)
bash examples/verify_algo.sh

# AIM24 benchmark verification
python examples/verify_aim24.py
```

Available `--topk-type` values: `naive` (CUB-based), `sglang` (SGLang-integrated kernel).

## AI-Powered Algorithm Generation

```bash
# Generate new sparse attention algorithms via OpenHands (requires LLM_API_KEY env var)
python openhands_gen.py
```

Note: Some auto-generated operators may not be fully optimized. Tune `mem_fraction_static` if OOM occurs.

## Building Documentation

```bash
make -C docs html
```

Uses Sphinx with myst_parser and furo theme. Deployed via GitHub Actions on push to v1 branch.

## Architecture

### Core Abstraction: vFlow (`vortex_torch/flow/flow.py`)

All sparse attention algorithms inherit from `vFlow` and implement three methods:

- **`forward_indexer(q, o, cache, ctx)`** — Compute sparse page indices from queries. Operates on page-packed tensor view `[S, r, c]`.
- **`forward_cache(cache, loc, ctx)`** — Update/summarize custom cache tensors when a page completes. Operates on batch-major view `[B, r, c]`.
- **`create_cache(page_size, head_dim)`** — Declare custom cache tensor shapes as a dict of `{name: (rows, cols)}`.

Algorithms are registered via `@register("name")` decorator and instantiated with `build_vflow()`.

### Operator System (`vortex_torch/indexer/`, `vortex_torch/cache/`)

Operators (`vOp` subclasses) run in two modes:
- **Profile mode**: Pre-compute output shapes and allocate buffers
- **Execute mode**: Perform actual GPU computation

Operators are split into two parallel hierarchies:
- **Indexer ops** (`vortex_torch/indexer/`): GeMM, GeMV, topK, reduce (Mean/Max/Min/Sum/L2Norm), softmax, elementwise, transpose, save/load
- **Cache ops** (`vortex_torch/cache/`): GeMM, reduce, elementwise, fill, KV buffer setup

Both use Triton kernels (in respective `triton_kernels/` subdirectories) for GPU execution.

### Tensor Format (`vortex_torch/abs/tensor.py`)

`vTensor` wraps `torch.Tensor` with format metadata (BATCHED, RAGGED, PAGED) to enforce layout consistency across operations.

### Context System (`vortex_torch/abs/context_base.py`)

`ContextBase` carries per-step runtime state. Specialized as:
- `Indexer.Context`: Page layout, head config, hardware info
- `Cache.Context`: Page size, total pages, model info

### Concrete Algorithms (`vortex_torch/flow/algorithms.py`)

- **BlockSparseAttention**: Centroid-based routing (query avg → GeMV with centroids → topK)
- **GQABlockSparseAttention**: Grouped-query variant with softmax + group aggregation
- **GQAQuestSparseAttention**: Query-envelope matching using per-page max/min bounds

### Algorithm Registry (`vortex_torch/flow/registry.py`)

Algorithms are registered via `@register("name")` and looked up with `get(name)`, `has(name)`, `list_keys()`. Factory: `build_vflow(name)` in `loader.py`.

### SGLang Integration

Custom SGLang fork lives in `third_party/sglang` (git submodule, "graph" branch). CUDA extensions in `csrc/` provide PyBind11 bindings for `sglang_plan_decode`, `sglang_plan_prefill`, transpose operations (NH↔HN), and top-K output routing.

## Key Conventions

- **Tensor shapes**: Query `[B, H_q, D]`, sparse output `[S_sparse, 1, 1]`, cache indexer-view `[S, r, c]`, cache batch-view `[B, r, c]`
- **GeMM semantics**: `GeMM(x, y)` computes `y @ x^T` (note transposition)
- **Standard cache keys**: `"k"` and `"v"` have inner shape `(page_size, head_dim)`; custom caches declared in `create_cache()`
- **Branch**: Main development is on `v1`

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.