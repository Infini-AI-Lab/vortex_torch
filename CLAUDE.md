# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vortex is a lightweight, modular framework for building custom sparse attention algorithms for LLM inference. It provides a PyTorch-like frontend that abstracts away batching, caching, and paged attention, running on optimized backends (FlashInfer, CUDA Graph) via SGLang integration.

## Build & Install

```bash
# Install SGLang dependency (custom fork in third_party/)
cd third_party/sglang && bash install.sh && cd ../../

# Install Vortex (editable mode, compiles CUDA extensions for SM_89/SM_90)
pip install -e .
```

Requires Python >=3.10, torch>=2.7. CUDA extensions are built from `csrc/` (register.cc, utils_sglang.cu, topk.cu).

## Running Examples

```bash
# Single algorithm verification against SGLang
python examples/verify_algo.py --trials 2 --topk-val 30 --vortex-module-name block_sparse_attention

# Batch test multiple algorithms
bash examples/verify_algo.sh
```

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

### SGLang Integration

Custom SGLang fork lives in `third_party/sglang` (git submodule, "graph" branch). CUDA extensions in `csrc/` provide PyBind11 bindings for `sglang_plan_decode`, `sglang_plan_prefill`, and transpose operations.

## Key Conventions

- **Tensor shapes**: Query `[B, H_q, D]`, sparse output `[S_sparse, 1, 1]`, cache indexer-view `[S, r, c]`, cache batch-view `[B, r, c]`
- **GeMM semantics**: `GeMM(x, y)` computes `y @ x^T` (note transposition)
- **Standard cache keys**: `"k"` and `"v"` have inner shape `(page_size, head_dim)`; custom caches declared in `create_cache()`
- **Branch**: Main development is on `v1`
