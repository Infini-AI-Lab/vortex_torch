# Examples

This page collects the detailed recipes: configuring a run with `VortexConfig`,
writing a programmable per-sequence budget, targeting **MLA** models, and serving
over HTTP. If you haven't yet, start with the [Quick Start](quickstart.md).

## Anatomy of a flow

Every flow is a `vFlow` with three methods, written entirely from
`vortex_torch.indexer.*` / `vortex_torch.cache.*` ops (no native torch):

| Method | Runs | Job |
| --- | --- | --- |
| `create_cache` | once at setup | Declare auxiliary per-page state (e.g. centroids). `"k"`/`"v"` are auto-provided — don't declare them. |
| `forward_cache` | once per finished page | Reduce keys/values into that state (cross-block reductions belong on the indexer side). |
| `forward_indexer` | every decode step | Score cached pages from the query and emit the selected pages. **Must end in** `topK(...)` or `approxTopK(tolerate_ratio=...)(...)`. |

See {mod}`vortex_torch.indexer` and {mod}`vortex_torch.cache` for the full op
set, and {mod}`vortex_torch.flow.algorithms` for ready-made flows (block-sparse,
Quest-style envelopes, LServe sub-block centroids, …).

## What is `VortexConfig`?

`VortexConfig` ({mod}`vortex_torch.engine.sgl.config`) is a single dataclass that
holds **every** Vortex hyper-parameter in one place, instead of ~18 loose
`vortex_*` arguments scattered across SGLang's `ServerArgs`. Its presence on the
engine is also the on/off switch: pass a `VortexConfig` and sparsity is enabled;
leave it out and the model runs ordinary dense attention.

```python
from vortex_torch.engine.sgl.config import VortexConfig

cfg = VortexConfig(
    module_name="custom_sparse_attention",
    topk_val=30,
    layers_skip=[0],
)
llm = sgl.Engine(model_path="Qwen/Qwen3-0.6B", attention_backend="flashinfer",
                 disable_overlap_schedule=True, vortex=cfg)
```

Every field, with what it controls and an example value:

| Field | Explanation | Example |
| --- | --- | --- |
| `module_path` | Path to your flow's `.py` file. `None` → search `vortex_torch.flow.algorithms`. | `"submissions/custom.py"` |
| `module_name` | The `@register(...)` name of the `vFlow` to load. Must match exactly. | `"custom_sparse_attention"` |
| `topk_val` | **Static page budget** — fixed minimum pages each sequence keeps. The core accuracy↔throughput knob. | `30` |
| `topk_ratio` | **Dynamic page budget** — a fraction of the sequence's pages; engine keeps `max(static floor, topk_ratio × pages)`. `0.0` disables it. | `0.0625` |
| `max_topk_val` | Upper bound on the selected-page count, used to size/pick the top-k kernel. `None` → derived from `max_seq_lens`. | `256` |
| `layers_skip` | Layer indices that **bypass sparse attention and run dense**. `None` → all sparse. | `[0, 4, 8, 12]` |
| `block_reserved_bos` | Pages at the **start** always selected (attention sink). Int ≥ 1. | `1` |
| `block_reserved_eos` | Pages at the **end** (most recent) always selected. Int ≥ 1. | `1` |
| `max_seq_lens` | Maximum sequence length to plan buffers for. `-1` → model default. | `8192` |
| `block_size` | Vortex **page size** (unit of sparsity). Power of 2; defaults to SGLang's `page_size`. | `16` |
| `workload_chunk_size` | Planner granularity — blocks grouped into one indexer workload. Power of 2. | `32` |
| `dtype` | dtype for **intermediate** indexer tensors. | `"bfloat16"` |
| `compilation_cache_dir` | Directory for the JIT kernel cache. `None` → next to the compiler module. | `"~/.vortex_compilation_cache"` |
| `schedule_policy` | A CUDA C++ snippet computing each sequence's page budget (see below). `None` → default formula. | `None` |
| `attention_backend` | Sparse-attention kernel family: `"flashinfer"` (default) or `"trtllm"`. | `"flashinfer"` |
| `impl_backend` | Indexer op implementation backend: `"triton"` (default) or `"cuda"`. | `"triton"` |
| `use_tensor_core` | Tensor-core (`bf16 tl.dot`) codegen in the Triton kernel. Triton-only. | `False` |

```{tip}
**Budget recap:** pages attended per sequence ≈
`min(num_pages, max(topk_val + bos + eos, topk_ratio × num_pages))`.
`topk_val` dominates short sequences, `topk_ratio` long ones.
```

The legacy flat form —
`sgl.Engine(enable_vortex_sparsity=True, vortex_topk_val=30, vortex_module_name=..., ...)`
— still works (the adapter folds those `vortex_*` kwargs into a `VortexConfig`),
but the explicit object is clearer and self-documenting.

## Programmable budget — the `schedule_policy`

Instead of a fixed formula, the per-sequence **page budget can be computed by a
CUDA C++ snippet you provide**. Vortex injects it as the body of a `__device__`
function, JIT-compiles it into the decode planner (cached by content hash), and
runs it for every sequence on every backend. The default body *is* the standard
budget formula:

```cpp
// default schedule_policy — returns the number of pages to attend to.
const int static_kv_budget  = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
```

The snippet must `return` an `int`. In scope: `cached_block_len` (the sequence's
length in pages), `topk_val`, `topk_ratio`, `block_reserved_bos`,
`block_reserved_eos`. Because it's real device code, you can express budgets the
scalar knobs can't — e.g. a length-adaptive budget that grows slowly and caps:

```python
vortex=VortexConfig(
    module_name="custom_sparse_attention",
    topk_val=32,
    schedule_policy=r"""
        // base budget + 1 extra page per 64 cached pages, capped at 256
        const int base  = topk_val + block_reserved_bos + block_reserved_eos;
        const int extra = cached_block_len / 64;
        return min(base + extra, 256);
    """,
)
```

The planner is JIT-compiled once per distinct snippet, so there's no per-step
overhead.

## MLA models (DeepSeek-V3 / GLM-4.7 / Kimi-style)

Models with **Multi-head Latent Attention** compress the KV cache into a single
shared low-rank *latent*. Vortex supports them with a parallel base class,
{class}`~vortex_torch.flow.flow_mla.vFlowMLA`:

- The cache exposes **one auto-provided field, `cache["latent"]`** (the fused
  `[ kv_c | k_pe ]`) — there is no `"k"`/`"v"`.
- `create_cache(block_size, kv_lora_rank, qk_rope_head_dim)` declares only your
  aux tensors.
- `forward_indexer` receives the **fused absorbed query** `[ q_nope_out | q_pe ]`;
  a single dot `⟨q, centroid⟩` equals the full decode logit (RoPE included).

```python
from typing import Dict
import torch

from vortex_torch.flow import vFlowMLA, register
from vortex_torch.indexer import GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("rope_aware_block_sparse_mla")
class RopeAwareBlockSparseMLA(vFlowMLA):

    def __init__(self):
        super().__init__()
        self.mean = Mean(dim=1)        # average the fused query over its H heads
        self.gemm = GeMM()             # per-page score
        self.output_func = topK()
        self.reduction = CMean(dim=1)  # centroid = mean of the fused latent per page

    def forward_indexer(self, q, o, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.mean(q, ctx=ctx)                          # [B, 1, latent_dim]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)  # [S, 1, 1] — FULL logit
        self.output_func(score, o, ctx=ctx)

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc, ctx: ContextBase):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        # "latent" is auto-provided — declare only the aux centroid (full width).
        return {"centroids": (1, kv_lora_rank + qk_rope_head_dim)}
```

Launching is the same `VortexConfig` flow, with the **MLA decode backend** on the
engine and the tensor-core indexer enabled:

```python
import sglang as sgl
import vortex_torch  # noqa: F401
from vortex_torch.engine.sgl.config import VortexConfig

llm = sgl.Engine(
    model_path="zai-org/GLM-4.7-Flash",   # any MLA model (DeepSeek-V3, Kimi, …)
    trust_remote_code=True,
    page_size=32,
    attention_backend="trtllm_mla",       # Vortex CUDA MLA decode kernel
    kv_cache_dtype="auto",
    mem_fraction_static=0.9,
    vortex=VortexConfig(
        module_name="rope_aware_block_sparse_mla",
        attention_backend="trtllm",       # 2D block-table indexer
        impl_backend="triton",
        use_tensor_core=True,
        block_size=32,
        topk_val=61,
        block_reserved_bos=1,
        block_reserved_eos=2,
        max_seq_lens=8192,
    ),
)
```

A runnable single-GPU MLA demo lives in `examples/run_ruler_mla.py`.

## Server mode (OpenAI-compatible endpoint)

To serve Vortex over HTTP, use `examples/server_launch.sh`, which boots an SGLang
server with an OpenAI-compatible API on `127.0.0.1:30000`:

```bash
# ./server_launch.sh <MODEL_NAME> <TP_SIZE>
examples/server_launch.sh Qwen/Qwen3-4B 1
```

Two details make server mode work:

1. **`import vortex_torch` must run first.** The script doesn't call
   `python -m sglang.launch_server` directly — that builds `ServerArgs` before
   Vortex is imported, so the adapter wouldn't be installed yet. It imports
   `vortex_torch`, then calls SGLang's `run_server`, so the `ServerArgs` ↔
   `VortexConfig` adapter is in place before the args are pickled to the worker.
2. **Knobs are passed as JSON via `--vortex-config`.** The per-knob `--vortex-*`
   flags no longer exist; the script writes the `VortexConfig` fields (prefix
   stripped) to a temp JSON file and feeds it through `--vortex-config '<json>'`.
   A non-null config implicitly enables sparsity.

Query it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "Hello!"}]}'
```
