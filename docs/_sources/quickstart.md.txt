# Quick Start

A working setup is **two files**:

1. **The flow module** — a `.py` file that *defines* your sparse-attention
   algorithm as a `vFlow` subclass and `@register`s it under a name. It contains
   only Vortex ops; it never imports SGLang.
2. **The launch script** — imports `sglang` + `vortex_torch` and starts the
   engine, pointing at the flow by its registered name.

## 1. Define a flow — `custom_sparse_attention.py`

A `vFlow` declares its cache layout in `create_cache`, refreshes per-page state
in `forward_cache`, and scores/selects pages every decode step in
`forward_indexer`. Save this anywhere on disk:

```python
from typing import Dict
import torch

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("custom_sparse_attention")
class CustomSparseAttention(vFlow):

    def __init__(self):
        super().__init__()
        # Indexer-side ops (run every decode step)
        self.mean = Mean(dim=1)        # average over the query heads
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ
        self.output_func = topK()      # must end in topK / approxTopK
        # Cache-side ops (run once per finished page)
        self.reduction = CMean(dim=1)  # one centroid (mean key) per page

    def forward_indexer(self, q, o, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        # No native torch ops here — every tensor flows through Vortex ops.
        q_mean = self.mean(q, ctx=ctx)                          # [B, 1, D]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)  # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)                     # selected pages -> o

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc, ctx: ContextBase):
        # triggered only when a page is finished
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        # "k" and "v" are provided automatically — do not declare them
        return {"centroids": (1, head_dim)}
```

## 2. Launch SGLang with your flow

The launch script is a **separate file**. Importing `vortex_torch` is what wires
Vortex into SGLang (it installs the `ServerArgs` ↔ `VortexConfig` adapter), so the
import is required even though you don't call it directly. Every Vortex knob lives
in a single [`VortexConfig`](examples.md); passing it turns sparsity **on**.

```python
import sglang as sgl
import vortex_torch  # noqa: F401 — installs the VortexConfig adapter
from vortex_torch.engine.sgl.config import VortexConfig

llm = sgl.Engine(
    model_path="Qwen/Qwen3-0.6B",
    page_size=16,
    attention_backend="flashinfer",   # mandatory: SGLang's base backend
    disable_overlap_schedule=True,    # mandatory for Vortex sparsity
    mem_fraction_static=0.85,
    vortex=VortexConfig(
        module_path="path/to/custom_sparse_attention.py",
        module_name="custom_sparse_attention",  # the @register name
        topk_val=30,            # pages kept per query
        layers_skip=[0],        # layer 0 runs full/dense attention
        block_reserved_bos=1,   # always keep the first page (sink)
        block_reserved_eos=1,   # always keep the most-recent page
        max_seq_lens=8192,
    ),
)
```

## 3. Generate

```python
out = llm.generate(
    ["The capital of France is"],
    {"temperature": 0.0, "max_new_tokens": 32},
)
print(out[0]["text"])
```

That's the whole loop: write a flow, point the engine at it, generate — with the
sparse attention running as fused kernels inside SGLang. Next, see
[Examples](examples.md) for `VortexConfig` in full, a programmable per-sequence
budget, MLA models, and serving over HTTP.
