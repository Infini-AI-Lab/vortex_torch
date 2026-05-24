Vortex
======

**Vortex** is a JIT-compiled **sparse-attention** framework for fast LLM
decoding. You describe a sparse-attention *flow* in a few lines of high-level
Python ops, and Vortex compiles it into fused Triton kernels that plug straight
into `SGLang <https://github.com/sgl-project/sglang>`_'s decode loop — no manual
kernel writing required.

A flow is just three methods on a :class:`~vortex_torch.flow.flow.vFlow`:

* **create_cache** — declare the auxiliary per-page state you want to keep
  (e.g. a centroid, a min/max envelope) alongside the K/V cache.
* **forward_cache** — fill that state from the keys/values as each page
  completes (runs once per page).
* **forward_indexer** — score the cached pages against the query and emit the
  sparse set of pages to attend to (runs every decode step).

See :mod:`vortex_torch.flow.algorithms` for ready-made flows
(block-sparse, Quest-style envelopes, LServe sub-block centroids, …).

Installation
------------

.. code-block:: bash

   git clone --recursive https://github.com/Infini-AI-Lab/vortex_torch.git
   cd vortex_torch
   cd third_party/sglang/v0.5.9/sglang
   pip install -e "python"
   cd ../../../../
   pip install -e .

Quick Example
-------------

Define a custom flow — centroid-based block-sparse routing in a dozen lines:

.. code-block:: python

   @register("custom_sparse_attention")
   class CustomSparseAttention(vFlow):

       def __init__(self):
           super().__init__()
           # Indexer-side ops (run every decode step)
           self.gemv = GeMV()
           self.output_func = topK()
           # Cache-side ops (run once per finished page)
           self.reduction = CMean(dim=1)

       def forward_indexer(
           self,
           q: torch.Tensor,                  # viewed as [B, H_q, D]
           o: torch.Tensor,
           cache: Dict[str, torch.Tensor],   # viewed as [S, r, c] per create_cache()
           ctx: ContextBase,
       ):
           q_mean = self.mean(q, ctx=ctx)
           score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
           self.output_func(score, o, ctx=ctx)   # must end in topK / approxTopK

       def forward_cache(
           self,
           cache: Dict[str, torch.Tensor],   # viewed as [B, r, c] per create_cache()
           loc: torch.Tensor,
           ctx: ContextBase,
       ):
           # triggered only when a page is finished
           self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

       def create_cache(self, page_size: int, head_dim: int):
           # "k" and "v" are provided automatically — do not declare them
           return {"centroids": (1, head_dim)}

Then run it through an SGLang engine:

.. code-block:: python

   llm = sgl.Engine(
       model_path="Qwen/Qwen3-0.6B",
       page_size=16,
       attention_backend="flashinfer",      # SGLang's base backend
       enable_vortex_sparsity=True,          # otherwise computes full attention
       vortex_topk_val=30,                   # pages kept per request
       vortex_block_reserved_bos=1,          # always-attended prefix blocks
       vortex_block_reserved_eos=2,          # always-attended recent blocks
       vortex_layers_skip=[0],               # full attention for layer 0
       vortex_module_name="custom_sparse_attention",
       vortex_module_path="path/to/custom_sparse_attention.py",  # omit to search vortex_torch.flow.algorithms
       vortex_max_seq_lens=8192,
       mem_fraction_static=0.8,
   )

API Reference
-------------

.. autosummary::
   :toctree: api
   :recursive:

   vortex_torch.indexer
   vortex_torch.cache
   vortex_torch.flow
