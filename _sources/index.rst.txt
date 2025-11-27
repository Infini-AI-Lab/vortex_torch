Vortex
============


Installation
------------

.. code-block:: bash

   git clone -b v1 --recursive https://github.com/Infini-AI-Lab/vortex_torch.git
   cd third_party/sglang
   bash install.sh
   cd ../../
   cd vortex_torch
   pip install -e .

Quick Example
-------------
.. code-block:: python

   @register("custom_sparse_attention")
   class CustomSparseAttention(vFlow):
    
    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemv = GeMV()
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor, # viewed as [1, H_q, D]
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor], # viewed as [S, r, c] depending on create_cache()
        ctx: ContextBase,
    ):  
        q_mean = q.mean(dim=1, keepdim=True)
        score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor], # viewed as [B, r, c] depending on create_cache()
        loc: torch.Tensor,
        ctx: ContextBase,
    ):  
        # computation is triggered only when a page is finished
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        
        return {
            "centroids": (1, head_dim),
        }

   

.. code-block:: python

   llm = sgl.Engine(model_path="Qwen/Qwen3-0.6B", 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_topk_val=30,   
                    disable_overlap_schedule=True,  # Mandatory
                    attention_backend="flashinfer", # Mandatory
                    enable_vortex_sparsity=True, # otherwise will compute full attention
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=1,
                    vortex_layers_skip=list(range(1)), # full attention for layer 0
                    vortex_module_path="path/to/custom_sparse_attention.py" #if not specify, vortex will try to search in vortex_torch.flow.algorithms
                    vortex_module_name="custom_sparse_attention",
                    vortex_max_seq_lens=8192,
                    mem_fraction_static=0.6
                    )

API Reference
-------------

.. autosummary::
   :toctree: api
   :recursive:

   vortex_torch.indexer
   vortex_torch.cache
   vortex_torch.flow
