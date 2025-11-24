Vortex Torch
============

A concise description of what your package does and why it exists.

Installation
------------

.. code-block:: bash

   git clone https://github.com/Infini-AI-Lab/vortex_torch.git
   cd vortex_torch
   pip install -e .

Quick Example
-------------
.. code-block:: python

   

.. code-block:: python

   llm = sgl.Engine(model_path="Qwen/Qwen3-0.6B", 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_topk_val=30,   
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=True,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=1,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_path="path/to/custom_sparse_attention.py"
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
