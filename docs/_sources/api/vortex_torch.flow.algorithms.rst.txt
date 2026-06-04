vortex\_torch.flow.algorithms
=============================

.. automodule:: vortex_torch.flow.algorithms

.. currentmodule:: vortex_torch.flow.algorithms

Ready-made sparse-attention flows. Each entry shows the routing **math**
(rendered from the class docstring) followed by the **code** that implements
it — the ``__init__`` op set and the ``forward_cache`` / ``forward_indexer`` /
``create_cache`` methods.

.. note::
   ``GeMM(x, y) = y xᵀ``; reductions keep dims; ``q`` is ``[B, H_q, D]`` and a
   cache tensor is ``[S, r, D]`` in the indexer (page-packed) / ``[B, r, D]`` in
   the cache (batch-major). Register key is in ``@register(...)``.

block_sparse_attention
----------------------

.. autoclass:: BlockSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: BlockSparseAttention

gqa_block_sparse_attention
--------------------------

.. autoclass:: GQABlockSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: GQABlockSparseAttention

gqa_quest_sparse_attention
--------------------------

.. autoclass:: GQAQuestSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: GQAQuestSparseAttention

lserve_sparse_attention
-----------------------

.. autoclass:: LServeSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: LServeSparseAttention

lserve_centroid_sparse_attention
--------------------------------

.. autoclass:: LServeCentroidSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: LServeCentroidSparseAttention

masked_quest_sparse_attention
-----------------------------

.. autoclass:: MaskedQuestSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: MaskedQuestSparseAttention

centered_block_sparse_attention
-------------------------------

.. autoclass:: CenteredBlockSparseAttention

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: CenteredBlockSparseAttention

running_avg_block_sparse
------------------------

.. autoclass:: RunningAvgBlockSparse

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: RunningAvgBlockSparse

venergy_gated_centroid
----------------------

.. autoclass:: VEnergyGatedCentroid

.. literalinclude:: ../../vortex_torch/flow/algorithms.py
   :pyobject: VEnergyGatedCentroid
