Vortex
======

**Vortex** turns sparse-attention algorithm design into something *AI agents can
do*. You describe a sparse-attention **flow** in a few lines of high-level Python
ops, and Vortex compiles it into fused Triton/CUDA kernels that plug straight
into `SGLang <https://github.com/sgl-project/sglang>`_'s decode loop — no manual
kernel writing, and the result runs (and is benchmarked) inside a real serving
stack.

A flow is just three methods on a :class:`~vortex_torch.flow.flow.vFlow`:

* **create_cache** — declare the auxiliary per-page state you want to keep
  (e.g. a centroid, a min/max envelope) alongside the K/V cache.
* **forward_cache** — fill that state from the keys/values as each page
  completes (runs once per page).
* **forward_indexer** — score the cached pages against the query and emit the
  sparse set of pages to attend to (runs every decode step).

See :doc:`quickstart` for the shortest end-to-end path, :doc:`examples` for
detailed recipes (custom flows, ``VortexConfig``, a programmable budget, MLA
models, and server mode), and the :doc:`API reference <api>` for the full op set.

.. note::

   Looking for the big picture and benchmarks? See the
   `project page <https://infini-ai-lab.github.io/vortex_torch/>`_ and the
   `paper <https://arxiv.org/abs/2606.06453>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Guides

   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
