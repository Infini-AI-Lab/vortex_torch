r"""
Cache-side operator API.

This module exposes the core primitives used on the cache path:

- :class:`Context`:
  Runtime context carrying layout, paging, and auxiliary metadata.

- Reductions:
  :class:`Mean`, :class:`Max`, :class:`Min`, :class:`L2Norm`
  for per-page / per-request aggregation.

- Matrix–matrix/vector ops:
  :class:`GeMM` for generalized page-wise matmul on cached tensors.

- Unary elementwise ops:
  :class:`Relu`, :class:`Silu`, :class:`Sigmoid`, :class:`Abs`,
  :class:`Add_Mul`.

- Binary elementwise ops:
  :class:`Maximum`, :class:`Minimum`, :class:`Multiply`, :class:`Add`.

These building blocks are typically used inside vFlow cache update
pipelines (e.g., to maintain centroids, envelopes, or other summaries).
"""

from .context import Context
from .reduce import Mean, Max, Min, L2Norm
from .matmul import GeMM
from .elementwise import Relu, Silu, Sigmoid, Abs, Add_Mul
from .elementwise_binary import Maximum, Minimum, Multiply, Add
from .triton_kernels import set_kv_buffer_launcher, set_kv_buffer_int8_launcher, dequant_paged_int8_to_bf16_inplace


__all__ = [
    "set_kv_buffer_launcher",
    "set_kv_buffer_int8_launcher",
    "dequant_paged_int8_to_bf16_inplace",
    "Mean", "Max", "Min", "L2Norm",
    "GeMM",
    "Relu", "Silu", "Sigmoid", "Abs", "Add_Mul",
    "Maximum", "Minimum", "Multiply", "Add",
    "Context"
]

