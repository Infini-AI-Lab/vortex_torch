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
  :class:`Add_Mul`, :class:`Log`, :class:`Exp`.

- Binary elementwise ops:
  :class:`Maximum`, :class:`Minimum`, :class:`Multiply`, :class:`Add`,
  :class:`WhereEqual`, :class:`WhereNotEqual`, :class:`WhereGreater`,
  :class:`WhereGreaterEqual`, :class:`WhereLess`, :class:`WhereLessEqual`.

These building blocks are typically used inside vFlow cache update
pipelines (e.g., to maintain centroids, envelopes, or other summaries).
"""

from .context import Context, ctx, get_ctx
from .reduce import Mean, Max, Min, L2Norm
from .reduce_interleave import (
    MeanInterleave, MaxInterleave, MinInterleave, L2NormInterleave,
)
from .matmul import GeMM
from .elementwise import Relu, Silu, Sigmoid, Abs, Add_Mul, Log, Exp
from .elementwise_binary import (
    Maximum, Minimum, Multiply, Add,
    WhereEqual, WhereNotEqual, WhereGreater,
    WhereGreaterEqual, WhereLess, WhereLessEqual,
)
from .fill import Fill
from .mask import MaskSlice
from .reshape import Reshape
from .learned_descriptor import LearnedDescriptor
from .triton_kernels import set_kv_buffer_launcher, set_kv_buffer_fp8_e4m3_launcher, set_kv_buffer_fp8_e5m2_launcher
from . import compiler


__all__ = [
    "set_kv_buffer_launcher",
    "set_kv_buffer_fp8_e4m3_launcher",
    "set_kv_buffer_fp8_e5m2_launcher",
    "Mean", "Max", "Min", "L2Norm",
    "MeanInterleave", "MaxInterleave", "MinInterleave", "L2NormInterleave",
    "GeMM",
    "Relu", "Silu", "Sigmoid", "Abs", "Add_Mul", "Log", "Exp",
    "Maximum", "Minimum", "Multiply", "Add",
    "WhereEqual", "WhereNotEqual", "WhereGreater",
    "WhereGreaterEqual", "WhereLess", "WhereLessEqual",
    "Fill",
    "MaskSlice",
    "Reshape",
    "LearnedDescriptor",
    "Context", "ctx", "get_ctx",
    "compiler",
]
