r"""
Indexer-side operator API.

This module collects the core ops used on the *indexer path* of vFlow
pipelines. These operators are typically applied to query tensors or
intermediate scoring tensors to construct sparse routing decisions
(e.g., top-k page selection, attention scoring, pagewise normalization).

Included components:

- Matrix operations:
  :class:`GeMM`, :class:`GeMV`
  for page-tiled GEMM/GEMV used in similarity scoring.

- Output routing:
  :class:`topK`, :class:`approxTopK`
  for selecting sparse page indices based on per-token scores.
  :class:`approxTopK` is a faster, approximate variant with a tunable
  ``tolerate_ratio`` quality / cost knob; useful when topK selection
  cost dominates the indexer path.

- Reductions:
  :class:`Max`, :class:`Mean`, :class:`Min`, :class:`L2Norm`, :class:`Sum`
  for aggregating scores along query or key dimensions.

- Scans / normalization:
  :class:`Softmax`, :class:`Normalize`
  for in-place probability and magnitude normalization.

- Data layout transforms:
  :class:`Transpose`
  for switching between [B, N, D] and [B, D, N] style views.

- Binary/unary elementwise ops:
  :class:`Maximum`, :class:`Minimum`, :class:`Multiply`, :class:`Add`,
  :class:`WhereEqual`, :class:`WhereNotEqual`, :class:`WhereGreater`,
  :class:`WhereGreaterEqual`, :class:`WhereLess`, :class:`WhereLessEqual`,
  :class:`Relu`, :class:`Sigmoid`, :class:`Silu`, :class:`Add_Mul`,
  :class:`Abs`, :class:`Log`, :class:`Exp`.

- Utilities:
  :mod:`utils_sglang` for SGLang-related helpers.

- Runtime context:
  :class:`Context`, :func:`get_ctx`
  for accessing per-step dynamic state (page offsets, head count,
  max token budget, etc.).

These operators constitute the standard toolkit for building sparse
attention indexers in vFlow-compatible systems.
"""


from .matmul import GeMM, GeMV
from .factor_score import FactorScore
from .output_func import topK, approxTopK, Union
from .select import TopK
from .reduce import Max, Mean, Min, L2Norm, Sum
from .scan import Softmax, Normalize, Conv1d
from .transpose import Transpose
from .elementwise_binary import (
    Maximum, Minimum, Multiply, Add,
    WhereEqual, WhereNotEqual, WhereGreater,
    WhereGreaterEqual, WhereLess, WhereLessEqual,
)
from .elementwise import Relu, Sigmoid, Silu, Add_Mul, Abs, Log, Exp
from .save_load import Save, Load
from .mask import MaskSlice
from .kron import Kron
from .reshape import Reshape
from ..abs import Parameter   # re-exported for convenience; defined in vortex_torch.abs
from . import utils_sglang, compiler
from .context import Context, get_ctx
from .metadata import MetaData
__all__ = [
    "GeMM", "GeMV",
    "FactorScore",
    "topK", "approxTopK", "TopK", "Union",
    "Max", "Mean", "Min", "L2Norm", "Sum",
    "Softmax", "Normalize", "Conv1d",
    "Transpose",
    "Maximum", "Minimum", "Multiply", "Add",
    "WhereEqual", "WhereNotEqual", "WhereGreater",
    "WhereGreaterEqual", "WhereLess", "WhereLessEqual",
    "Relu", "Sigmoid", "Silu", "Add_Mul", "Abs", "Log", "Exp",
    "Save", "Load",
    "MaskSlice",
    "Kron",
    "Reshape",
    "Parameter",
    "utils_sglang",
    "Context",
    "MetaData",
    "get_ctx",
    "compiler",
]
