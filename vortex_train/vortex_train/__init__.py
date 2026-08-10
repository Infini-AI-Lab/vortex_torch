"""vortex_train — training-side block-sparse attention with user-specified selection."""
from .flow import Budget, Field, Selection, register  # noqa: F401
from .nn import SparseAttention, sparse_attention  # noqa: F401
from .pattern import SparsePattern  # noqa: F401

__version__ = "0.1.0"
