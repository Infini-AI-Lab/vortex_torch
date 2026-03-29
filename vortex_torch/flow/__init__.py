r"""
High-level vFlow API.

This module provides the public entry points for working with flow-style
sparse attention:

- :class:`vFlow`:
  Abstract base class for all flow implementations (block-sparse,
  GQA-based flows, QUEST-style flows, etc.).

- :func:`register`:
  Decorator used to register a vFlow implementation under a string key,
  e.g.::

      @register("block_sparse_attention")
      class BlockSparseAttention(vFlow):
          ...

- :func:`build_vflow`:
  Factory helper that instantiates a registered vFlow by name (and
  optional configuration), e.g.::

      flow = build_vflow("block_sparse_attention", **cfg)

- :mod:`algorithms`:
  Collection of concrete vFlow implementations and related building
  blocks.

These symbols form the high-level interface for constructing and using
vFlows in downstream models and runtime systems.
"""

from .flow import vFlow
from .registry import register
from .loader import build_vflow
from . import algorithms
from . import external_algorithms
__all__ = [
    "vFlow",
    "register",
    "build_vflow",
    "algorithms",
    "external_algorithms",
]