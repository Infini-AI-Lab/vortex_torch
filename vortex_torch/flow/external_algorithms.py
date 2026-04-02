"""
External sparse attention algorithm registrations for NSA, FSA, and FlashMoBA.

These vFlow subclasses use simple centroid-based routing for the DECODE path
(forward_indexer + forward_cache), identical to BlockSparseAttention.

The EXTEND path (forward_extend) is handled directly in vtx_graph_backend.py
using each algorithm's own sparse attention kernel — these vFlow classes are
not involved in extend.
"""

import torch
from typing import Dict, Tuple

from .flow import vFlow
from ..indexer import topK, GeMV
from ..cache import Mean as CMean
from ..abs import ContextBase
from .registry import register


class _ExternalAlgoBase(vFlow):
    """
    Base vFlow for external sparse attention algorithms (NSA, FSA, FlashMoBA).

    Decode routing: centroid-based (same as BlockSparseAttention).
    Extend: bypassed — vtx_graph_backend dispatches to algorithm-specific kernels.
    """

    def __init__(self):
        super().__init__()
        self.gemv = GeMV()
        self.output_func = topK()
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = q.mean(dim=1, keepdim=True)
        score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int) -> Dict[str, Tuple[int, int]]:
        return {
            "centroids": (1, head_dim),
        }


@register("nsa")
class NSASparseAttention(_ExternalAlgoBase):
    """Naive Sparse Attention — decode uses centroid routing, extend uses NSA kernels."""
    pass


@register("fsa")
class FSASparseAttention(_ExternalAlgoBase):
    """Flash Sparse Attention — decode uses centroid routing, extend uses FSA kernels."""
    pass


@register("flash_moba")
class FlashMoBASparseAttention(_ExternalAlgoBase):
    """FlashMoBA — decode uses centroid routing, extend uses FlashMoBA kernels."""
    pass
