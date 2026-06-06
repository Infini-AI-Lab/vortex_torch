import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    topK, approxTopK, GeMV, Softmax, Max, Sum, GeMM,
    Maximum, Multiply, Add, L2Norm, Save, Load, Mean, MaskSlice, Kron,
)
from vortex_torch.cache import (
    Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm,
    Fill as CFill, MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave,
    MeanInterleave as CMeanInterleave,
)
from vortex_torch.abs import ContextBase


@register("lserve_centroid_sparse_attention_sub")
class LServeCentroidSparseAttention(vFlow):
    r"""Centroid routing at sub-block granularity (see flow/algorithms.py)."""
    SUB_BLOCK_SIZE = 16

    def __init__(self):
        super().__init__()
        self.mean = Mean(dim=1)
        self.gemm = GeMM()
        self.max_sub = Max(dim=1)
        self.output_func = topK()
        self.reduction = CMeanInterleave(dim=1, k=self.SUB_BLOCK_SIZE)

    def forward_indexer(self, q, o, cache, ctx):
        q_summary = self.mean(q, ctx=ctx)
        score = self.gemm(q_summary, cache["centroids"], ctx=ctx)
        page_score = self.max_sub(score, ctx=ctx)
        self.output_func(page_score, o, ctx=ctx)

    def forward_cache(self, cache, loc, ctx):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (block_size // self.SUB_BLOCK_SIZE, head_dim),
        }
