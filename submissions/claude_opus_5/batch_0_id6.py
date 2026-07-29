"""block_sparse_attention + approxTopK(tolerate_ratio=0.65) — trtllm indexer.

Variant 7/10 of a tolerate_ratio sweep on AIME24. Identical to the built-in
``block_sparse_attention`` flow (one key-centroid per page; pages scored by the
dot product of the mean query against each centroid) except that the terminal
op is ``approxTopK(tolerate_ratio=0.65)`` instead of exact ``topK()``.

tolerate_ratio=0.65

Runs on the trtllm indexer backend, which uses the bf16 two-pass radix leaf at
``custom_ops/topk_output/trtllm/approx/``: two 8-bit rounds cover the full
16-bit bf16 key, and the single-pass gate fires only once at least
ceil((1 - tol) * k) blocks are already known to be true top-k members — so
recall >= 1 - tol is guaranteed.
"""
import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import approxTopK, GeMM, Mean
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_5_batch_0_id6_cls")
class ClaudeOpus5Batch0Id6Cls(vFlow):
    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemm = GeMM()
        self.mean = Mean(dim=1)
        self.output_func = approxTopK(tolerate_ratio=0.65)

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }
