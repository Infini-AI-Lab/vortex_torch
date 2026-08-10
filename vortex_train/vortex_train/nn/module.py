"""``SparseAttention`` — the user-facing module. Selection policy in, training out.

Usage::

    attn = SparseAttention("block_topk", num_kv_heads=8)
    out = attn(q, k, v)          # trains through it; backward is exact

or with a policy class directly::

    attn = SparseAttention(MyPolicy, num_kv_heads=8)

The compile happens once here, in ``__init__``. Per step the path is: build state
(1 launch) -> score+select (1 launch) -> transpose (3 launches) -> attention
(1 launch), with a matching set in backward. No host work, no ``.item()``, and
no launch count that depends on sequence length.

Selection is a **hard mask**: no gradient flows into the scorer (that is the
straight-through choice from the design), so the pattern crosses the autograd
boundary as a constant and the backward stays an exact sparse FlashAttention over
the selected blocks.
"""
from __future__ import annotations

import torch
from torch import nn

from ..compiler.compile import CompiledSelection, compile_selection
from ..flow.spec import Selection, get_selection
from ..kernels.select import score_and_select
from ..kernels.state import build_state
from ..pattern import SparsePattern
from .functional import sparse_attention


class SparseAttention(nn.Module):
    """Block-sparse attention driven by a compiled selection policy.

    Parameters
    ----------
    selection:
        A registered policy name, a ``Selection`` subclass, or an instance.
    num_kv_heads:
        Needed to interpret ``q``'s head axis as GQA groups. Selection is shared
        within a group, which is what lets the attention kernel load a KV block
        once and use it for every query head in the group.
    softmax_scale:
        Defaults to ``1/sqrt(head_dim)``.
    """

    def __init__(
        self,
        selection: str | type[Selection] | Selection,
        *,
        num_kv_heads: int,
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__()
        if isinstance(selection, str):
            selection = get_selection(selection)
        self.compiled: CompiledSelection = compile_selection(selection)
        self.num_kv_heads = num_kv_heads
        self.softmax_scale = softmax_scale

    def build_pattern(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> SparsePattern:
        """Run state build + fused score/select. Exposed for tests and ablations."""
        c = self.compiled
        skv = k.shape[2]
        n_kv = (skv + c.block_kv - 1) // c.block_kv

        # No grad: selection is a hard mask (design §0), so this whole subgraph is
        # outside autograd and costs no saved activations.
        with torch.no_grad():
            state = build_state(k, v, c.fields, block_kv=c.block_kv)
            cnt, idx = score_and_select(
                q, state, c.tape,
                num_kv_blocks=n_kv,
                seqlen_kv=skv,
                block_q=c.block_q,
                block_kv=c.block_kv,
                num_kv_heads=self.num_kv_heads,
                topk=c.budget.topk,
                reserve_bos=c.budget.reserve_bos,
                reserve_local=c.budget.reserve_local,
                reserve_eos=c.budget.reserve_eos,
                causal=c.causal,
                q_how=c.q_how,
            )
        return SparsePattern(
            cnt=cnt, idx=idx,
            block_q=c.block_q, block_kv=c.block_kv,
            seqlen_q=q.shape[2], seqlen_kv=skv,
        )

    def forward(
        self,
        q: torch.Tensor,            # [B, Hq, Sq, D]
        k: torch.Tensor,            # [B, Hkv, Skv, D]
        v: torch.Tensor,            # [B, Hkv, Skv, D]
    ) -> torch.Tensor:
        if k.shape[1] != self.num_kv_heads:
            raise ValueError(
                f"k has {k.shape[1]} heads but the module was built with "
                f"num_kv_heads={self.num_kv_heads}"
            )
        if q.shape[1] % self.num_kv_heads:
            raise ValueError(
                f"Hq={q.shape[1]} is not a multiple of num_kv_heads={self.num_kv_heads}"
            )
        pattern = self.build_pattern(q, k, v)
        return sparse_attention(
            q, k, v, pattern,
            causal=self.compiled.causal,
            softmax_scale=self.softmax_scale,
        )

    def extra_repr(self) -> str:
        c = self.compiled
        b = c.budget
        return (
            f"{c.name}, topk={b.topk}, reserve(bos={b.reserve_bos}, "
            f"local={b.reserve_local}, eos={b.reserve_eos}), "
            f"block_q={c.block_q}, block_kv={c.block_kv}, causal={c.causal}"
        )
