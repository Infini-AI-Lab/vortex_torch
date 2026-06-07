r"""``FactorScore`` — indexer-side learned factorized block scorer.

Indexer half of the learned factorized block compressor. Consumes the
per-block descriptors ``g[H*m, r]`` written by the cache-side
:class:`~vortex_torch.cache.LearnedDescriptor` and scores each block against
the per-request query with a per-layer learned ``Wq`` projection:

.. math::

    u[h, r] = W_q[\ell, h]^\top q_h, \qquad
    s[h, m] = \langle u_h,\, g[b, h, m]\rangle, \qquad
    \operatorname{score}(b) = \sum_h \max_m s[h, m].

``Wq = I`` and ``m=1`` reduces to the head-summed centroid scorer.

Mechanism: ``Schedule.S`` — the per-layer ``Wq`` is a baked
:class:`~vortex_torch.abs.Parameter` (reached at runtime via
``ctx.op_list[<op_id>]``), materialized to device + bf16 once at compile time
(:meth:`profile`, pre cuda-graph capture); the active layer's slice is gathered
with an explicit host-int ``cur_layer`` (no ``.item()`` / device sync). The
launcher gathers descriptors per request via the trtllm block tables and emits a
RAGGED ``[S, 1, 1]`` score in the exact ``bx * max_blocks_per_seq + col`` layout
the ``topK`` kernel consumes — cuda-graph safe.
"""
from __future__ import annotations

import torch
from typing import Optional

from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..abs.parameter import Parameter
from ..utils import Schedule


class FactorScore(vOp):
    r"""Per-layer learned factorized block scorer (indexer side, Schedule.S).

    :__init__: ``FactorScore(Wq, H, m, r)`` —
        * ``Wq`` : :class:`Parameter` ``[Lfull, H, d, r]`` query projection.
        * ``H`` / ``m`` / ``r`` : head count / landmarks / proj rank.
    :__call__: ``score = op(q, descriptors, ctx=ctx)`` — ``q`` BATCHED
        ``[B, H, d]``; ``descriptors`` PAGED ``[S, H*m, r]``; returns RAGGED
        ``score`` ``[S, 1, 1]`` for ``topK``.
    """

    def __init__(self, Wq: Parameter, H: int, m: int, r: int):
        super().__init__()
        assert isinstance(Wq, Parameter), (
            "FactorScore: Wq must be a Vortex.Parameter operand"
        )
        self.Wq = Wq
        self.H = int(H)
        self.m = int(m)
        self.r = int(r)
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        self.schedule = Schedule.S
        self._d: Optional[int] = None

    # ---------------- profile ----------------
    def profile(self, q: vTensor, descriptors: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate BATCHED ``q`` ``[B, H, d]`` / PAGED
        ``descriptors`` ``[S, H*m, r]``, bake + materialize ``Wq`` (pre
        cuda-graph capture), register the op, return the RAGGED ``[S, 1, 1]``
        score buffer."""
        prefix = self._prefix()
        assert isinstance(q, vTensor), f"{prefix}q must be vTensor, got {type(q)}"
        assert isinstance(descriptors, vTensor), (
            f"{prefix}descriptors must be vTensor, got {type(descriptors)}"
        )
        assert q.dim() == 3 and descriptors.dim() == 3, (
            f"{prefix}expected 3D q/descriptors; got {q.dim()}/{descriptors.dim()}"
        )
        assert q._format == FORMAT.BATCHED, (
            f"{prefix}q must be BATCHED (per-request), got {q._format}"
        )
        assert q.shape[1] == self.H, (
            f"{prefix}q.shape[1] must be H={self.H}, got {q.shape[1]}"
        )
        d = int(q.shape[2])
        assert descriptors.shape[1] == self.H * self.m, (
            f"{prefix}descriptors.shape[1] must be H*m={self.H * self.m}, "
            f"got {descriptors.shape[1]}"
        )
        assert descriptors.shape[2] == self.r, (
            f"{prefix}descriptors.shape[2] must be r={self.r}, got {descriptors.shape[2]}"
        )
        assert self.Wq.value.shape[-2] == d, (
            f"{prefix}Wq channel axis {self.Wq.value.shape[-2]} != q dim {d}"
        )
        assert self.Wq.value.shape[-1] == self.r, (
            f"{prefix}Wq rank axis {self.Wq.value.shape[-1]} != r {self.r}"
        )
        self._d = d

        # Bake Wq onto device + bf16 ONCE (compile time, pre cuda-graph capture).
        self.Wq.materialize(device=q.device, dtype=torch.bfloat16)

        # RAGGED [S, 1, 1] score buffer (one score per block).
        self.output_format = FORMAT.RAGGED
        self.output_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=ctx.vortex_dtype,
            device=q.device,
            _format=FORMAT.RAGGED,
            tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        # q + descriptors are graph inputs; Wq is baked on the op.
        ctx.op_to_input_tensor_list.append([q.tensor_id, descriptors.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
        return self.output_buffer

    # ---------------- runtime (Schedule.S launcher) ----------------
    @torch.no_grad()
    def compute_score(
        self,
        q: torch.Tensor,
        descriptors: torch.Tensor,
        score_out: torch.Tensor,
        dense_seqlens: torch.Tensor,
        dense_block_tables: torch.Tensor,
        batch_size: int,
        block_size: int,
        block_reserved_bos: int,
        block_reserved_eos: int,
        cur_layer: int,
    ) -> None:
        r"""Fill ``score_out[bx * row_stride + col]`` for every (request, block)
        with ``Σ_h max_m ⟨W_q[h]^T q_h, g[h, m]⟩``. cuda-graph-safe: ``Wq`` is
        gathered by a host-int ``cur_layer``; ``batch_size`` / ``block_size`` are
        host ints baked at plan time; all indexing/einsum stay on-device.

        Layout matches the ``topK`` kernel exactly (``TopKOutput_Kernel``):
        ``row_stride = dense_block_tables.shape[1]``; per request ``bx`` the
        live blocks are columns ``[0, ceil(tokens/block_size))`` and
        ``dense_block_tables[bx, col]`` is the global block id into the PAGED
        descriptor buffer.
        """
        H, m, r = self.H, self.m, self.r
        bs = int(batch_size)
        if bs <= 0:
            return
        Wq = self.Wq.gather(cur_layer)              # [H, d, r] bf16, device
        row_stride = dense_block_tables.shape[1]

        # CUDA-GRAPH-SAFE: score EVERY (request, col) slot in the fixed
        # [bs, row_stride] grid (both sizes fixed at capture) — no ``nonzero`` /
        # boolean-mask gather (which would produce data-dependent shapes that
        # cuda-graph capture cannot replay). Invalid cols (col >= block_len) get
        # a computed score that ``topK`` never reads (it bounds each row by
        # dense_seqlens); block ids are clamped so the gather stays in-bounds.
        q_live = q[:bs].to(torch.bfloat16)              # [bs, H, d]
        u = torch.einsum("bhd,hdr->bhr", q_live, Wq)    # [bs, H, r]

        bids = dense_block_tables[:bs].to(torch.int64)  # [bs, R]
        bids = bids.clamp(0, descriptors.shape[0] - 1)
        g = descriptors.index_select(0, bids.reshape(-1)).to(torch.bfloat16)  # [bs*R, H*m, r]
        g = g.reshape(bs, row_stride, H, m, r)          # [bs, R, H, m, r]

        # s[bx, c, h, m] = <u[bx, h], g[bx, c, h, m]>
        s = torch.einsum("bhr,bchmr->bchm", u, g)       # [bs, R, H, m]
        scores = s.amax(dim=-1).sum(dim=2)              # [bs, R]

        # Write the full [bs, row_stride] region in the topK layout
        # score_out[bx * row_stride + col]; fixed-shape copy, graph-safe.
        out_flat = score_out.view(-1)
        out_flat[: bs * row_stride].copy_(scores.reshape(-1).to(out_flat.dtype))
