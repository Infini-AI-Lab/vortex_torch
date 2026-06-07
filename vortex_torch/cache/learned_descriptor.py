r"""``LearnedDescriptor`` — cache-side learned factorized block compressor.

Cache mirror of the indexer-side ``GeMM``-with-``Parameter`` machinery. Holds
two per-layer :class:`~vortex_torch.abs.Parameter` operands and, for each block
written this step, compresses the block's ``block_size`` latent rows into a
small per-head descriptor tensor ``g[H*m, r]`` stored in ``cache["descriptors"]``.

Math (factorized, per layer ``ℓ``, per block ``b`` with latents
``L_b ∈ R^{bs×d}``):

.. math::

    g[h, m, r] = (W_t[\ell, h]^\top \cdot L_b) \cdot W_k[\ell, h],
    \qquad g \in R^{H \times m \times r}

stored as ``[H*m, r]``. ``W_t = 1/bs`` (uniform), ``m=1``, ``W_k=I`` reduces to
the centroid scorer (a single mean-pooled descriptor per block).

Mechanism: this is a ``Schedule.S`` op — the big per-layer weights are baked
on the op instance (reached at runtime via ``ctx.op_list[<op_id>]``), moved to
device + bf16 once at compile time (:meth:`profile`, pre cuda-graph capture),
and the active layer's slice is gathered with an explicit host-int ``cur_layer``
(no ``.item()`` / device sync) — exactly like the indexer ``GeMM`` Parameter
path. The launcher runs a vectorized ``torch.einsum`` over the blocks whose
end-token appears in ``loc`` and scatters the result by block id — cuda-graph
safe (no ``.to(device)`` / host syncs in the hot path).
"""
from __future__ import annotations

import torch
from typing import Optional

from ..abs import vOp, vTensor, FORMAT
from ..abs.parameter import Parameter
from .context import Context
from ..utils import Schedule


class LearnedDescriptor(vOp):
    r"""Per-layer learned factorized block descriptor (cache side, Schedule.S).

    :__init__: ``LearnedDescriptor(Wt, Wk, H, m, r)`` —
        * ``Wt`` : :class:`Parameter` ``[Lfull, H, bs, m]`` token-mixing weights.
        * ``Wk`` : :class:`Parameter` ``[Lfull, H, d, r]`` channel-compression.
        * ``H`` / ``m`` / ``r`` : head count / landmarks / proj rank
          (descriptor output is ``[H*m, r]``).
    :__call__: ``op(latent, descriptors, loc=loc, ctx=ctx)`` — ``latent`` is the
        PAGED fused latent ``[B, bs, d]``; ``descriptors`` the PAGED output
        ``[B, H*m, r]``. Runs once per cache refresh in ``forward_cache``.
    """

    def __init__(self, Wt: Parameter, Wk: Parameter, H: int, m: int, r: int):
        super().__init__()
        assert isinstance(Wt, Parameter) and isinstance(Wk, Parameter), (
            "LearnedDescriptor: Wt and Wk must be Vortex.Parameter operands"
        )
        self.Wt = Wt
        self.Wk = Wk
        self.H = int(H)
        self.m = int(m)
        self.r = int(r)
        self.output_buffer: Optional[vTensor] = None
        self.output_format: Optional[FORMAT] = None
        # Standalone torch launcher outside the fused per-block kernel.
        self.schedule = Schedule.S
        self._block_size: Optional[int] = None
        self._latent_dim: Optional[int] = None

    # ---------------- profile ----------------
    def profile(
        self, latent: vTensor, descriptors: vTensor, loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""Trace-time: validate the PAGED ``latent`` ``[B, bs, d]`` /
        ``descriptors`` ``[B, H*m, r]``, bake + materialize the per-layer
        weights (pre cuda-graph capture), register the op, return ``descriptors``."""
        prefix = self._prefix()
        assert isinstance(latent, vTensor), f"{prefix}latent must be vTensor, got {type(latent)}"
        assert isinstance(descriptors, vTensor), (
            f"{prefix}descriptors must be vTensor, got {type(descriptors)}"
        )
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert latent.dim() == 3 and descriptors.dim() == 3, (
            f"{prefix}expected 3D latent/descriptors; got {latent.dim()}/{descriptors.dim()}"
        )
        bs, d = int(latent.shape[1]), int(latent.shape[2])
        assert descriptors.shape[1] == self.H * self.m, (
            f"{prefix}descriptors.shape[1] must be H*m={self.H * self.m}, "
            f"got {descriptors.shape[1]}"
        )
        assert descriptors.shape[2] == self.r, (
            f"{prefix}descriptors.shape[2] must be r={self.r}, got {descriptors.shape[2]}"
        )
        # Weight geometry checks against the block / latent dims (full Parameter
        # value is [L,H,bs,m] / [L,H,d,r]; .shape is the collapsed (0,N,K) view).
        assert self.Wt.value.shape[-2] == bs, (
            f"{prefix}Wt block-size axis {self.Wt.value.shape[-2]} != block_size {bs}"
        )
        assert self.Wt.value.shape[-1] == self.m, (
            f"{prefix}Wt landmark axis {self.Wt.value.shape[-1]} != m {self.m}"
        )
        assert self.Wk.value.shape[-2] == d, (
            f"{prefix}Wk channel axis {self.Wk.value.shape[-2]} != latent_dim {d}"
        )
        assert self.Wk.value.shape[-1] == self.r, (
            f"{prefix}Wk rank axis {self.Wk.value.shape[-1]} != r {self.r}"
        )

        self._block_size = bs
        self._latent_dim = d

        # Bake the weights onto device + bf16 ONCE (compile time, pre cuda-graph
        # capture) and snapshot the host layer lookup — runtime gather is sync-free.
        self.Wt.materialize(device=latent.device, dtype=torch.bfloat16)
        self.Wk.materialize(device=latent.device, dtype=torch.bfloat16)

        # descriptors is the caller-provided PAGED output buffer.
        assert descriptors._format in (FORMAT.PAGED, FORMAT.RAGGED), (
            f"{prefix}descriptors._format must be PAGED or RAGGED, got {descriptors._format}"
        )
        self.output_format = descriptors._format
        self.output_buffer = descriptors

        # Register: input is latent only (weights are baked on the op, not graph
        # inputs); output is the descriptors buffer.
        ctx.output_tensor_to_op_list[descriptors.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([latent.tensor_id])
        ctx.op_to_output_tensor_list.append([descriptors.tensor_id])
        return descriptors

    # ---------------- runtime (Schedule.S launcher) ----------------
    @torch.no_grad()
    def compute_g(
        self,
        latent: torch.Tensor,
        descriptors: torch.Tensor,
        loc: torch.Tensor,
        block_size: int,
        page_size: int,
        num_blocks_per_page: int,
        cur_layer: int,
    ) -> None:
        r"""Compute ``g[H*m, r]`` per just-written block and scatter into
        ``descriptors`` (block-major). cuda-graph-safe: weights are gathered by a
        host-int ``cur_layer`` (no ``.item()`` / ``.to(device)``); ``loc`` masking
        and the einsum stay on-device.

        ``latent`` is the flat fused-latent buffer; block ``block_id`` occupies a
        contiguous ``block_size * latent_dim`` run, so it reshapes to
        ``[n_blocks, block_size, latent_dim]``.
        """
        bs = block_size
        d = self._latent_dim
        Wt = self.Wt.gather(cur_layer)         # [H, bs, m] bf16, device
        Wk = self.Wk.gather(cur_layer)         # [H, d, r]  bf16, device

        tp = loc.to(torch.int64)                                       # [N]
        block_id = (tp // page_size) * num_blocks_per_page \
            + (tp % page_size) // bs                                   # [N]
        block_id = block_id.clamp(0, descriptors.shape[0] - 1)
        lat_blocks = latent.reshape(-1, bs, d)                         # [nb, bs, d]

        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            # Eager path (prefill / aux rebuild): block-end tokens only. Boolean
            # gather is fine here (not under cuda-graph capture). OVERWRITE
            # (index_copy_) matches CMean semantics — safe even on reused pages.
            is_end = ((tp + 1) % bs) == 0
            sel = is_end.nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                return
            bid = block_id.index_select(0, sel)
            Lb = lat_blocks.index_select(0, bid).to(torch.bfloat16)    # [Nb, bs, d]
            tok = torch.einsum("nsd,hsm->hnmd", Lb, Wt)
            g = torch.einsum("hnmd,hdr->hnmr", tok, Wk)
            g = g.permute(1, 0, 2, 3).reshape(g.shape[1], self.H * self.m, self.r)
            descriptors.index_copy_(0, bid, g.to(descriptors.dtype))
            return

        # CUDA-GRAPH-SAFE path (decode): every loc slot is processed at a fixed
        # shape (loc.shape[0] is fixed at capture) — no ``nonzero`` / boolean
        # gather. Decode writes one new token per request, so each block-end
        # slot has a UNIQUE target block id (no within-block duplicates); the
        # block-end mask redirects non-end slots' value to 0 and ``index_copy_``
        # of an overwrite is race-free because targets are unique. To preserve
        # OVERWRITE semantics (CMean parity, reused-page safety) we blend the
        # finalized rows: written rows get g, all other touched rows keep their
        # current stored value (read-modify-write at unique indices).
        is_end = (((tp + 1) % bs) == 0).to(torch.bfloat16)            # [N] 1/0
        Lb = lat_blocks.index_select(0, block_id).to(torch.bfloat16)  # [N, bs, d]
        tok = torch.einsum("nsd,hsm->hnmd", Lb, Wt)     # [H, N, m, d]
        g = torch.einsum("hnmd,hdr->hnmr", tok, Wk)     # [H, N, m, r]
        g = g.permute(1, 0, 2, 3).reshape(g.shape[1], self.H * self.m, self.r)  # [N, H*m, r]
        cur = descriptors.index_select(0, block_id).to(torch.bfloat16)         # [N, H*m, r]
        blended = is_end[:, None, None] * g + (1.0 - is_end[:, None, None]) * cur
        descriptors.index_copy_(0, block_id, blended.to(descriptors.dtype))
