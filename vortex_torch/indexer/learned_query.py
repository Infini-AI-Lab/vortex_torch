r"""Learned bilinear query transform with a baked, per-layer constant weight.

``LearnedQuery`` is the query-side op of the trained per-layer block
compressor (``vortex_torch/compressor``). It transforms the absorbed MLA
query ``q[B, H, d]`` into a single per-request descriptor ``V[B, 1, d]``
such that a plain centroid dot ``⟨V, centroid_b⟩`` reproduces the trained
bilinear block score summed over heads:

.. math::

    \operatorname{score}(b)
      = \sum_h \big(W_q[\ell,h]^\top q_h\big)^\top \big(W_k[\ell,h]^\top c_b\big)
      = \Big\langle \underbrace{\sum_h W_k[\ell,h]\,(W_q[\ell,h]^\top q_h)}_{V}\;,\; c_b \Big\rangle ,

using the identity :math:`u^\top (W_k^\top c) = (W_k u)^\top c`. So the
cache side is unchanged (centroid via ``CMean``); only the query side gets
the learned transform.

**Per-layer constant baking.** The same compiled indexer runs for every
decode layer, but the trained weights differ per layer. The op therefore
holds *all* layers' weights as plain Python tensor attributes
(``Wq``/``Wk`` of shape ``[L, H, d, r]``) — NOT graph ``vTensor`` inputs —
plus a ``layer_lookup`` mapping ``global layer_id -> row``. The generated
Schedule.S launcher reaches these constants back through
``ctx.op_list[<op_id>]`` (the same producer-less-constant trick
``Conv1d`` uses for its ``weight``), reads the active layer from
the explicit ``cur_layer`` arg, gathers ``Wq[row]``/``Wk[row]`` and computes ``V``.

Set ``Wq = Wk = I`` (identity, truncated to rank ``r``) for the graceful
default: ``V = Σ_h q_h`` (i.e. the head-summed query), so
``⟨V, c_b⟩ == Σ_h ⟨q_h, c_b⟩``, the plain (head-summed) centroid scorer.
"""
from __future__ import annotations

from typing import Optional

import torch

from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class LearnedQuery(vOp):
    r"""Per-request learned bilinear query transform :math:`q[B,H,d]\to V[B,1,d]`.

    :__init__:
        ``LearnedQuery(Wq, Wk, layer_lookup, scaling=1.0)`` —

        * ``Wq`` / ``Wk`` : ``torch.Tensor`` of shape ``[L, H, d, r]`` (the
          trained per-layer projections; held as a Python attr, moved to the
          query's device/dtype at first use). ``L`` = number of trained
          layers (rows), ``H`` = query heads, ``d`` = latent dim, ``r`` =
          proj rank.
        * ``layer_lookup`` : 1-D int ``torch.Tensor`` of length
          ``max(global_layer_id)+1`` mapping a global ``cur_layer`` to a
          row in ``Wq``/``Wk``. Unknown layers map to ``-1`` (identity
          fallback: ``V = Σ_h q_h``).
        * ``scaling`` : optional scalar folded into ``V`` (irrelevant to
          top-k ranking, kept for parity with the trained scorer).
    :__call__:
        ``V = op(q, ctx=ctx)`` — ``q`` is ``[B, H, d]`` (``BATCHED``);
        returns ``V`` ``[B, 1, d]`` (``BATCHED``), consumable by
        ``GeMM(V, centroids)``.
    """

    def __init__(
        self,
        Wq: Optional[torch.Tensor] = None,
        Wk: Optional[torch.Tensor] = None,
        layer_lookup: Optional[torch.Tensor] = None,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        prefix = self._prefix()
        self.scaling = float(scaling)
        # ``Wq is None`` => lazy identity: build a truncated-identity weight at
        # profile time from the real query geometry (V = Σ_h q_h). This is the
        # graceful default when no trained checkpoint is supplied; the flow
        # still compiles and reproduces the head-summed centroid scorer.
        if Wq is None:
            self.Wq = None
            self.Wk = None
            self.layer_lookup = (
                None if layer_lookup is None else layer_lookup.to(torch.long)
            )
            self.num_layers = self.num_heads = None
            self.latent_dim = self.proj_dim = None
        else:
            assert isinstance(Wq, torch.Tensor) and isinstance(Wk, torch.Tensor), (
                f"{prefix}Wq/Wk must be torch.Tensor, got {type(Wq)}, {type(Wk)}"
            )
            assert Wq.dim() == 4 and Wk.dim() == 4, (
                f"{prefix}Wq/Wk must be 4D [L, H, d, r]; got "
                f"{tuple(Wq.shape)}, {tuple(Wk.shape)}"
            )
            assert Wq.shape == Wk.shape, (
                f"{prefix}Wq/Wk shape mismatch: {tuple(Wq.shape)} vs {tuple(Wk.shape)}"
            )
            assert isinstance(layer_lookup, torch.Tensor) and layer_lookup.dim() == 1, (
                f"{prefix}layer_lookup must be a 1-D int tensor, got "
                f"{type(layer_lookup)}"
            )
            # Held as plain Python attrs (constants), NOT graph vTensors. The
            # generated launcher reads them back via ``ctx.op_list[<id>]``.
            self.Wq = Wq
            self.Wk = Wk
            self.layer_lookup = layer_lookup.to(torch.long)
            self.num_layers, self.num_heads, self.latent_dim, self.proj_dim = Wq.shape

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Set in profile(): bf16 weights live on the compile device and a plain
        # CPU python list for host-side layer lookup (so compute_V does NO
        # host↔device copy or .item() sync — required for cuda-graph capture).
        self._lookup_cpu: Optional[list] = None
        # Standalone per-request transform — runs once per decode step before
        # the block-tiled GeMM, so it is a standalone (Schedule.S) op, not
        # fused into the per-workload kernel.
        self.schedule = Schedule.S

    # ---------------- profile ----------------
    def profile(self, q: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``q`` ``[B, H, d]`` (``BATCHED``), register
        the op, and return a ``vTensor`` view of the ``[B, 1, d]`` output."""
        prefix = self._prefix()

        assert isinstance(q, vTensor), (
            f"{prefix}profile expects q to be vTensor, got {type(q)}"
        )
        assert q.dim() == 3, (
            f"{prefix}expected 3D query [B, H, d]; got ndim={q.dim()} "
            f"shape={tuple(q.shape)}"
        )
        assert q._format == FORMAT.BATCHED, (
            f"{prefix}query must be BATCHED, got {q._format}"
        )

        H, d = int(q.shape[1]), int(q.shape[2])
        if self.Wq is None:
            # Lazy identity: build [1, H, d, d] truncated-identity weights from
            # the real query geometry, so V = Σ_h q_h (centroid scorer).
            eye = torch.zeros(d, d)
            eye[: min(d, d), : min(d, d)] = torch.eye(d)
            self.Wq = eye.expand(1, H, d, d).contiguous()
            self.Wk = self.Wq
            self.num_layers, self.num_heads = 1, H
            self.latent_dim, self.proj_dim = d, d
            if self.layer_lookup is None:
                self.layer_lookup = torch.zeros(1, dtype=torch.long)
        else:
            # H and d must agree with the baked weights (the compiled function
            # is specialised to one model geometry).
            assert H == self.num_heads, (
                f"{prefix}query head count {H} != weight H {self.num_heads}"
            )
            assert d == self.latent_dim, (
                f"{prefix}query latent dim {d} != weight d {self.latent_dim}"
            )

        # Move the baked constants onto the compile device ONCE here (this runs
        # at compile time, BEFORE any cuda-graph capture) and store them in
        # bf16 (weight + compute are bf16, matching the query), plus snapshot the
        # layer lookup as a host python list. compute_V then does no .to(device)
        # / .item() — both illegal during graph capture.
        self.Wq = self.Wq.to(device=q.device, dtype=torch.bfloat16).contiguous()
        self.Wk = self.Wk.to(device=q.device, dtype=torch.bfloat16).contiguous()
        self._lookup_cpu = self.layer_lookup.to("cpu", torch.long).tolist()

        self.output_format = FORMAT.BATCHED
        # Pure-metadata vTensor; the compiler allocates the real BATCHED
        # buffer (leading dim ``max_bs * num_kv_heads``).
        self.output_buffer = vTensor(
            shape=(0, 1, self.latent_dim),
            dtype=ctx.vortex_dtype,
            device=q.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([q.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer

    # ---------------- runtime helper (called from generated launcher) ---- #
    def compute_V(self, q: torch.Tensor, cur_layer: int) -> torch.Tensor:
        r"""Compute ``V[B, 1, d]`` from ``q[B, H, d]`` for ``cur_layer``.

        Selects the layer's weight slice via ``layer_lookup`` (identity
        fallback ``V = Σ_h q_h`` when the layer was not trained), then

        .. math:: V = \text{scaling}\cdot \sum_h W_k[\ell,h]\,(W_q[\ell,h]^\top q_h).

        Computed in bf16 (weight + compute), matching the bf16 query.
        """
        # Weights are bf16 on the right device (moved in profile()); the lookup
        # is a host python list. No .to(device)/.item() here → capturable.
        assert q.dtype == torch.bfloat16, (
            f"{self._prefix()}compute_V expects a bf16 query, got {q.dtype}"
        )
        qf = q                                                     # [B, H, d] bf16

        lid = int(cur_layer)
        row = self._lookup_cpu[lid] if (self._lookup_cpu is not None
                                        and 0 <= lid < len(self._lookup_cpu)) else -1

        if row < 0:
            # Identity fallback: V = Σ_h q_h  (== plain head-summed centroid scorer)
            V = qf.sum(dim=1, keepdim=True)                        # [B, 1, d]
        else:
            Wq = self.Wq[row]                                      # [H, d, r] (bf16 view)
            Wk = self.Wk[row]                                      # [H, d, r] (bf16 view)
            # u[b,h,r] = Σ_d q[b,h,d] Wq[h,d,r]
            u = torch.einsum("bhd,hdr->bhr", qf, Wq)              # [B, H, r]
            # v[b,h,d] = Σ_r Wk[h,d,r] u[b,h,r]
            vh = torch.einsum("hdr,bhr->bhd", Wk, u)              # [B, H, d]
            V = vh.sum(dim=1, keepdim=True)                        # [B, 1, d]

        if self.scaling != 1.0:
            V = V * self.scaling
        return V.to(q.dtype)
