import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class GeMV(vOp):
    r"""
    Per-request batched matrix–vector product, :math:`O = Y X^{\top}`.

    :Math:
        Batched query :math:`X\in\mathbb{R}^{B\times 1\times D}`, packed pages
        :math:`Y\in\mathbb{R}^{S\times 1\times D}`; for page :math:`s` in
        request :math:`i(s)`,

        .. math::

            O_{s,0,0} = \sum_{d=0}^{D-1} Y_{s,0,d}\,X_{i(s),0,d}
                      = \langle Y_s,\, X_{i(s)} \rangle,
            \qquad O\in\mathbb{R}^{S\times 1\times 1}.
    :__init__: ``GeMV()`` — no arguments.
    :__call__: ``o = op(x, y, ctx=ctx)`` — ``x`` is ``[B, 1, D]``, ``y`` is
        ``[S, 1, D]`` (matching ``D``); returns ``o`` ``[S, 1, 1]``. Output is
        ``BATCHED`` iff both inputs are, else ``RAGGED``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` ``[B, 1, D]`` / ``y`` ``[S, 1, D]``,
        register the op, and return a ``vTensor`` view of the ``[S, 1, 1]``
        output (see the class docstring)."""
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        assert x.shape[1] == 1, f"{prefix}expected x.shape[1] == 1, got {tuple(x.shape)}"
        assert y.shape[1] == 1, f"{prefix}expected y.shape[1] == 1, got {tuple(y.shape)}"
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Output is BATCHED iff both inputs are BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )
        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer



# ------------------------------ GeMM ------------------------------ #
class GeMM(vOp):
    r"""
    Per-page matrix–matrix product, :math:`O[s] = Y[s]\,X[s]^{\top}`.

    :Math:
        :math:`Y\in\mathbb{R}^{S\times N_y\times K}`,
        :math:`X\in\mathbb{R}^{(B\text{ or }S)\times N_x\times K}`; per page
        :math:`s` this is :math:`O_s = Y_s X_s^{\top}` (i.e. ``GeMM(x, y) = y xᵀ``):

        .. math::

            O_{s,a,b} = \sum_{k=0}^{K-1} Y_{s,a,k}\,X_{s,b,k},
            \qquad O\in\mathbb{R}^{S\times N_y\times N_x}.
    :__init__: ``GeMM()`` — no arguments.
    :__call__: ``o = op(x, y, ctx=ctx)`` — ``x`` is ``[B|S, N_x, K]``, ``y`` is
        ``[S, N_y, K]`` (matching ``K``); returns ``o`` ``[S, N_y, N_x]``.
        Output is ``BATCHED`` iff both inputs are, else ``RAGGED``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
        self._param = None        # set when the y operand is a Vortex.Parameter

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` ``[B|S, N_x, K]`` / ``y`` ``[S, N_y, K]``
        (matching ``K``), register the op, and return a ``vTensor`` view of the
        ``[S, N_y, N_x]`` output (see the class docstring).

        If ``y`` is a :class:`~vortex_torch.indexer.Parameter` (``FORMAT.PARAMETER``,
        a batch-shared learned constant) the op runs as a standalone
        ``Schedule.S`` ``torch.matmul`` over the per-layer slice instead of the
        fused per-workload kernel (see :meth:`_profile_param`)."""
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # Batch-shared learned constant operand → Schedule.S torch.matmul path.
        # Checked BEFORE the K-match: a Parameter contracts the FLATTENED
        # activation (H·d), not x's last dim, so the normal K rule doesn't apply.
        if y._format == FORMAT.PARAMETER:
            return self._profile_param(x, y, ctx)

        # K must match (normal fused path)
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Output is BATCHED iff both inputs are BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )

        # Output logical sizes: Ny x Nx
        Ny, Nx = y.shape[1], x.shape[1]

        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, Ny, Nx),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer

    # ------------------------------------------------------------------ #
    # Parameter (batch-shared constant) path: Schedule.S + torch.matmul
    # ------------------------------------------------------------------ #
    def _profile_param(self, x: vTensor, y, ctx: Context) -> vTensor:
        r"""``y`` is a :class:`Parameter` ``[L, N_y, K]`` (batch-shared constant).
        The contraction is the **standard GeMM** one over the last dim ``K``
        (``x.shape[2] == K``): ``O[b, a, nx] = Σ_k W[ℓ, a, k] x[b, nx, k]`` →
        ``[B, N_y, N_x]`` (BATCHED), computed by :meth:`compute_param` (gather row
        ``cur_layer``, torch.matmul). Same definition as fused ``GeMM`` — only the
        ``y`` operand is a baked constant and the launch is ``Schedule.S`` so the
        big weight never enters the tiled kernel. Reshape the activation around
        this op for any flatten/transpose (GeMM stays self-contained)."""
        prefix = self._prefix()
        assert x._format == FORMAT.BATCHED, (
            f"{prefix}a Parameter operand requires a BATCHED activation (per-request); "
            f"got x._format={x._format}"
        )
        assert int(x.shape[2]) == int(y.shape[2]), (
            f"{prefix}K mismatch: x.shape[2]={x.shape[2]} vs Parameter K={y.shape[2]} "
            f"(Reshape the activation so its last dim matches the Parameter's K)"
        )
        self.schedule = Schedule.S
        self._param = y
        # Bake the weight onto device + bf16 ONCE here (compile time, pre
        # cuda-graph-capture) and snapshot the host layer lookup.
        y.materialize(device=x.device, dtype=torch.bfloat16)

        Ny, Nx = int(y.shape[1]), int(x.shape[1])
        self.output_format = FORMAT.BATCHED
        self.output_buffer = vTensor(
            shape=(0, Ny, Nx), dtype=ctx.vortex_dtype, device=x.device,
            _format=FORMAT.BATCHED, tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        # The Parameter is NOT a graph input (it is baked on the op); only x is.
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
        return self.output_buffer

    @torch.no_grad()
    def compute_param(self, x: torch.Tensor, cur_layer: int) -> torch.Tensor:
        r"""Runtime (Schedule.S launcher): the standard GeMM contraction
        ``O[b,a,nx] = Σ_k W[a,k] x[b,nx,k]`` with ``W = self._param.gather(
        cur_layer)`` ``[N_y, K]`` (bf16, on device). ``x`` ``[B, N_x, K]`` (bf16) →
        ``[B, N_y, N_x]``. No ``.to(device)`` / ``.item()`` — cuda-graph-safe."""
        assert x.dtype == torch.bfloat16, (
            f"{self._prefix()}compute_param expects a bf16 activation, got {x.dtype}"
        )
        W = self._param.gather(cur_layer)                      # [Ny, K] bf16
        return torch.einsum("nk,bxk->bnx", W, x)             # [B, Ny, Nx]