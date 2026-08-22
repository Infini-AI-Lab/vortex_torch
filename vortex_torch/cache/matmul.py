import torch
from .context import Context
from ..abs import vOp, vTensor, FORMAT
from ..utils import Schedule
from typing import Optional


class GeMM(vOp):
    r"""
    Per-block matrix–matrix product, :math:`O_b = Y_b X_b^{\top}` (cache side).

    :Math:
        For :math:`X\in\mathbb{R}^{B\times N_x\times K}`,
        :math:`Y\in\mathbb{R}^{B\times N_y\times K}`:

        .. math::

            O_{b,a,c} = \sum_{k=0}^{K-1} Y_{b,a,k}\,X_{b,c,k},
            \qquad O\in\mathbb{R}^{B\times N_y\times N_x}.
    :__init__: ``GeMM()`` — no arguments.
    :__call__: ``o = op(x, y, output, loc=loc, ctx=ctx)`` — ``x``
        ``[B, N_x, K]``, ``y`` ``[B, N_y, K]`` (matching ``K``); ``o``
        ``[B, N_y, N_x]``. ``PAGED`` iff a ``PAGED`` ``output`` is supplied,
        else ``RAGGED``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache GeMM fuses into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # --------------------------------------------------------------------- #
    # profile: validate and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, y: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs and optionally allocate an internal output buffer.

        The logical shapes are:

        - ``x``: ``[B, N_x, K]``
        - ``y``: ``[B, N_y, K]``
        - ``output`` (if provided): ``[B_out, N_y, N_x]``

        with the constraint that the inner dimension :math:`K` matches:

        .. math::

            x.\text{shape}[2] = y.\text{shape}[2].

        Parameters
        ----------
        x : vTensor
            Right-hand operand in ``Y @ X^T``, with logical shape
            ``[B, N_x, K]``.

        y : vTensor
            Left-hand operand in ``Y @ X^T``, with logical shape
            ``[B, N_y, K]``.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            RAGGED buffer with shape ``[B, N_y, N_x]`` is allocated using
            ``ctx.max_new_tokens_per_batch * ctx.head_num`` for the
            leading dimension. If not ``None``, this tensor must have
            rank 3, last two dimensions ``(N_y, N_x)``, and format in
            ``{PAGED, RAGGED}``.

        loc : torch.Tensor
            Auxiliary tensor carrying metadata required by the GEMM
            implementation.

        ctx : Context
            Execution context that provides the runtime value of ``B``
            and is used for auxiliary memory accounting.

        Returns
        -------
        vTensor
            A :class:`vTensor` view representing the resolved output.

        Raises
        ------
        AssertionError
            If types, ranks, inner-dimension match, shapes, or devices
            are incompatible.
        """
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}y must be vTensor, got {type(y)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"

        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert y.dim() == 3, f"{prefix}y must be 3D, got ndim={y.dim()} shape={tuple(y.shape)}"

        # --- K match: x[..., K] == y[..., K] ---
        Kx, Ky = x.shape[2], y.shape[2]
        assert Kx == Ky, f"{prefix}K mismatch: x.shape[2]={Kx} vs y.shape[2]={Ky}"

        # Output logical shape: [B, Ny, Nx]
        Ny, Nx = y.shape[1], x.shape[1]

        # Case A: output not provided -> allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = FORMAT.RAGGED

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, Ny, Nx),
                dtype=ctx.vortex_dtype,
                device=x.device,
                _format=self.output_format,
                tensor_id=len(ctx.tensor_list),
            )
            ctx.tensor_list.append(self.output_buffer)
            ctx.output_tensor_to_op_list.append(len(ctx.op_list))
            ctx.op_list.append(self)
            ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
            ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
            return self.output_buffer

        # Case B: output provided -> output_format follows output._format.
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )
        assert output._format in (FORMAT.PAGED, FORMAT.RAGGED, FORMAT.SLOTTED), (
            f"{prefix}output._format must be PAGED, RAGGED or SLOTTED, got {output._format}"
        )
        self.output_format = output._format

        # Shape consistency: GEMM yields [*, Ny, Nx]
        assert output.shape[1] == Ny and output.shape[2] == Nx, (
            f"{prefix}output shape mismatch. Expected (*,{Ny},{Nx}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == y.device == output.device, (
            f"{prefix}x, y, and output must be on the same device "
            f"(x.device={x.device}, y.device={y.device}, output.device={output.device})"
        )

        # Register in the cache graph (caller-provided output path).
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output
