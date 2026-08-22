import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ElementwiseBinaryOpType, Schedule
from typing import Optional


class Elementwise_Binary(vOp):
    r"""
    Pointwise binary op over two cache blocks (cache side).

    :Math:
        For :math:`X, Y \in \mathbb{R}^{B\times N\times D}` and a scalar
        function :math:`g` fixed by the subclass (max / min / affine / product
        / comparison mask):

        .. math::

            Z_{b,n,d} = g(X_{b,n,d},\, Y_{b,n,d};\, \alpha, \beta).

        ``N`` and ``D`` broadcast when one operand's extent is ``1``.
    :__init__: ``Elementwise_Binary(alpha=1.0, beta=1.0)`` — scalars consumed
        only by the ops that need them (e.g. :class:`Add`); abstract, pick a
        concrete subclass.
    :__call__: ``z = op(x, y, output, loc=loc, ctx=ctx)`` — ``x`` ``[B, N_x, D_x]``,
        ``y`` ``[B, N_y, D_y]`` → ``[B, max(N_x,N_y), max(D_x,D_y)]``. ``PAGED``
        iff a ``PAGED`` ``output`` is supplied, else ``RAGGED``.
    :Note: subclasses — :class:`Maximum`, :class:`Minimum`, :class:`Add`,
        :class:`Multiply`, and the comparison masks :class:`WhereEqual`,
        :class:`WhereNotEqual`, :class:`WhereGreater`, :class:`WhereGreaterEqual`,
        :class:`WhereLess`, :class:`WhereLessEqual`.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache binary elementwise ops fuse into the per-block kernel —
        # see ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # ------------------------------------------------------------------ #
    def profile(
        self, x: vTensor, y: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""Trace-time: validate ``x``/``y`` ``[B, N, D]`` (broadcasting on
        ``N``/``D``), resolve the output format, register the op, and return a
        ``vTensor`` view of the ``[B, max(N_x,N_y), max(D_x,D_y)]`` output."""
        prefix = self._prefix()

        # --- type checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}y must be vTensor, got {type(y)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"

        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert y.dim() == 3, f"{prefix}y must be 3D, got ndim={y.dim()} shape={tuple(y.shape)}"

        # --- broadcastability checks ---
        assert (
            x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1
        ), f"{prefix}dim-1 not broadcastable: x={x.shape}, y={y.shape}"
        assert (
            x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1
        ), f"{prefix}dim-2 not broadcastable: x={x.shape}, y={y.shape}"

        exp_N, exp_D = max(x.shape[1], y.shape[1]), max(x.shape[2], y.shape[2])

        # Case A: output None → allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = FORMAT.RAGGED
            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, exp_N, exp_D),
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

        # Case B: output provided → output_format follows output._format.
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )
        assert output._format in (FORMAT.PAGED, FORMAT.RAGGED, FORMAT.SLOTTED), (
            f"{prefix}output._format must be PAGED, RAGGED or SLOTTED, got {output._format}"
        )
        self.output_format = output._format

        assert output.shape[1] == exp_N and output.shape[2] == exp_D, (
            f"{prefix}output shape mismatch. Expected (*,{exp_N},{exp_D}), got {tuple(output.shape)}"
        )
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


class Maximum(Elementwise_Binary):
    r"""
    Elementwise maximum (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \max(X_{b,n,d},\, Y_{b,n,d}).
    :__init__: ``Maximum(alpha=1.0, beta=1.0)`` — ``alpha`` / ``beta`` unused.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Maximum


class Minimum(Elementwise_Binary):
    r"""
    Elementwise minimum (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \min(X_{b,n,d},\, Y_{b,n,d}).
    :__init__: ``Minimum(alpha=1.0, beta=1.0)`` — ``alpha`` / ``beta`` unused.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Minimum
        

class Add(Elementwise_Binary):
    r"""
    Affine combination :math:`\alpha x + \beta y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \alpha\,X_{b,n,d} + \beta\,Y_{b,n,d}.
    :__init__: ``Add(alpha=1.0, beta=1.0)`` — multipliers for :math:`x` and
        :math:`y` (defaults give :math:`x+y`).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Add


class Multiply(Elementwise_Binary):
    r"""
    Elementwise product (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = X_{b,n,d}\cdot Y_{b,n,d}.
    :__init__: ``Multiply(alpha=1.0, beta=1.0)`` — ``alpha`` / ``beta`` unused.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Mul


class WhereEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x = y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \begin{cases} 0, & X_{b,n,d} = Y_{b,n,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereEqual()`` — no arguments.
    :Note: additive mask for score-space gating.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereEqual


class WhereNotEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x \ne y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \begin{cases} 0, & X_{b,n,d} \ne Y_{b,n,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereNotEqual()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereNotEqual


class WhereGreater(Elementwise_Binary):
    r"""
    Comparison mask :math:`x > y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \begin{cases} 0, & X_{b,n,d} > Y_{b,n,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereGreater()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreater


class WhereGreaterEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x \ge y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \begin{cases} 0, & X_{b,n,d} \ge Y_{b,n,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereGreaterEqual()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreaterEqual


class WhereLess(Elementwise_Binary):
    r"""
    Comparison mask :math:`x < y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \begin{cases} 0, & X_{b,n,d} < Y_{b,n,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereLess()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLess


class WhereLessEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x \le y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{b,n,d} = \begin{cases} 0, & X_{b,n,d} \le Y_{b,n,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereLessEqual()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLessEqual
