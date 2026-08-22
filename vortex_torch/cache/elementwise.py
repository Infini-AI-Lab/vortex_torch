import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ElementwiseOpType, Schedule
from typing import Optional

class Elementwise(vOp):
    r"""
    Unary elementwise op — applies a scalar function pointwise.

    :Math:
        .. math::

            Y_{b,n,d} = f(X_{b,n,d};\, \alpha, \beta),

        where :math:`f` is fixed by the subclass (ReLU / SiLU / Sigmoid /
        affine / abs / log / exp).
    :__init__: ``Elementwise(alpha=0.0, beta=1.0)`` — scalar parameters
        :math:`\alpha`, :math:`\beta` consumed by :math:`f`.
    :__call__: ``y = op(x, output, loc=loc, ctx=ctx)`` — ``x`` is ``[B, N, D]``;
        ``output`` is an optional preallocated buffer (``None`` → fresh
        ``RAGGED``). Returns the same ``[B, N, D]`` shape; ``PAGED`` iff a
        ``PAGED`` ``output`` is supplied, else ``RAGGED``.
    :Note: use a concrete subclass — :class:`Relu`, :class:`Silu`,
        :class:`Sigmoid`, :class:`Add_Mul`, :class:`Abs`, :class:`Log`,
        :class:`Exp`.
    """

    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[ElementwiseOpType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache elementwise ops fuse with neighbours into a single
        # token-driven kernel — see ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # --------------------------------------------------------------------- #
    # profile: validate and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""Trace-time: validate ``x`` ``[B, N, D]`` (and ``output`` if given),
        register the op, and return a ``vTensor`` view of the same-shape output
        (a fresh ``RAGGED`` buffer when ``output is None``)."""
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        N, D = x.shape[1], x.shape[2]

        # Case A: output not provided -> allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = FORMAT.RAGGED

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, N, D),
                dtype=ctx.vortex_dtype,
                device=x.device,
                _format=self.output_format,
                tensor_id=len(ctx.tensor_list),
            )
            ctx.tensor_list.append(self.output_buffer)
            ctx.output_tensor_to_op_list.append(len(ctx.op_list))
            ctx.op_list.append(self)
            ctx.op_to_input_tensor_list.append([x.tensor_id])
            ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
            return self.output_buffer

        # Case B: output provided -> output_format follows output._format
        # (PAGED iff caller supplied a PAGED tensor; otherwise RAGGED).
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, "
            f"got ndim={output.dim()} shape={tuple(output.shape)}"
        )
        assert output._format in (FORMAT.PAGED, FORMAT.RAGGED, FORMAT.SLOTTED), (
            f"{prefix}output._format must be PAGED, RAGGED or SLOTTED, got {output._format}"
        )
        self.output_format = output._format

        # Shape consistency: unary elementwise keeps (N,D)
        assert output.shape[1] == N and output.shape[2] == D, (
            f"{prefix}output shape mismatch. Expected (*,{N},{D}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        # Register in the cache graph. ``output`` is provided by the caller
        # and assumed to already have a valid ``tensor_id`` and to live in
        # ``ctx.tensor_list``. We claim it as this op's produced tensor
        # (override producer), mirroring ``indexer.save_load.Save``.
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output


class Relu(Elementwise):
    r"""
    ReLU-like activation with threshold/fallback (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \begin{cases} x, & x \ge \alpha, \\ \beta, & x < \alpha. \end{cases}
    :__init__: ``Relu(alpha=0.0, beta=0.0)`` — threshold :math:`\alpha`,
        fallback value :math:`\beta` (used when :math:`x<\alpha`).
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu


class Silu(Elementwise):
    r"""
    SiLU-like activation with configurable shift/slope (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \frac{x}{1 + \exp(\beta x + \alpha)}.
    :__init__: ``Silu(alpha=0.0, beta=0.0)`` — bias :math:`\alpha`, slope
        :math:`\beta` inside the exponential.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu


class Sigmoid(Elementwise):
    r"""
    Sigmoid activation with configurable shift/slope (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \frac{1}{1 + \exp(\beta x + \alpha)}.
    :__init__: ``Sigmoid(alpha=0.0, beta=0.0)`` — bias :math:`\alpha`, slope
        :math:`\beta` inside the exponential.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid


class Add_Mul(Elementwise):
    r"""
    Affine transform :math:`\beta x + \alpha` (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \beta x + \alpha.
    :__init__: ``Add_Mul(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta`.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    Absolute value of an affine transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \lvert \beta x + \alpha \rvert.
    :__init__: ``Abs(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` inside the absolute value.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs


class Log(Elementwise):
    r"""
    Natural logarithm of an affine transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \log(\beta x + \alpha).
    :__init__: ``Log(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` inside the logarithm.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Log


class Exp(Elementwise):
    r"""
    Exponential of an affine transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \exp(\beta x + \alpha).
    :__init__: ``Exp(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` inside the exponential.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Exp
