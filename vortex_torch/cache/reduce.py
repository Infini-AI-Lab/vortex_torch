import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ReduceType, QuantizationType, Schedule
from typing import Optional


class Reduce(vOp):
    r"""
    Generic 1-D reduction over one inner axis of a rank-3 cache tensor.

    :Math:
        For input :math:`X\in\mathbb{R}^{B\times N\times D}` and a per-axis
        reduction :math:`\rho` (mean / max / min / L2-norm, fixed by the
        subclass):

        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,0,d} = \rho_{\,0 \le i < N}\, X_{b,i,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,0} = \rho_{\,0 \le j < D}\, X_{b,n,j}.
            \end{aligned}
    :__init__: ``Reduce(dim=1)`` — inner axis to reduce, ``1`` (over :math:`N`)
        or ``2`` (over :math:`D`).
    :__call__: ``op(x, output, loc=loc, ctx=ctx)`` — runs once per page in
        ``forward_cache``; ``x`` is ``[B, N, D]`` and the reduced axis becomes
        size 1. ``PAGED`` iff a ``PAGED`` ``output`` is supplied, else
        ``RAGGED``.
    :Note: use a concrete subclass — :class:`Mean`, :class:`Max`,
        :class:`Min`, :class:`L2Norm`. Cache-side reductions support
        ``dim ∈ {1, 2}`` only.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.quantization_type: Optional[QuantizationType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache reductions fuse into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W
        # Validate reduction dimension at construction time. Cache reduce
        # always runs inside the per-block fused kernel, so dim==0 (which
        # would need to span blocks/tokens) is not allowed — use an indexer
        # ``Reduce(dim=0)`` if you need a cross-row summary.
        cls = self.__class__.__name__
        assert self.dim in (1, 2), (
            f"{cls}.__init__: dim must be 1 or 2 (cache reduce cannot operate on dim=0), "
            f"got dim={self.dim}"
        )

    # ------------------------------ helpers ------------------------------ #
    def _resolve_quantization(self, x: vTensor) -> QuantizationType:
        r"""Map ``x.dtype`` to the matching :class:`QuantizationType`.

        Centralized so both branches of :meth:`profile` agree on FP8/BF16
        dispatch and we don't silently leave ``quantization_type`` as ``None``.
        """
        prefix = self._prefix()
        if x.dtype == torch.bfloat16:
            return QuantizationType.BF16
        if x.dtype == torch.float8_e5m2:
            return QuantizationType.FP8_E5M2
        if x.dtype == torch.float8_e4m3fn:
            return QuantizationType.FP8_E4M3
        raise ValueError(f"{prefix}unsupported dtype {x.dtype} for reduction")

    # --------------------------------------------------------------------- #
    # profile: validate, pick format, and return the provided vTensor
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""Trace-time: validate ``x`` ``[B, N, D]`` (and ``output`` if given),
        register the op, and return a ``vTensor`` view of the reduced output
        (fresh ``RAGGED`` buffer when ``output is None``; see the class
        docstring for shapes)."""
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        # Compute expected output (N, D) given reduction dim
        # dim==1 -> reduce rows: keep D, set N=1
        # dim==2 -> reduce cols: keep N, set D=1
        exp_N = 1 if self.dim == 1 else x.shape[1]
        exp_D = 1 if self.dim == 2 else x.shape[2]

        # Case A: output not provided -> allocate a RAGGED metadata buffer
        # in ``ctx.vortex_dtype`` (the intermediate dtype used by the cache
        # pipeline, default bf16).
        if output is None:
            self.output_format = FORMAT.RAGGED
            self.quantization_type = self._resolve_quantization(x)

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
            ctx.op_to_input_tensor_list.append([x.tensor_id])
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

        # Shape checks per reduction dim
        if self.dim == 1:
            # Expect (*, 1, x.D)
            assert output.shape[1] == 1, (
                f"{prefix}profile(dim=1): expected output.shape[1] == 1, "
                f"got {tuple(output.shape)}"
            )
            assert output.shape[2] == x.shape[2], (
                f"{prefix}profile(dim=1): expected output.shape[2] == x.shape[2], "
                f"got {output.shape[2]} vs {x.shape[2]}"
            )
        else:  # self.dim == 2
            # Expect (*, x.N, 1)
            assert output.shape[2] == 1, (
                f"{prefix}profile(dim=2): expected output.shape[2] == 1, "
                f"got {tuple(output.shape)}"
            )
            assert output.shape[1] == x.shape[1], (
                f"{prefix}profile(dim=2): expected output.shape[1] == x.shape[1], "
                f"got {output.shape[1]} vs {x.shape[1]}"
            )

        # Device consistency
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        self.quantization_type = self._resolve_quantization(x)

        # Register in the cache graph. Caller-provided ``output`` must already
        # have a valid ``tensor_id`` in ``ctx.tensor_list``; we claim it as
        # this op's produced tensor (override producer slot).
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output


class Mean(Reduce):
    r"""
    Mean reduction over one inner axis (a :class:`Reduce`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,0,d} = \frac{1}{N}\sum_{n=0}^{N-1} X_{b,n,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,0} = \frac{1}{D}\sum_{d=0}^{D-1} X_{b,n,d}.
            \end{aligned}
    :__init__: ``Mean(dim=1)`` — axis to reduce (``1`` → :math:`N`,
        ``2`` → :math:`D`).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean
    


class Max(Reduce):
    r"""
    Max reduction over one inner axis (a :class:`Reduce`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,0,d} = \max_{0 \le n < N} X_{b,n,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,0} = \max_{0 \le d < D} X_{b,n,d}.
            \end{aligned}
    :__init__: ``Max(dim=1)`` — axis to reduce (``1`` → :math:`N`,
        ``2`` → :math:`D`).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max
        


class Min(Reduce):
    r"""
    Min reduction over one inner axis (a :class:`Reduce`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,0,d} = \min_{0 \le n < N} X_{b,n,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,0} = \min_{0 \le d < D} X_{b,n,d}.
            \end{aligned}
    :__init__: ``Min(dim=1)`` — axis to reduce (``1`` → :math:`N`,
        ``2`` → :math:`D`).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min



class L2Norm(Reduce):
    r"""
    L2-norm reduction (not RMS) over one inner axis (a :class:`Reduce`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,0,d} = \Big(\sum_{n=0}^{N-1} X_{b,n,d}^2\Big)^{1/2}, \\
            (\text{dim}=2):\quad & Y_{b,n,0} = \Big(\sum_{d=0}^{D-1} X_{b,n,d}^2\Big)^{1/2}.
            \end{aligned}
    :__init__: ``L2Norm(dim=1)`` — axis to reduce (``1`` → :math:`N`,
        ``2`` → :math:`D`).
    :Note: a pure :math:`L_2` norm (no division by element count) — not RMS.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm