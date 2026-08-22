import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ReduceType, QuantizationType, Schedule
from typing import Optional


class ReduceInterleave(vOp):
    r"""
    Interleaved 1-D reduction — folds consecutive groups of ``k`` elements.

    :Math:
        For input :math:`X\in\mathbb{R}^{B\times N\times D}` and a per-group
        reduction :math:`\rho` (mean / max / min / L2-norm, fixed by the
        subclass), each group of :math:`k` adjacent elements collapses to one:

        .. math::

            \begin{aligned}
            (\text{dim}=1,\ N\%k=0):\quad & Y_{b,m,d} = \rho_{\,0 \le j < k}\, X_{b,\,mk+j,\,d}, \\
            (\text{dim}=2,\ D\%k=0):\quad & Y_{b,n,e} = \rho_{\,0 \le j < k}\, X_{b,n,\,ek+j}.
            \end{aligned}
    :__init__: ``ReduceInterleave(dim=1, k=2)`` — inner axis (``1`` over
        :math:`N`, ``2`` over :math:`D`) and group size ``k`` (must divide the
        reduced axis; ``k=1`` is identity, ``k`` = full length recovers the
        plain reduction).
    :__call__: ``op(x, output, loc=loc, ctx=ctx)`` — ``x`` ``[B, N, D]``; the
        reduced axis shrinks by a factor of ``k`` (``N→N/k`` or ``D→D/k``).
        ``PAGED`` iff a ``PAGED`` ``output`` is supplied, else ``RAGGED``.
    :Note: use a concrete subclass — :class:`MeanInterleave`,
        :class:`MaxInterleave`, :class:`MinInterleave`,
        :class:`L2NormInterleave`.
    """

    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__()
        self.dim = dim
        self.k = k
        self.reduce_type: Optional[ReduceType] = None
        self.quantization_type: Optional[QuantizationType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache reductions fuse into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W
        # Validate reduction dimension and group size at construction time.
        # Cache reduce always runs inside the per-block fused kernel, so
        # dim==0 (which would need to span blocks/tokens) is not allowed —
        # use an indexer ``Reduce(dim=0)`` if you need a cross-row summary.
        cls = self.__class__.__name__
        assert self.dim in (1, 2), (
            f"{cls}.__init__: dim must be 1 or 2 (cache reduce cannot operate on dim=0), "
            f"got dim={self.dim}"
        )
        assert isinstance(self.k, int) and self.k >= 1, (
            f"{cls}.__init__: k must be a positive int, got k={self.k!r}"
        )

    # ------------------------------ helpers ------------------------------ #
    def _resolve_quantization(self, x: vTensor) -> QuantizationType:
        r"""Map ``x.dtype`` to the matching :class:`QuantizationType`.

        Centralized so both branches of :meth:`profile` agree on FP8/BF16
        dispatch and we don't silently leave ``quantization_type`` as ``None``.
        """
        prefix = self._prefix()
        # ``logical_dtype``, not ``dtype``: a packed-INT4 tensor is stored as uint8 but its load site
        # yields a dequantized fp32 block, so the question "can this be reduced?" is about the
        # DECODED value. Dispatching on the container instead gave
        # `unsupported dtype torch.uint8 for reduction` on a tensor the codegen unpacks for free.
        if x.int4 is not None:
            # The unpack already produced fp32, so the load needs no further decode -- BF16 is the
            # "no bitcast on load" case, which is what applies here.
            return QuantizationType.BF16
        if x.logical_dtype == torch.bfloat16:
            return QuantizationType.BF16
        if x.logical_dtype == torch.float8_e5m2:
            return QuantizationType.FP8_E5M2
        if x.logical_dtype == torch.float8_e4m3fn:
            return QuantizationType.FP8_E4M3
        raise ValueError(f"{prefix}unsupported dtype {x.dtype} for reduction")

    # --------------------------------------------------------------------- #
    # profile: validate, pick format, and return the provided vTensor
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""Trace-time: validate ``x`` ``[B, N, D]`` (reduced axis divisible by
        ``k``, and ``output`` if given), register the op, and return a
        ``vTensor`` view of the folded output (see the class docstring for
        shapes)."""
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        N, D = x.shape[1], x.shape[2]

        # Divisibility check on the reduced axis.
        if self.dim == 1:
            assert N % self.k == 0, (
                f"{prefix}profile(dim=1): expected x.shape[1] (N={N}) divisible by k={self.k}"
            )
            exp_N, exp_D = N // self.k, D
        else:  # self.dim == 2
            assert D % self.k == 0, (
                f"{prefix}profile(dim=2): expected x.shape[2] (D={D}) divisible by k={self.k}"
            )
            exp_N, exp_D = N, D // self.k

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

        # Shape checks per reduction dim and group size.
        assert output.shape[1] == exp_N, (
            f"{prefix}profile(dim={self.dim}, k={self.k}): expected output.shape[1] == {exp_N}, "
            f"got {tuple(output.shape)}"
        )
        assert output.shape[2] == exp_D, (
            f"{prefix}profile(dim={self.dim}, k={self.k}): expected output.shape[2] == {exp_D}, "
            f"got {tuple(output.shape)}"
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


class MeanInterleave(ReduceInterleave):
    r"""
    Interleaved mean over groups of ``k`` (a :class:`ReduceInterleave`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,m,d} = \frac{1}{k}\sum_{j=0}^{k-1} X_{b,\,mk+j,\,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,e} = \frac{1}{k}\sum_{j=0}^{k-1} X_{b,n,\,ek+j}.
            \end{aligned}
    :__init__: ``MeanInterleave(dim=1, k=2)`` — inner axis (``1``/``2``) and
        group size ``k``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.Mean


class MaxInterleave(ReduceInterleave):
    r"""
    Interleaved max over groups of ``k`` (a :class:`ReduceInterleave`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,m,d} = \max_{0 \le j < k} X_{b,\,mk+j,\,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,e} = \max_{0 \le j < k} X_{b,n,\,ek+j}.
            \end{aligned}
    :__init__: ``MaxInterleave(dim=1, k=2)`` — inner axis (``1``/``2``) and
        group size ``k``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.Max


class MinInterleave(ReduceInterleave):
    r"""
    Interleaved min over groups of ``k`` (a :class:`ReduceInterleave`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,m,d} = \min_{0 \le j < k} X_{b,\,mk+j,\,d}, \\
            (\text{dim}=2):\quad & Y_{b,n,e} = \min_{0 \le j < k} X_{b,n,\,ek+j}.
            \end{aligned}
    :__init__: ``MinInterleave(dim=1, k=2)`` — inner axis (``1``/``2``) and
        group size ``k``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.Min


class L2NormInterleave(ReduceInterleave):
    r"""
    Interleaved L2-norm over groups of ``k`` (a :class:`ReduceInterleave`).

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{b,m,d} = \Big(\sum_{j=0}^{k-1} X_{b,\,mk+j,\,d}^2\Big)^{1/2}, \\
            (\text{dim}=2):\quad & Y_{b,n,e} = \Big(\sum_{j=0}^{k-1} X_{b,n,\,ek+j}^2\Big)^{1/2}.
            \end{aligned}
    :__init__: ``L2NormInterleave(dim=1, k=2)`` — inner axis (``1``/``2``) and
        group size ``k``.
    :Note: a pure :math:`L_2` norm per group (no division) — not RMS.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.L2Norm
