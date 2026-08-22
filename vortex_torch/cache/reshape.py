import torch
from typing import Optional
from .context import Context
from ..abs import vOp, vTensor, FORMAT
from ..utils import QuantizationType, Schedule


class Reshape(vOp):
    r"""
    Same-numel reshape of a cache block's inner two axes.

    :Math:
        .. math::

            X\in\mathbb{R}^{1\times x_1\times y_1} \;\longrightarrow\;
            Y\in\mathbb{R}^{1\times x_2\times y_2},\qquad x_2\,y_2 = x_1\,y_1,

        reading the flat :math:`x_1 y_1` elements row-major into the new
        :math:`(x_2, y_2)` layout (a Triton-tile :func:`tl.reshape`; no data
        movement beyond the existing load/store).
    :__init__: ``Reshape(-1, x2, y2)`` — the leading dim must be ``-1`` (the
        per-block batch axis, always 1); ``x2*y2`` must equal the input's
        ``x1*y1`` (checked at trace time).
    :__call__: ``y = op(x, output, loc=loc, ctx=ctx)`` — ``x``
        ``[1, x_1, y_1]`` → ``[1, x_2, y_2]``. ``PAGED`` iff a ``PAGED``
        ``output`` is supplied, else ``RAGGED``.
    """

    def __init__(self, batch_dim: int, x2: int, y2: int):
        super().__init__()
        cls = self.__class__.__name__
        assert batch_dim == -1, (
            f"{cls}.__init__: leading dim must be -1 (auto-infer; the "
            f"per-block batch axis is always 1 on the cache side), got "
            f"{batch_dim}"
        )
        assert isinstance(x2, int) and x2 >= 1, (
            f"{cls}.__init__: x2 must be a positive int, got {x2!r}"
        )
        assert isinstance(y2, int) and y2 >= 1, (
            f"{cls}.__init__: y2 must be a positive int, got {y2!r}"
        )
        self.x2 = x2
        self.y2 = y2

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        self.quantization_type: Optional[QuantizationType] = None
        # Cache reshape fuses into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # ------------------------------ helpers ------------------------------ #
    def _resolve_quantization(self, x: vTensor) -> QuantizationType:
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
        raise ValueError(f"{prefix}unsupported dtype {x.dtype} for reshape")

    # ---------------- profile ----------------
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate, resolve formats, and register the reshape in the cache
        graph.

        Checks ``x.dim() == 3`` and the same-numel constraint
        ``x.shape[1] * x.shape[2] == self.x2 * self.y2``. If ``output``
        is supplied, also validates its shape matches
        ``[*, self.x2, self.y2]``.
        """
        prefix = self._prefix()

        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), (
            f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        )
        assert x.dim() == 3, (
            f"{prefix}x must be 3D [B, x1, y1], got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        x1, y1 = x.shape[1], x.shape[2]
        in_numel = x1 * y1
        out_numel = self.x2 * self.y2
        assert in_numel == out_numel, (
            f"{prefix}same-numel reshape required: "
            f"x.shape[1]*x.shape[2] = {x1}*{y1} = {in_numel}  vs  "
            f"target x2*y2 = {self.x2}*{self.y2} = {out_numel}"
        )

        # Case A: no preallocated output — allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = FORMAT.RAGGED
            self.quantization_type = self._resolve_quantization(x)

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, self.x2, self.y2),
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

        # Case B: caller-provided output -> output_format follows output._format.
        assert isinstance(output, vTensor), (
            f"{prefix}output must be vTensor, got {type(output)}"
        )
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )
        assert output.shape[1] == self.x2 and output.shape[2] == self.y2, (
            f"{prefix}output inner shape must be ({self.x2}, {self.y2}), "
            f"got {tuple(output.shape)}"
        )
        assert output._format in (FORMAT.PAGED, FORMAT.RAGGED, FORMAT.SLOTTED), (
            f"{prefix}output._format must be PAGED, RAGGED or SLOTTED, got {output._format}"
        )
        self.output_format = output._format

        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        self.quantization_type = self._resolve_quantization(x)

        # Register in the cache graph. Caller-provided ``output`` already has
        # a tensor_id in ``ctx.tensor_list``; we claim it as this op's product.
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output
