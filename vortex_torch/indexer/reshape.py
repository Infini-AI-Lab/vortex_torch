import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Reshape(vOp):
    r"""
    Same-numel reshape of the inner two axes (indexer side).

    :Math:
        .. math::

            X\in\mathbb{R}^{S\times x_1\times y_1} \;\longrightarrow\;
            Y\in\mathbb{R}^{S\times x_2\times y_2},\qquad x_2\,y_2 = x_1\,y_1,

        reading the flat :math:`x_1 y_1` elements row-major into the new
        :math:`(x_2, y_2)` layout, independently per leading index :math:`s`
        (a Triton-tile :func:`tl.reshape`; no data movement beyond the
        existing load/store).
    :__init__: ``Reshape(-1, x2, y2)`` — the leading dim must be ``-1`` (the
        ``S`` axis is preserved); ``x2*y2`` must equal the input's ``x1*y1``
        (checked at trace time).
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, x_1, y_1]`` →
        ``[S, x_2, y_2]``. ``BATCHED`` iff the input is ``BATCHED``, else
        ``RAGGED``.
    """

    def __init__(self, batch_dim: int, x2: int, y2: int):
        super().__init__()
        cls = self.__class__.__name__
        assert batch_dim == -1, (
            f"{cls}.__init__: leading dim must be -1 (auto-infer; the "
            f"leading S axis is preserved), got {batch_dim}"
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
        # Fused into the per-workload kernel.
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` ``[S, x1, y1]`` and the same-numel
        constraint (``x1*y1 == x2*y2``), resolve the output format, register
        the op, and return a ``vTensor`` view of the ``[S, x2, y2]`` output."""
        prefix = self._prefix()

        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, x1, y1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        x1, y1 = x.shape[1], x.shape[2]
        in_numel = x1 * y1
        out_numel = self.x2 * self.y2
        assert in_numel == out_numel, (
            f"{prefix}same-numel reshape required: "
            f"x.shape[1]*x.shape[2] = {x1}*{y1} = {in_numel}  vs  "
            f"target x2*y2 = {self.x2}*{self.y2} = {out_numel}"
        )

        # Output is BATCHED iff the input is BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        # Pure-metadata vTensor — leading dim is a placeholder; the
        # runtime knows the actual ``S`` from the pipeline.
        self.output_buffer = vTensor(
            shape=(0, self.x2, self.y2),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # The fused ``tl.reshape`` (Schedule.W) needs pow2 inner dims AND an
        # unpadded tile — under padding it would mix padding into real positions.
        # When either side's real inner dims aren't pow2 (padded), fall back to a
        # standalone ``Schedule.S`` torch reshape over the REAL ``[:, :x1, :y1] ->
        # [:, :x2, :y2]`` region (slicing the real data avoids any padding mixup;
        # format-agnostic, any dims). Otherwise keep the fast fused path.
        self.schedule = (
            Schedule.S
            if (x.needs_padding() or self.output_buffer.needs_padding())
            else Schedule.W
        )

        # Register in the indexer graph.
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
