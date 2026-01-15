import torch
import warnings
from ..abs import vOp
from .context import Context
from .triton_kernels.reduce_impl import reduce_pp, reduce_rp, reduce_pr, reduce_rr
from ..abs import vTensor, FORMAT, as_vtensor
from ..utils import ReduceType
from typing import Tuple, Dict, Callable, Optional, Union
from .unified_view import UnifiedCacheView


class Reduce(vOp):
    r"""
    Generic reduction dispatcher over the last two logical axes.

    This dispatcher covers a family of reductions (mean/max/min/L2-norm/sum,
    etc.) on rank-3 tensors. The input is treated as

    .. math::

        X \in \mathbb{R}^{B \times N \times D},

    where:

    - :math:`B` is a leading batch-like axis (typically derived from the
      runtime, e.g. ``max_new_tokens_per_batch * head_num``),
    - :math:`N` is a sequence or position dimension, and
    - :math:`D` is a feature/channel dimension.

    The reduction dimension is chosen by :attr:`dim`:

    - ``dim == 1`` (row-wise reduction over :math:`N`):

      .. math::

         \text{out} \in \mathbb{R}^{B \times 1 \times D},

    - ``dim == 2`` (column-wise reduction over :math:`D`):

      .. math::

         \text{out} \in \mathbb{R}^{B \times N \times 1}.

    The exact reduction operation (mean, max, min, L2-norm, sum, etc.) is
    encoded in :attr:`reduce_type` and interpreted by the implementation.

    Dispatch is based on the pair of tensor formats
    ``(x_format, o_format)`` and a registry mapping:

    .. code-block:: text

        (x_format, o_format) -> (impl, resolved_output_format)

    Policy
    ------
    - If ``output`` is ``None``:

      - :meth:`profile` selects an implementation for
        ``(x_fmt, FORMAT.RAGGED)`` (i.e. with RAGGED output).
      - An internal buffer is allocated with logical shape
        ``[B, exp_N, exp_D]``, where:

        - for ``dim == 1``: ``exp_N = 1``, ``exp_D = D``,
        - for ``dim == 2``: ``exp_N = N``, ``exp_D = 1``.

    - If ``output`` is provided:

      - :meth:`profile` requires an exact implementation key for
        ``(x_fmt, o_fmt)``.
      - The shape of ``output`` must match the expected
        ``(exp_N, exp_D)`` given :attr:`dim`.
      - Device consistency is enforced between ``x`` and ``output``.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]]
        Dispatch table keyed by ``(x_format, o_format)``. Each entry maps to
        ``(callable_impl, resolved_output_format)``.

    dim : int
        Reduction dimension in the logical 3D tensor. Must be either:

        - ``1`` for row-wise reduction over :math:`N`, or
        - ``2`` for column-wise reduction over :math:`D`.

    reduce_type : Optional[ReduceType]
        Enum describing which reduction to perform (mean, max, min, L2-norm,
        sum, etc.).

    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Internal output buffer allocated when ``output`` is ``None``.
    """

    # Consistent 2-tuple dispatch table
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED,  FORMAT.PAGED):  (reduce_pp, FORMAT.PAGED),
        (FORMAT.RAGGED, FORMAT.PAGED):  (reduce_rp, FORMAT.PAGED),
        (FORMAT.PAGED,  FORMAT.RAGGED): (reduce_pr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.RAGGED): (reduce_rr, FORMAT.RAGGED),
    }

    def __init__(self, dim: int = 1):
        r"""
        Initialize a reduction dispatcher.

        Parameters
        ----------
        dim : int, optional
            Logical reduction dimension:

            - ``1`` to reduce over the sequence axis :math:`N` (rows),
            - ``2`` to reduce over the feature axis :math:`D` (columns).

            Default is ``1``.

        Notes
        -----
        The specific reduction operation (mean, max, min, L2-norm, sum,
        etc.) must be configured separately via :attr:`reduce_type`, e.g.
        in subclasses like :class:`Mean`, :class:`Max`, :class:`Min`,
        :class:`L2Norm`, :class:`Sum`.
        """
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        # Validate reduction dimension at construction time.
        cls = self.__class__.__name__
        assert self.dim in (1, 2), f"{cls}.__init__: dim must be 1 or 2, got dim={self.dim}"

    # ------------------------------ helpers ------------------------------ #
    def _infer_impl_ragged(self, x_fmt: FORMAT) -> Tuple[Callable, FORMAT]:
        r"""
        Infer an implementation assuming a RAGGED output format.

        This helper is used when :meth:`profile` is called with
        ``output is None``. It selects an implementation for the key
        ``(x_fmt, FORMAT.RAGGED)`` in :attr:`_impl_map`.

        Parameters
        ----------
        x_fmt : FORMAT
            The format of the input tensor ``x``.

        Returns
        -------
        (Callable, FORMAT)
            The implementation callable and the resolved output format.

        Raises
        ------
        AssertionError
            If there is no entry for ``(x_fmt, FORMAT.RAGGED)`` in
            :attr:`_impl_map`.
        """
        key = (x_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]  # -> (impl, out_fmt)

    # --------------------------------------------------------------------- #
    # profile: validate, pick impl/format, and return the provided vTensor
    # --------------------------------------------------------------------- #
    def profile(
        self, x: Union[vTensor, UnifiedCacheView], output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs, resolve the reduction implementation and output
        format, and optionally allocate an internal output buffer.

        The input tensor ``x`` is expected to have logical shape
        ``[B, N, D]``. According to :attr:`dim`, the expected output
        logical shape is:

        - ``dim == 1``: ``[B, 1, D]``
        - ``dim == 2``: ``[B, N, 1]``

        The auxiliary tensor ``loc`` carries per-position metadata used
        by the implementation; its shape and semantics are
        implementation-defined.

        Parameters
        ----------
        x : vTensor or UnifiedCacheView
            Input tensor with logical shape ``[B, N, D]``. Can be a UnifiedCacheView
            for CPU/GPU hybrid memory.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            buffer with shape ``[B, exp_N, exp_D]`` is allocated using
            ``ctx.max_new_tokens_per_batch * ctx.head_num`` for ``B`` and a
            RAGGED-output implementation is selected. If not ``None``,
            this tensor must have rank 3 and shape compatible with
            :attr:`dim` as described above.

        loc : torch.Tensor
            Auxiliary tensor carrying metadata required by the reduction
            implementation.

        ctx : Context
            Execution context that provides the runtime value of ``B`` and
            is used for auxiliary memory accounting.

        Returns
        -------
        vTensor
            A :class:`vTensor` view representing the resolved output:
            either the provided ``output`` or an internally allocated
            buffer.

        Raises
        ------
        AssertionError
            If types, ranks, formats, shapes, or devices are incompatible,
            or if no implementation is found in :attr:`_impl_map`.
        """
        prefix = self._prefix()

        # --- NEW: Check if x is UnifiedCacheView ---
        is_unified = isinstance(x, UnifiedCacheView)

        if is_unified:
            # Set unified execution mode
            self.impl = self._unified_reduce
            self.is_unified_execution = True
            self.unified_view = x
            self.output_format = x.format

            # Compute expected output shape
            exp_N = 1 if self.dim == 1 else x.shape[1]
            exp_D = 1 if self.dim == 2 else x.shape[2]

            if output is None:
                # Allocate GPU output buffer
                B = ctx.max_new_tokens_per_batch * ctx.head_num
                self.output_buffer = torch.empty(
                    (B, exp_N, exp_D),
                    device=x.device,
                    dtype=x.dtype,
                )
                ctx.add_aux_memory(self.output_buffer)
                return as_vtensor(self.output_buffer, self.output_format)
            else:
                # Validate provided output - accept both vTensor and regular Tensor
                if isinstance(output, vTensor):
                    output_tensor = output
                elif isinstance(output, torch.Tensor):
                    # Wrap regular tensor as vTensor for unified mode
                    output_tensor = as_vtensor(output, self.output_format)
                else:
                    raise TypeError(f"{prefix}output must be vTensor or torch.Tensor, got {type(output)}")

                assert output_tensor.device == x.device, f"{prefix}output must be on same device as input"
                self.output_buffer = output if isinstance(output, torch.Tensor) else output._t
                return output_tensor

        # --- EXISTING: type & rank checks for vTensor ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        x_fmt = x._format

        # Compute expected output (N, D) given reduction dim
        # dim==1 -> reduce rows: keep D, set N=1
        # dim==2 -> reduce cols: keep N, set D=1
        exp_N = 1 if self.dim == 1 else x.shape[1]
        exp_D = 1 if self.dim == 2 else x.shape[2]

        # Case A: output not provided -> infer impl (RAGGED) and allocate buffer
        if output is None:
            self.impl, self.output_format = self._infer_impl_ragged(x_fmt)

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = torch.empty(
                (B, exp_N, exp_D),
                device=x.device,
                dtype=x.dtype,
            )
            ctx.add_aux_memory(self.output_buffer)
            return as_vtensor(self.output_buffer, self.output_format)

        # Case B: output provided -> validate and pick exact impl by (x_fmt, o_fmt)
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )

        o_fmt = output._format
        key = (x_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

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

        return output

    # --------------------------------------------------------------------- #
    # execute: run impl and return the (plain) output tensor
    # --------------------------------------------------------------------- #
    def execute(
        self, x: torch.Tensor, output: Optional[torch.Tensor], loc: torch.Tensor, ctx: Context
    ) -> torch.Tensor:
        r"""
        Execute the selected reduction implementation.

        This method assumes that :meth:`profile` has already selected an
        implementation and, if needed, allocated an internal output buffer.

        Parameters
        ----------
        x : torch.Tensor
            Plain input tensor with shape compatible with the
            :class:`vTensor` validated in :meth:`profile`.

        output : Optional[torch.Tensor]
            Optional preallocated output tensor. If ``None``, the internal
            buffer created during :meth:`profile` will be used.

        loc : torch.Tensor
            Auxiliary tensor carrying metadata required by the reduction
            implementation.

        ctx : Context
            Execution context forwarded to the implementation.

        Returns
        -------
        torch.Tensor
            The output tensor written by the implementation: either the
            provided ``output`` or the internal buffer.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation or
            internal buffer is available.
        """
        prefix = self._prefix()

        # Must have selected impl in profile()
        assert self.impl is not None, f"{prefix}called before profile() (impl is None)"

        if output is None:
            assert self.output_buffer is not None, (
                f"{prefix}internal output buffer is None; did profile() run?"
            )
            output = self.output_buffer

        # --- Check for unified execution mode at runtime ---
        if isinstance(x, UnifiedCacheView):
            return self._execute_unified(x, output, loc, ctx)

        # --- EXISTING: Launch the kernel/implementation ---
        self.impl(x, output, loc, ctx, self.dim, self.reduce_type)
        return output

    # --------------------------------------------------------------------- #
    # Unified CPU/GPU execution methods
    # --------------------------------------------------------------------- #
    def _execute_unified(
        self,
        x: UnifiedCacheView,
        output: torch.Tensor,
        loc: torch.Tensor,
        ctx: Context
    ) -> torch.Tensor:
        """
        Execute unified CPU/GPU reduction using CUDA kernel.

        This method is called when the input is a UnifiedCacheView. It
        dispatches to the custom CUDA kernel that can read from both CPU
        pinned memory and GPU memory via Unified Virtual Addressing.

        Parameters
        ----------
        x : UnifiedCacheView
            Unified cache view with CPU and GPU buffers
        output : torch.Tensor
            Output tensor (always GPU memory)
        loc : torch.Tensor
            Token position metadata
        ctx : Context
            Execution context

        Returns
        -------
        torch.Tensor
            Output tensor with reduction results
        """
        import vortex_torch_C

        cpu_base = x.get_cpu_base_ptr() if x.has_cpu_pages() else 0
        gpu_base = x.get_gpu_base_ptr()

        vortex_torch_C.unified_reduce(
            output,
            loc,
            x.cpu_to_gpu_slot_map,
            cpu_base,
            gpu_base,
            x.shape[1],  # x_D0 (page rows)
            x.shape[2],  # x_D1 (page cols)
            ctx.head_num,
            ctx.page_size,
            self.reduce_type.value,
            self.dim,
            x.num_cpu_slots
        )

        return output

    def _unified_reduce(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        loc: torch.LongTensor,
        ctx: Context,
        dim: int,
        reduce_type: ReduceType,
    ):
        """
        Placeholder implementation function for unified path.

        The actual execution happens in _execute_unified(). This method
        exists to satisfy the impl attribute requirement.
        """
        pass  # Execution logic is in _execute_unified()



class Mean(Reduce):
    r"""
    Mean reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the arithmetic mean along one of the inner dimensions, as
    configured by :attr:`dim`:

    - ``dim == 1``: row-wise mean over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \frac{1}{N} \sum_{n=0}^{N-1} X[b, n, d].

    - ``dim == 2``: column-wise mean over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \frac{1}{D} \sum_{d=0}^{D-1} X[b, n, d].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean
    


class Max(Reduce):
    r"""
    Max reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the maximum along one of the inner dimensions, as
    configured by :attr:`dim`:

    - ``dim == 1``: row-wise maximum over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \max_{0 \le n < N} X[b, n, d].

    - ``dim == 2``: column-wise maximum over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \max_{0 \le d < D} X[b, n, d].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max
        


class Min(Reduce):
    r"""
    Min reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the minimum along one of the inner dimensions, as
    configured by :attr:`dim`:

    - ``dim == 1``: row-wise minimum over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \min_{0 \le n < N} X[b, n, d].

    - ``dim == 2``: column-wise minimum over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \min_{0 \le d < D} X[b, n, d].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min



class L2Norm(Reduce):
    r"""
    L2-norm reduction (not RMS) over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by computing an L2 norm along one of the inner dimensions, as
    configured by :attr:`dim`. The reduction is *not* normalized by the
    number of elements (it is an L2 norm, not an RMS):

    - ``dim == 1``: row-wise L2 norm over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \sqrt{\sum_{n=0}^{N-1} X[b, n, d]^2}.

    - ``dim == 2``: column-wise L2 norm over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \sqrt{\sum_{d=0}^{D-1} X[b, n, d]^2}.

    Notes
    -----
    This is a pure L2 norm over the reduced axis:

    .. math::

        \|v\|_2 = \sqrt{\sum_i v_i^2},

    with no division by the number of elements. It should not be
    confused with RMS (root mean square).

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm