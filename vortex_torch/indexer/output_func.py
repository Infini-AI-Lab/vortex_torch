import torch
from typing import Dict, Callable, List, Optional
from ..abs import vOp
from vortex_torch_C import topk_output, topk_output_sglang, topk_profile_histogram
from .context import Context
from ..abs import vTensor, FORMAT
from ..utils import UNSET

# --- Module-level histogram accumulator for offline calibration ---
_calibration_histograms: List[torch.Tensor] = []

def get_calibration_histograms() -> List[torch.Tensor]:
    """Return collected histogram tensors (each [eff_bs, 256] int32 on CPU)."""
    return _calibration_histograms

def clear_calibration_histograms() -> None:
    """Clear all collected calibration histograms."""
    _calibration_histograms.clear()

class topK(vOp):
    r"""
    Piecewise top-k dispatcher for packed sequences with reserved pages.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times 1 \times 1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments:

    .. math::

        S_{\text{pack}} = \sum_{b=0}^{B-1} S_b, \qquad
        X =
        \begin{bmatrix}
            X_0 \\
            X_1 \\
            \vdots \\
            X_{B-1}
        \end{bmatrix},

    with

    .. math::

        X_b \in \mathbb{R}^{S_b \times 1 \times 1}.

    For each segment :math:`b`, the operator selects a subset of pages
    according to scores stored in ``x`` (and additional key/value
    metadata from :class:`Context`), but **always** preserves:

    - the first ``page_reserved_bos`` pages in that segment,
    - the last ``page_reserved_eos`` pages in that segment, and
    - an additional ``topk_val`` pages chosen by top-k over the remaining
      candidates in the segment.

    Let :math:`\mathcal{I}_b \subset [0, S_{\text{pack}})` denote the index
    range of segment :math:`b`. The implementation computes a subset
    :math:`\mathcal{J}_b \subset \mathcal{I}_b` such that:

    - all indices corresponding to the reserved prefix (BOS) and suffix
      (EOS) pages in :math:`\mathcal{I}_b` are included, and
    - up to ``topk_val`` additional indices are selected by score.

    The result is written into a preallocated output tensor ``o``; the
    exact layout of ``o`` is defined by the upstream contract and the
    implementation.

    Key properties
    --------------
    - Dispatch is keyed **only** by the input format ``x._format``.
    - The operation is logically out-of-place, but writes into ``o`` in-place.
    - :meth:`profile` only validates and selects the implementation; it does
      not allocate or return any buffers.
    - :meth:`execute` performs the per-segment selection using context
      metadata (indptr arrays, indices, reserved-page counts, and ``topk_val``).

    Attributes
    ----------
    _impl_map : Dict[FORMAT, Callable]
        Dispatch table keyed by ``x_format``. Each entry maps to a callable
        implementation.
    impl : Optional[Callable]
        The resolved implementation selected during :meth:`profile`.
    """

    # Dispatch by input format; only RAGGED is supported for now.
    _impl_map: Dict[FORMAT, Dict[str, Callable]] = {
        FORMAT.RAGGED: {
            "naive": topk_output,
            "sglang": topk_output_sglang,
        },
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.topk_type: str = "naive"
        self.last_histograms: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> None:
        r"""
        Validate input/output tensors and select the implementation.

        This method checks:

        - that ``x`` and ``o`` are both rank-3 :class:`vTensor` objects,
        - that ``x`` has shape ``[S_pack, 1, 1]`` (one scalar score per page),
        - that a top-k implementation is registered for ``x._format``, and
        - that ``x`` and ``o`` reside on the same device.

        No buffers are allocated here and nothing is returned; this call
        simply sets :attr:`impl` so that :meth:`execute` can run.

        Parameters
        ----------
        x : vTensor
            Input tensor carrying per-page scalar scores, with logical shape
            ``[S_pack, 1, 1]``.

        o : vTensor
            Preallocated output tensor that will be filled in-place by the
            top-k implementation. Its shape and semantics are defined by
            the upstream contract and the implementation.

        ctx : Context
            Execution context providing:

            - ``dense_kv_indptr`` and ``sparse_kv_indptr``: segment
              boundaries in the packed axis,
            - ``dense_kv_indices``: indices into underlying storage, and
            - scalar configuration such as ``batch_size``, ``num_kv_heads``,
              ``topk_val``, ``page_reserved_bos``, and ``page_reserved_eos``.

        Raises
        ------
        AssertionError
            If types, ranks, shapes, formats, or devices are incompatible,
            or if no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # ---- type checks ----
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"

        # ---- rank checks ----
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert o.dim() == 3, (
            f"{prefix}expected o to be 3D, "
            f"got ndim={o.dim()} shape={tuple(o.shape)}"
        )

        # ---- shape checks for x ----
        # x is expected to carry per-page scalars at dims (1,2)
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )

        # ---- implementation availability ----
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x._format={x_fmt}. "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.topk_type = getattr(ctx, "topk_type", "naive")
        impl_variants = self._impl_map[x_fmt]
        assert self.topk_type in impl_variants, (
            f"{prefix}no topk implementation for topk_type='{self.topk_type}'. "
            f"Available: {list(impl_variants.keys())}"
        )
        self.impl = impl_variants[self.topk_type]

        # ---- optional sanity checks on `o` ----
        # We only assert device consistency and leave exact (S_pack, D0, D1)
        # to the upstream contract and the implementation.
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, o: torch.Tensor, ctx: Context) -> torch.Tensor:
        r"""
        Run the selected piecewise top-k implementation and return ``o``.

        This method invokes the implementation on the packed input ``x``
        using segment boundaries and configuration provided by
        :class:`Context`. For each segment in the packed axis, the kernel:

        - identifies the BOS and EOS reserved pages using
          ``page_reserved_bos`` and ``page_reserved_eos``,
        - selects an additional ``topk_val`` pages by score, and
        - writes the resulting indices/values into ``o``.

        Expected implementation signature::

            impl(
                x,
                dense_kv_indptr,
                sparse_kv_indptr,
                dense_kv_indices,
                o,
                batch_heads,
                topk_val,
                page_reserved_bos,
                page_reserved_eos,
                max_num_pages_per_request,
            )

        Parameters
        ----------
        x : torch.Tensor
            Input tensor carrying per-page scalar scores, compatible with the
            shape validated in :meth:`profile`.

        o : torch.Tensor
            Preallocated output tensor; will be filled in-place by the
            implementation.

        ctx : Context
            Execution context providing all auxiliary arrays and scalar
            parameters needed for segment-wise top-k selection.

        Returns
        -------
        torch.Tensor
            The tensor ``o`` after it has been filled by the top-k
            implementation.

        Raises
        ------
        AssertionError
            If :meth:`profile` has not been called and no implementation is
            available.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        if self.topk_type == "sglang":
            # topk_output_sglang: (x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices, ...)
            mapping_mode = getattr(ctx, 'topk_mapping_mode', 0)
            mapping_power = getattr(ctx, 'topk_mapping_power', 0.5)
            mapping_lut = getattr(ctx, 'topk_mapping_lut', None)
            mapping_quantiles = getattr(ctx, 'topk_mapping_quantiles', None)
            mapping_noscale = getattr(ctx, 'topk_mapping_noscale', False)
            # UNSET sentinel is not a valid torch.Tensor — coerce to None
            if mapping_lut is UNSET:
                mapping_lut = None
            if mapping_quantiles is UNSET:
                mapping_quantiles = None
            self.impl(
                x,
                ctx.dense_kv_indptr,
                ctx.sparse_kv_indptr,
                ctx.dense_kv_indices,
                o,
                ctx.batch_size * ctx.num_kv_heads,
                ctx.topk_val,
                ctx.page_reserved_bos,
                ctx.page_reserved_eos,
                ctx.max_num_pages_per_request,
                mapping_mode,
                mapping_power,
                mapping_lut,
                mapping_quantiles,
                mapping_noscale,
            )
        else:
            # topk_output (naive): (x, dense_kv_indptr, dense_kv_indices, sparse_kv_indptr, sparse_kv_indices, ...)
            self.impl(
                x,
                ctx.dense_kv_indptr,
                ctx.dense_kv_indices,
                ctx.sparse_kv_indptr,
                o,
                ctx.batch_size * ctx.num_kv_heads,
                ctx.topk_val,
                ctx.page_reserved_bos,
                ctx.page_reserved_eos,
                ctx.max_num_pages_per_request,
            )

        # Optional histogram profiling (default disabled, no overhead when off).
        # Skip entirely during CUDA graph capture — allocations and D2H copies
        # are not permitted while a stream is being captured.
        if (
            getattr(ctx, 'topk_histogram_enabled', False)
            and self.topk_type == "sglang"
            and not torch.cuda.is_current_stream_capturing()
        ):
            eff_bs = ctx.batch_size * ctx.num_kv_heads
            self.last_histograms = torch.zeros(eff_bs, 256, dtype=torch.int32, device=x.device)
            topk_profile_histogram(
                x,
                ctx.dense_kv_indptr,
                self.last_histograms,
                eff_bs,
                ctx.page_reserved_bos,
                ctx.page_reserved_eos,
                mapping_mode,
                mapping_power,
                mapping_lut,
                mapping_quantiles,
                mapping_noscale,
                ctx.topk_val,
            )
            # Accumulate histograms for offline calibration
            _calibration_histograms.append(self.last_histograms.cpu().clone())

        return o
