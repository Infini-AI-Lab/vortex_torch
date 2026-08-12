from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch

from ..abs import ContextBase
from ..utils import resolve_dtype

class vFlow(ABC):
    r"""
    Base class for flow-style sparse attention modules.

    This abstraction is conceptually similar to :class:`torch.nn.Module`,
    but specialized for **sparse attention flows** that:

    - maintain a structured key/value cache,
    - define how to **index** into sparse pages (top-k style routing), and
    - define how to **update** / **summarize** that cache as new pages arrive.

    Query tensor
    ------------
    The query tensor ``q`` passed to :meth:`forward_indexer` has logical
    shape

    .. math::

        q \in \mathbb{R}^{B \times H_q \times D},

    where

    - :math:`B` is a batch-like axis (commonly ``batch_size * num_heads``),
    - :math:`H_q` is the number of query positions per batch/head, and
    - :math:`D` is the head dimension.

    In practice ``q`` is typically stored in :class:`torch.bfloat16`.

    Sparse index tensor
    -------------------
    The sparse index tensor ``o`` produced by :meth:`forward_indexer` has
    logical shape

    .. math::

        o \in \mathbb{R}^{S_{\text{sparse}} \times 1 \times 1},

    and stores integer page indices. The packed sparse length is

    .. math::

        S_{\text{sparse}}
        = \sum_{i=0}^{B-1} S_{\text{sparse}, i},

    where for each request :math:`i` with :math:`S_i` candidate pages,

    .. math::

        S_{\text{sparse}, i}
        = \min\Bigl(
            S_i,\;
            \text{topk_val}
            + \text{page_reserved_bos}
            + \text{page_reserved_eos}
        \Bigr).

    Here:

    - ``topk_val`` is the number of pages selected by the indexer,
    - ``page_reserved_bos`` is the number of always-kept pages at the
      beginning (BOS region),
    - ``page_reserved_eos`` is the number of always-kept pages at the
      end (EOS region),

    and these values are typically provided by the runtime context.

    Cache tensors: two logical views
    --------------------------------
    Each cache entry ``cache[key]`` (including the standard keys
    ``"k"`` and ``"v"`` plus any extra entries declared by
    :meth:`create_cache`) is a rank-3 tensor that is **viewed in two
    different logical layouts**:

    1. **Indexer view (page-packed)** — used in :meth:`forward_indexer`:

       .. math::

           \text{cache[key]} \sim
           \mathbb{R}^{S \times r \times c},

       

       :math:`(r, c)` is the per-key inner shape declared via
       :meth:`create_cache` or implicitly for ``"k"``/``"v"``.

        Here :math:`S` is the leading page axis. Internally it is a packed
        axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
        concatenating the pages from all requests. As a user, you can simply
        think of :math:`S` as "the number of pages for this request"; the
        vFlow kernels and :class:`ContextBase` will take care of mapping
        between per-request page counts and the packed layout automatically.
    
    2. **Cache-update view (batch-major)** — used in :meth:`forward_cache`:

       .. math::

           \text{cache[key]} \sim
           \mathbb{R}^{B \times r \times c}.

       The leading axis is the request/batch index :math:`B`, while
       the inner shape :math:`(r, c)` is the same as in the indexer view.

    The runtime (via :class:`ContextBase`) is responsible for mapping
    between these two views using indptr arrays and layout metadata.

    Cache metadata
    --------------
    Subclasses declare only **extra** cache tensors via
    :meth:`create_cache`, e.g.::

        {
            "centroids": (1, head_dim),
            "my_aux_tensor": (block_size, head_dim),
            ...
        }

    The helper :meth:`get_cache_meta_info` then injects the standard
    entries:

    .. math::

        \text{k} &: (\text{block_size}, \text{head_dim}), \\
        \text{v} &: (\text{block_size}, \text{head_dim}),

    so subclasses must not add ``"k"`` or ``"v"`` themselves.

    Token ratio
    -----------
    :meth:`get_token_ratio` computes a simple proxy for how much cache
    storage is used (per head) relative to one ``k``/``v`` page:

    .. math::

        \text{token_ratio}
        = \sum_{\text{key}}
          \frac{r_{\text{key}} \cdot c_{\text{key}}}
               {\text{block_size} \cdot \text{head_dim}}.

    This ignores the leading dimension (whether :math:`B` or
    :math:`S`) and compares only inner shapes to the
    baseline ``(block_size, head_dim)``.

    Subclass responsibilities
    -------------------------
    Concrete flows must implement:

    - :meth:`forward_indexer(q, o, cache, ctx)`:
      compute sparse page indices (or routing scores) from queries,
      using cache in the :math:`S` view.

    - :meth:`forward_cache(cache, loc, ctx)`:
      update cache tensors using the :math:`B`-major view and positional
      metadata.

    - :meth:`create_cache(block_size, head_dim)`:
      declare inner shapes :math:`(r, c)` for all extra cache tensors
      (excluding ``"k"`` and ``"v"``).
    """

    def __init__(self):
        super().__init__()

        self.block_size = None
        self.head_dim = None
        self.kv_cache_dtype = None
        self.q_data_type = None
        self.intermediate_dtype = None
        self.cache_meta_info = None
        self.token_ratio = None

    # ------------------------------------------------------------------ #
    # abstract API to be implemented by concrete flows
    # ------------------------------------------------------------------ #
    @abstractmethod
    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: "ContextBase",
    ):
        r"""
        Compute sparse page indices (or equivalent routing information)
        from queries and cache.

        Canonical shapes
        ----------------
        - ``q`` (queries):

          .. math::

              q \in \mathbb{R}^{B \times H_q \times D},

          typically stored in :class:`torch.bfloat16`.

        - ``o`` (sparse indices):

          .. math::

              o \in \mathbb{R}^{S_{\text{sparse}} \times 1 \times 1},

          integer dtype (e.g. :class:`torch.int32` or
          :class:`torch.int64`). The packed length
          :math:`S_{\text{sparse}}` is defined in the class docstring.

        - ``cache[key]`` (indexer view):

          .. math::

              \text{cache[key]}
              \sim \mathbb{R}^{S \times r \times c},

          :math:`(r, c)` are the per-key inner dimensions obtained from
          :meth:`get_cache_meta_info`.

        - ``ctx``:

          An instance of :class:`ContextBase` carrying page layout,
          indptr arrays, and configuration such as ``topk_val``,
          ``page_reserved_bos``, and ``page_reserved_eos``.

        Contract
        --------
        Implementations should:

        - interpret ``cache`` in the :math:`S` view,
        - use ``q`` and relevant cache tensors to score/select pages,
        - respect per-request bounds derived from ``ctx``,
        - write the resulting sparse indices (or routing representation)
          into ``o`` in-place.

        The exact semantics of the integers stored in ``o`` (e.g.
        absolute page indices vs. offsets) are defined by the runtime
        convention and must be consistent with downstream kernels.
        """
        pass

    @abstractmethod
    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: "ContextBase",
    ):
        r"""
        Update or recompute cache tensors in the batch-major view.

        Canonical shapes
        ----------------
        - ``cache[key]`` (cache-update view):

          .. math::

              \text{cache[key]}
              \sim \mathbb{R}^{B \times r \times c},

          where :math:`B` is the number of requests and :math:`(r, c)`
          are the same inner dimensions as in the indexer view.

        - ``loc``:

          Positional / layout metadata (for example, page indices or
          token positions) used to decide how to aggregate over pages or
          tokens when producing per-request summaries.

        - ``ctx``:

          Execution context (same instance type as in
          :meth:`forward_indexer`), carrying runtime parameters and
          layout information.

        Contract
        --------
        Typical operations include recomputing per-request summaries
        such as:

        - averaging or pooling ``cache["k"]`` into a tensor
          ``cache["centroids"]`` of shape ``[B, r, c]``,
        - maintaining auxiliary statistics needed by the indexer stage.

        Implementations may update any entries in ``cache`` in-place, as
        long as they respect the shapes announced by
        :meth:`get_cache_meta_info`.
        """
        pass

    @abstractmethod
    def create_cache(
        self,
        block_size: int,
        head_dim: int,
    ) -> Dict[str, Tuple[Tuple[int, int]]]:
        r"""
        Declare inner shapes for non-``"k"`` / non-``"v"`` cache tensors.

        This method **does not allocate** tensors. It only declares the
        per-key inner dimensions :math:`(r, c)`; the runtime will attach
        the appropriate leading axis (:math:`B` or :math:`S`)
        depending on whether the cache is used in :meth:`forward_cache`
        or :meth:`forward_indexer`.

        Parameters
        ----------
        block_size : int
            Number of tokens per block (the inner length of a ``"k"`` / ``"v"``
            cache slot). For the standard ``"k"`` and ``"v"`` entries this is
            the first inner dimension.

        head_dim : int
            Head dimension. For the standard ``"k"`` and ``"v"`` entries,
            this will be the second dimension.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            A mapping from cache tensor names (excluding ``"k"`` and
            ``"v"``) to inner shapes ``(r, c)``. For example::

                {
                    "centroids": (1, head_dim),
                }

        Notes
        -----
        The keys ``"k"`` and ``"v"`` are reserved and **must not** be
        present in the returned dictionary. They are added automatically
        by :meth:`get_cache_meta_info` with inner shape
        ``(block_size, head_dim)``.
        """
        pass

    # ------------------------------------------------------------------ #
    # helper API used by the runtime to allocate / account cache
    # ------------------------------------------------------------------ #
    def get_cache_meta_info(
        self
    ) -> Dict[str, Tuple[Tuple[int, int], torch.dtype]]:
        
        return self.cache_meta_info

    def get_token_ratio(
        self, 
        ) -> float:
        
        return self.token_ratio

    def get_aux_token_ratio(self) -> float:
        """Token ratio counting only HBM-resident cache fields (host-KV mode)."""
        return self.aux_token_ratio

    def initialize(self,
        block_size: int,
        head_dim: int,
        kv_cache_dtype: Union[torch.dtype, str],
        q_data_type: Union[torch.dtype, str],
        intermediate_dtype: Union[torch.dtype, str] = torch.bfloat16,
        ):
        r"""
        Optional initialization method called by the runtime after cache
        tensors are allocated.

        This can be used to set up any internal state or invariants needed
        by the flow. By default this is a no-op, but concrete flows can
        override it if needed.

        Parameters
        ----------
        block_size : int
            Number of tokens per block.
        head_dim : int
            Head dimension.
        kv_cache_dtype : torch.dtype or str
            Data type for key/value caches. Accepts a :class:`torch.dtype`
            or one of the canonical strings in
            :data:`vortex_torch.utils.DTYPE_STR_TO_TORCH`
            (e.g. ``"bfloat16"``, ``"fp8_e5m2"``).
        q_data_type : torch.dtype or str
            Data type for query tensor. Same string convention as
            ``kv_cache_dtype``.
        intermediate_dtype : torch.dtype or str
            Data type for intermediate tensors. Defaults to ``torch.bfloat16``.
            Same string convention as ``kv_cache_dtype``.
        """

        self.block_size = block_size
        self.head_dim = head_dim
        self.kv_cache_dtype = resolve_dtype(kv_cache_dtype)
        self.q_data_type = resolve_dtype(q_data_type)
        self.intermediate_dtype = resolve_dtype(intermediate_dtype)
        self.token_ratio = 0.0
        raw_cache_meta_info = self.create_cache(block_size, head_dim)
        assert "k" not in raw_cache_meta_info, "create_cache must not declare 'k' key"
        assert "v" not in raw_cache_meta_info, "create_cache must not declare 'v' key"
        
        raw_cache_meta_info["k"] = (block_size, head_dim)
        raw_cache_meta_info["v"] = (block_size, head_dim)

        total_bytes = 0
        aux_bytes = 0
        # convert to a format that maps key -> ((r, c), dtype) for easier access during indexing and cache updates
        self.cache_meta_info = {}
        for key, (r, c) in raw_cache_meta_info.items():
            if key in ["k", "v"]:
                dtype = self.kv_cache_dtype
            else:
                dtype = self.intermediate_dtype  # default dtype for auxiliary tensors; can be customized as needed
            nbytes = r * c * torch._utils._element_size(dtype)
            total_bytes += nbytes
            if key not in ("k", "v"):
                aux_bytes += nbytes
            self.cache_meta_info[key] = ((r, c), dtype)

        base_bytes = block_size * head_dim * torch._utils._element_size(self.kv_cache_dtype)
        self.token_ratio = total_bytes / base_bytes
        #: Same ratio counting only the fields that stay in HBM when KV is hosted
        #: in pinned host memory (``vortex_host_kv_gb``): the auxiliary
        #: centroid/envelope/Save fields, without K and V. Used to size the *device*
        #: token budget in that mode — using ``token_ratio`` there would reserve HBM
        #: for a KV cache that is not on the device.
        self.aux_token_ratio = aux_bytes / base_bytes
