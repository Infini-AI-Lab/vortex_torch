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
        #: ``{name: (r, c)}`` for the REQUEST-bound domain, or ``{}``. Populated by
        #: :meth:`initialize` from :meth:`create_request_cache`.
        self.request_cache_meta_info = {}

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

    def create_request_cache(
        self,
        block_size: int,
        head_dim: int,
    ) -> Dict[str, Tuple[int, int]]:
        r"""
        Declare inner shapes for the **request-bound** cache domain. Optional; default ``{}``.

        A field declared here is allocated ``[n_slots, r, c]`` and addressed through a device-side
        ``block_id -> slot`` map (``FORMAT.SLOTTED``), rather than ``[num_blocks, r, c]`` addressed
        by block id. Use it for state that belongs to a **request in flight** rather than to a
        stored page. The motivating case is INT4 KV's bf16 staging area: a block's quantization
        scale is a reduction over the block, so a block still being appended to cannot be stored
        quantized, and must be held in full precision until it completes.

        The distinction is *lifetime*, not addressing -- keys are still block ids, because
        ``set_kv_buffer`` is passed no request id and recovering one needs a host sync that cudagraph
        capture forbids.

        The reason to declare it here rather than in :meth:`create_cache` is memory: a request-domain
        field is sized by CONCURRENCY, so its footprint is constant in context length, whereas the
        same state as a page field costs bytes per block. Measured on INT4: a per-block bf16 mirror
        came to 21632 B/block against bf16's own 16896 -- a 1.28x *regression*, i.e. it gave back
        more than quantization saved.

        For the same reason these fields are **excluded from** ``token_ratio``: that ratio is a
        per-token budget, and charging a constant-size buffer per token reserves HBM that grows with
        context for something that does not.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            ``{name: (r, c)}``. ``"k"`` / ``"v"`` are reserved here as well.
        """
        return {}

    # ------------------------------------------------------------------ #
    # helper API used by the runtime to allocate / account cache
    # ------------------------------------------------------------------ #
    def get_request_cache_meta_info(self) -> Dict[str, Tuple[int, int]]:
        """``{name: (r, c)}`` for the request domain; ``{}`` when the flow declares none."""
        return self.request_cache_meta_info

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
        kv_int4: bool = False,
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
        kv_int4 : bool
            Store K/V as packed INT4 instead of ``kv_cache_dtype``. Applied here, at the
            meta-declaration level, rather than by each flow: the transform is entirely on the
            declared shapes and dtypes (K/V become ``(block_size, head_dim // 2)`` uint8, two fp32
            scale fields appear, and a bf16 staging area is added to the request domain), so **any**
            flow gets INT4 without being edited. A flow that wants to score the indexer from
            pre-quant K can override :meth:`create_request_cache` to keep what it needs.
        """

        self.block_size = block_size
        self.head_dim = head_dim
        self.kv_cache_dtype = resolve_dtype(kv_cache_dtype)
        self.q_data_type = resolve_dtype(q_data_type)
        self.intermediate_dtype = resolve_dtype(intermediate_dtype)
        self.kv_int4 = bool(kv_int4)
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

        int4_request_meta = {}
        if self.kv_int4:
            # Overwrite k/v with the packed form and add the scales. Deliberately AFTER the loop
            # above, so ``total_bytes`` / ``aux_bytes`` are recomputed below rather than accumulated
            # from a mix of the two layouts -- a ratio built from bf16 K/V plus INT4 scales would
            # over-reserve, silently and by roughly the compression factor.
            from ..engine.sgl.int4_store import int4_cache_meta, int4_request_cache_meta
            self.cache_meta_info.update(int4_cache_meta(block_size, head_dim))
            int4_request_meta = int4_request_cache_meta(block_size, head_dim)
            total_bytes = sum(r * c * torch._utils._element_size(dt)
                              for (r, c), dt in self.cache_meta_info.values())
            aux_bytes = sum(r * c * torch._utils._element_size(dt)
                            for k, ((r, c), dt) in self.cache_meta_info.items()
                            if k not in ("k", "v"))

        # Request-domain fields are declared but NOT folded into the ratios below: they are sized by
        # concurrency, so charging them per token would reserve HBM growing with context for a
        # buffer that is constant. The runtime allocates them separately, from this dict.
        self.request_cache_meta_info = dict(self.create_request_cache(block_size, head_dim))
        # INT4's staging area is added on the same terms as a flow-declared field. The flow's own
        # declarations win a name collision by being applied second below, but the assert following
        # would already have caught a real clash.
        for name, shape in int4_request_meta.items():
            self.request_cache_meta_info.setdefault(name, shape)
        for reserved in ("k", "v"):
            assert reserved not in self.request_cache_meta_info, (
                f"create_request_cache must not declare {reserved!r}"
            )
        clash = set(self.request_cache_meta_info) & set(self.cache_meta_info)
        assert not clash, (
            f"create_request_cache and create_cache both declare {sorted(clash)}; the two domains "
            f"share one cache dict, so a duplicate name would make the field's ADDRESSING depend on "
            f"which declaration the runtime happened to read last"
        )

        # ``base_bytes`` stays anchored to ``kv_cache_dtype`` -- the *nominal* per-token KV -- and is
        # NOT recomputed for INT4. That is what makes ``token_ratio`` fall below 1 under INT4, which
        # is how the pool learns it can hold more tokens. Anchoring it to the packed size instead
        # would report a ratio of ~1 and give back the entire capacity win.
        base_bytes = block_size * head_dim * torch._utils._element_size(self.kv_cache_dtype)
        self.token_ratio = total_bytes / base_bytes
        #: Same ratio counting only the fields that stay in HBM when KV is hosted
        #: in pinned host memory (``vortex_host_kv_gb``): the auxiliary
        #: centroid/envelope/Save fields, without K and V. Used to size the *device*
        #: token budget in that mode — using ``token_ratio`` there would reserve HBM
        #: for a KV cache that is not on the device.
        self.aux_token_ratio = aux_bytes / base_bytes
