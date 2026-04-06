from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple
from ..abs import ContextBase

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
            "my_aux_tensor": (page_size, head_dim),
            ...
        }

    The helper :meth:`get_cache_meta_info` then injects the standard
    entries:

    .. math::

        \text{k} &: (\text{page_size}, \text{head_dim}), \\
        \text{v} &: (\text{page_size}, \text{head_dim}),

    so subclasses must not add ``"k"`` or ``"v"`` themselves.

    Token ratio
    -----------
    :meth:`get_token_ratio` computes a simple proxy for how much cache
    storage is used (per head) relative to one ``k``/``v`` page:

    .. math::

        \text{token_ratio}
        = \sum_{\text{key}}
          \frac{r_{\text{key}} \cdot c_{\text{key}}}
               {\text{page_size} \cdot \text{head_dim}}.

    This ignores the leading dimension (whether :math:`B` or
    :math:`S`) and compares only inner shapes to the
    baseline ``(page_size, head_dim)``.

    Subclass responsibilities
    -------------------------
    Concrete flows must implement:

    - :meth:`forward_indexer(q, o, cache, ctx)`:
      compute sparse page indices (or routing scores) from queries,
      using cache in the :math:`S` view.

    - :meth:`forward_cache(cache, loc, ctx)`:
      update cache tensors using the :math:`B`-major view and positional
      metadata.

    - :meth:`create_cache(page_size, head_dim)`:
      declare inner shapes :math:`(r, c)` for all extra cache tensors
      (excluding ``"k"`` and ``"v"``).
    """

    def __init__(self):
        super().__init__()

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
        page_size: int,
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
        page_size : int
            Number of tokens per page. For the standard ``"k"`` and
            ``"v"`` entries, this will be the first dimension.

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
        ``(page_size, head_dim)``.
        """
        pass

    # ------------------------------------------------------------------ #
    # helper API used by the runtime to allocate / account cache
    # ------------------------------------------------------------------ #
    def get_cache_meta_info(
        self,
        page_size: int,
        head_dim: int,
    ) -> Dict[str, Tuple[Tuple[int, int]]]:
        r"""
        Get full cache inner-shape metadata, including ``"k"`` and ``"v"``.

        This wraps :meth:`create_cache` and injects standard inner shapes
        for key/value tensors. The resulting dictionary includes:

        - all user-defined entries from :meth:`create_cache`, and
        - two additional entries:

          - ``"k"`` : ``(page_size, head_dim)``
          - ``"v"`` : ``(page_size, head_dim)``

        Parameters
        ----------
        page_size : int
            Number of tokens per page.

        head_dim : int
            Head dimension.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Mapping from cache tensor names to inner shapes ``(r, c)``.
            The runtime will later prepend either a batch axis ``B`` or a
            packed-page axis ``S`` when materializing the tensors.

        Raises
        ------
        AssertionError
            If :meth:`create_cache` tries to define entries for the
            reserved keys ``"k"`` or ``"v"``.
        """
        cache_meta_info = self.create_cache(page_size, head_dim)
        assert "k" not in cache_meta_info.keys()
        assert "v" not in cache_meta_info.keys()
        cache_meta_info["k"] = (page_size, head_dim)
        cache_meta_info["v"] = (page_size, head_dim)
        return cache_meta_info

    def get_token_ratio(self, page_size: int, head_dim: int) -> float:
        r"""
        Compute the relative cache size in "tokens" compared to a k/v page.

        Using :meth:`get_cache_meta_info`, this computes a simple ratio
        that measures how many *equivalent tokens* the entire cache
        consumes per head, relative to a baseline page of shape
        ``(page_size, head_dim)``.

        Let :math:`\mathcal{C}` be the set of cache tensors with inner
        shapes :math:`(r_{\text{key}}, c_{\text{key}})`. This method
        returns

        .. math::

            \text{token_ratio}
            = \sum_{\text{key} \in \mathcal{C}}
              \frac{r_{\text{key}} \cdot c_{\text{key}}}
                   {\text{page_size} \cdot \text{head_dim}}.

        The leading dimension (:math:`B` or :math:`S`) is
        not included in this ratio on purpose; it is a per-page
        normalization.

        Parameters
        ----------
        page_size : int
            Number of tokens per page.

        head_dim : int
            Head dimension.

        Returns
        -------
        float
            Total cache size expressed in units of
            ``page_size * head_dim`` (including the standard ``"k"`` and
            ``"v"`` entries).
        """
        token_ratio = 0.0
        for (_, cache_shape) in self.get_cache_meta_info(page_size, head_dim).items():
            token_ratio += (cache_shape[0] * cache_shape[1]) / (page_size * head_dim)
        return token_ratio

    def run_indexer_virtual(self, group_size: int, page_size: int, head_dim: int):
        
        from ..indexer import Context as IContext
        from ..abs import as_vtensor, FORMAT
        
        ctx = IContext()
        
        ctx.group_size = group_size
        ctx.num_kv_heads = 1
        ctx.num_qo_heads = group_size
        ctx.head_dim = head_dim
        ctx.page_size = page_size
        ctx.max_num_pages = 0
        ctx.max_num_pages_per_request = 0
        ctx.topk_type = "naive"
        
        device = "cuda"
        dtype = torch.bfloat16
        
        with torch.no_grad():
                # Dummy placeholders: used only for kernel / graph warm-up
                q_dummy = as_vtensor(torch.empty((1, group_size, head_dim), device=device, dtype=dtype), FORMAT.BATCHED)
                o_dummy = as_vtensor(torch.empty((0, 1, 1), device=device, dtype=dtype), FORMAT.RAGGED)
                cache_meta_info = self.get_cache_meta_info(page_size, head_dim)
                
                cache_dummy = {
                        cache_name:  as_vtensor(torch.zeros(
                                (0, cache_shape[0], cache_shape[1]),
                                dtype=dtype,
                                device=device,
                            ), FORMAT.PAGED)
                        
                        for (cache_name, cache_shape) in cache_meta_info.items()
                    }
                
                self.forward_indexer(q_dummy, o_dummy, cache_dummy, ctx=ctx)

        del q_dummy
        del o_dummy
        del cache_dummy
    
        return ctx._aux_total_flops
    