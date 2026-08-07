import torch
from typing import Dict, Optional

from .flow import vFlow
from ..indexer import topK, GeMV, Softmax, Max, Sum, GeMM, Maximum, Multiply, Add, L2Norm, Save, Load
from ..cache import Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm, Fill as CFill
from ..abs import ContextBase
from .registry import register

# The ops are not reusable, even if they have the same semantic meaning. Internally, they will initialize different memory buffer.
# For example, in Quest attention, we need to define two multiply operators.

# In forward indexer, q can be viewed as [1, H_q, D] or [B, H_q, D] (B=1) and cache["xxx"] can be viewed as [S, r, c] (r, c defined in create_cache) logically.
# In forward cache, the cache["xxx"] is viewed as [B, r, c]  (r, c defined in create_cache) logically.
# In forward cache, each page is computed only once if page_id appears in loc. During the entire computation, each page id will appear in loc only once.
# Thus, all the tensors have 3 dimensions. Reduce operators (Mean, Max, Min, etc) will always keep the dims.
# Tips 1: GeMM(x, y) = yx^t, which might be different from typical definitions.
# Tips 2: Except cache["k"], cache["v"] can also be used in forward_cache to collect information.

@register("block_sparse_attention")
class BlockSparseAttention(vFlow):
    r"""
    Block-sparse attention flow with centroid-based routing.

    This flow implements a simple **block-sparse routing** strategy
    inspired by the block-top-k routing used in Kinetics
    :cite:`sadhukhan2025kinetics` (arXiv:2506.05333). It maintains a
    per-request centroid over keys and uses query–centroid similarity to
    select a sparse set of pages.

    High-level behavior
    -------------------
    - During :meth:`forward_cache`, the flow computes a **centroid**
      vector for each request from its key cache ``cache["k"]`` and
      stores the result in ``cache["centroids"]`` with shape

      .. math::

          \text{cache["centroids"]} \in \mathbb{R}^{B \times 1 \times D},

      where :math:`B` is the number of requests and :math:`D` is the
      head dimension.

    - During :meth:`forward_indexer`, the flow:
      
      1. Averages query tokens per request to obtain a single
         **query summary** per request,
      2. Applies a generalized matrix–vector multiplication
         :class:`GeMV` between the query summaries and the cached
         centroids to obtain a scalar **score** for each (request, page),
      3. Uses :class:`topK` to convert these scores into sparse page
         indices ``o`` of shape

         .. math::

             o \in \mathbb{R}^{S} \times 1 \times 1},

         Here :math:`S` is the leading page axis. Internally it is a packed
         axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
         concatenating the pages from all requests. As a user, you can simply
         think of :math:`S` as "the number of pages for this request"; the
         vFlow kernels and :class:`ContextBase` will take care of mapping
         between per-request page counts and the packed layout automatically.

    Cache layout
    ------------
    This flow declares a single extra cache tensor via
    :meth:`create_cache`:

    .. code-block:: python

        {
            "centroids": (1, head_dim)
        }

    The runtime then also allocates ``"k"`` and ``"v"`` with inner shapes
    ``(page_size, head_dim)``. As per the :class:`vFlow` contract,
    each cache tensor has two logical views:

    - In :meth:`forward_indexer` (page-packed view):

      .. math::

          \text{cache["centroids"]} \sim
          \mathbb{R}^{S} \times 1 \times D},

    - In :meth:`forward_cache` (batch-major view):

      .. math::

          \text{cache["centroids"]} \sim
          \mathbb{R}^{B \times 1 \times D}.

    References
    ----------
    .. rubric:: Bibliography

    .. [sadhukhan2025kinetics]
       Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou,
       Emma Strubell, Beidi Chen.
       *Kinetics: Rethinking Test-Time Scaling Laws*.
       arXiv:2506.05333, 2025.
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemv = GeMV()
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Compute sparse page indices from queries and cached centroids.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor with shape ``[B, H_q, D]`` (typically
            :class:`torch.bfloat16`), where :math:`B` is the batch–head
            axis, :math:`H_q` is the number of query positions per
            request, and :math:`D` is the head dimension.

        o : torch.Tensor
            Output tensor for sparse page indices with shape
            ``[S_sparse, 1, 1]`` and integer dtype. It is filled
            in-place by :class:`topK` according to the scores computed
            by :class:`GeMV`.

        cache : Dict[str, torch.Tensor]
            Cache dictionary in the **indexer view**, where:

            - ``cache["k"]`` and ``cache["v"]`` are page-packed key/value
              tensors,
            - ``cache["centroids"]`` is interpreted as
              ``[S, 1, D]`` (page-packed centroids).

        ctx : ContextBase
            Runtime context carrying page layout, top-k configuration
            (``topk_val``, ``page_reserved_bos``, ``page_reserved_eos``),
            and other metadata.

        Notes
        -----
        The implementation:

        1. Computes a per-request query summary

           .. math::

              q_{\mathrm{mean}}[b, 0, :]
              = \frac{1}{H_q} \sum_{h=0}^{H_q-1} q[b, h, :],

        2. Applies :class:`GeMV` between ``q_mean`` and
           ``cache["centroids"]`` to obtain scalar scores per page,
        3. Uses :class:`topK` to select a sparse set of pages per request
           and write the corresponding indices into ``o`` in the packed
           sparse layout.
        """
        q_mean = q.mean(dim=1, keepdim=True)
        score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Update cache centroids from the key cache in batch-major view.

        Parameters
        ----------
        cache : Dict[str, torch.Tensor]
            Cache dictionary in the **batch-major view**, where:

            - ``cache["k"]`` has shape ``[B, page_size, D]``,
            - ``cache["centroids"]`` has shape ``[B, 1, D]``.

        loc : torch.Tensor
            Positional or layout metadata used by :class:`CMean` to
            aggregate keys into centroids (e.g. page boundaries or valid
            token masks).

        ctx : ContextBase
            Runtime context forwarded to the reduction op.

        Notes
        -----
        This method calls :class:`CMean` with ``dim=1`` so that for each
        request :math:`b` it computes a mean over the key axis and writes
        it to ``cache["centroids"][b, 0, :]``. The exact handling of
        padding or invalid positions is controlled by ``loc`` and the
        backend implementation of :class:`CMean`.
        """
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Parameters
        ----------
        page_size : int
            Number of tokens per page (unused here but part of the
            generic vFlow contract).

        head_dim : int
            Head dimension :math:`D`. Used as the second dimension of
            the centroid tensor.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Mapping from cache tensor names to inner shapes ``(r, c)``.
            This flow defines a single extra tensor:

            - ``"centroids"`` with inner shape ``(1, head_dim)``, which
              becomes

              - ``[S, 1, head_dim]`` in :meth:`forward_indexer`,
              - ``[B, 1, head_dim]`` in :meth:`forward_cache`.
        """
        return {
            "centroids": (1, head_dim),
        }


@register("gqa_block_sparse_attention")
class GQABlockSparseAttention(vFlow):
    r"""
    Grouped-query block-sparse attention flow.

    This flow uses a GQA-style block-sparse routing: queries are grouped,
    scored against per-request centroids, normalized with a softmax, then
    aggregated across groups before a top-k over pages is applied.

    - Queries ``q`` have shape ``[B, H_q, D]``.
    - Centroids cache ``cache["centroids"]`` has inner shape
      ``(1, head_dim)`` and is viewed as:

      - ``[S, 1, D]`` in :meth:`forward_indexer`,
      - ``[B, 1, D]`` in :meth:`forward_cache`.
      Here :math:`S` is the leading page axis. Internally it is a packed
      axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
      concatenating the pages from all requests. As a user, you can simply
      think of :math:`S` as "the number of pages for this request"; the
      vFlow kernels and :class:`ContextBase` will take care of mapping
      between per-request page counts and the packed layout automatically.
      
    For a design similar in spirit to grouped-query block sparsity, see
    the GQA sparse attention formulation in:

    - https://arxiv.org/abs/2502.11089
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2)
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Compute sparse page indices from grouped-query scores.

        Pipeline
        --------
        1. Apply :class:`GeMM` between queries and centroids (o = yx^t):

           - ``q``: ``[B, H_q, D]``
           - ``cache["centroids"]`` (indexer view): ``[S, 1, D]``
           - ``score``: ``[S, 1, H_q]`` (logical ``[S, Ny, Nx]``)

        2. Apply in-place softmax over the leading (page) axis with a
           scaling factor ``scale``:

           .. math::
              \mathrm{softmax}(x \cdot \mathrm{scale})

        3. Aggregate over the query-group dimension with :class:`Max`
           (``dim=2``), yielding a single scalar score per page.

        4. Use :class:`topK` on the aggregated scores to write packed
           sparse page indices into ``o`` with shape
           ``[S_sparse, 1, 1]``.
        """
        score = self.gemm(q, cache["centroids"], ctx=ctx)
        self.softmax(score, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Update per-request centroids from the key cache.

        - ``cache["k"]``: ``[B, page_size, D]`` (batch-major view)
        - ``cache["centroids"]``: ``[B, 1, D]``

        The :class:`CMean` operator with ``dim=1`` computes a mean over
        the key axis (optionally masked/structured via ``loc``) and
        writes the result into ``cache["centroids"]`` in-place.
        """
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Parameters
        ----------
        page_size : int
            Number of tokens per page (not used directly here).

        head_dim : int
            Head dimension ``D`` for centroids.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Custom cache metadata. This flow defines:

            - ``"centroids"``: inner shape ``(1, head_dim)``.
        """
        return {
            "centroids": (1, head_dim),
        }



@register("gqa_quest_sparse_attention")
class GQAQuestSparseAttention(vFlow):
    r"""
    GQA-style QUEST sparse attention flow.

    This flow uses **query–envelope matching** similar to QUEST sparse
    attention (see https://arxiv.org/abs/2406.10774). For each request,
    it maintains per-page **max** and **min** envelopes of keys and uses
    them to compute a conservative upper bound on query–key similarity.

    Shapes
    ------
    - Queries ``q``: ``[B, H_q, D]`` (typically bfloat16).
    - Cache entries (inner shapes as declared in :meth:`create_cache`):

      - ``cache["max"]`` and ``cache["min"]``: ``(1, head_dim)``
        → viewed as

        - ``[S, 1, D]`` in :meth:`forward_indexer`,
        - ``[B, 1, D]`` in :meth:`forward_cache`.

      - ``cache["k"]``: standard key cache with inner shape
        ``(page_size, head_dim)``.

      Here :math:`S` is the leading page axis. Internally it is a packed
      axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
      concatenating the pages from all requests. As a user, you can simply
      think of :math:`S` as "the number of pages for this request"; the
      vFlow kernels and :class:`ContextBase` will take care of mapping
      between per-request page counts and the packed layout automatically.
      
    Routing intuition
    -----------------
    For each query and page envelope:

    1. Compute elementwise products with the **max** and **min** envelopes.
    2. Take an elementwise maximum of these two products to form a
       QUEST-style upper bound.
    3. Sum over the feature dimension and then take a max over the
       grouped-query axis to get a single scalar score per page.
    4. Feed the resulting per-page scores into :class:`topK` to obtain
       sparse page indices.
    """

    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Multiply()      # q * max
        self.mul_min = Multiply()      # q * min
        self.maximum_op = Maximum()    # elementwise max(q*max, q*min)
        self.sum = Sum(dim=2)          # sum over feature dim D
        self.max_op = Max(dim=1)       # max over grouped-query axis
        self.output_func = topK()      # produce sparse indices

        # Cache-side ops
        self.reduction_max = CMax(dim=1)  # page-wise max envelope over k
        self.reduction_min = CMin(dim=1)  # page-wise min envelope over k

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Compute sparse page indices using QUEST-style envelope scores.

        Pipeline (indexer view)
        -----------------------
        Let:

        - ``q``: ``[B, H_q, D]``
        - ``cache["max"]``: ``[S, 1, D]``
        - ``cache["min"]``: ``[S, 1, D]``

        Steps:

        1. ``s_max = q * max_envelope``
        2. ``s_min = q * min_envelope``
        3. ``s = max(s_max, s_min)`` (elementwise)
        4. ``score = sum(s, dim=D)`` → ``[S, H_q, 1]``
        5. ``aggr_score = max(score, dim=H_q)`` → per-page scalar
        6. :class:`topK` converts ``aggr_score`` into sparse page
           indices ``o`` of shape ``[S_sparse, 1, 1]``.
        """
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Update per-page max/min envelopes from the key cache.

        Cache-update view
        -----------------
        - ``cache["k"]``: ``[B, page_size, D]``
        - ``cache["max"]``: ``[B, 1, D]``
        - ``cache["min"]``: ``[B, 1, D]``

        The :class:`CMax` and :class:`CMin` ops (with ``dim=1``) take
        page-wise maxima and minima over keys (optionally masked/structured
        via ``loc``) and write the envelopes into ``cache["max"]`` and
        ``cache["min"]``.
        """
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Parameters
        ----------
        page_size : int
            Number of tokens per page (unused here but part of the vFlow contract).

        head_dim : int
            Head dimension ``D`` used by the envelopes.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Custom cache metadata:

            - ``"max"``: inner shape ``(1, head_dim)``
            - ``"min"``: inner shape ``(1, head_dim)``
        """
        return {
            "max": (1, head_dim),
            "min": (1, head_dim),
        }


@register("h2o_sparse_attention")
class H2OSparseAttention(vFlow):
    r"""
    Page-level Heavy-Hitter Oracle (H2O) sparse attention flow.

    A **page-level approximation** of the H2O eviction criterion
    :cite:`zhang2023h2o` (arXiv:2306.14048). H2O ranks *tokens* by their
    cumulative attention mass; ``forward_cache`` runs only once per page and
    cannot accumulate across decode steps, so this flow instead ranks *pages*
    by two per-page statistics that together approximate "how much attention
    this page is likely to receive":

    1. **Average relevance** — :math:`q \cdot \mathrm{mean}_k(k)`. Up to the
       constant page size this is :math:`\sum_k q \cdot k`, i.e. the total
       pre-softmax attention mass of the page.
    2. **Peak relevance** — an upper bound on :math:`\max_k q \cdot k`,
       computed from the element-wise ``max``/``min`` envelope of the page:

       .. math::

           \max_k q \cdot k \;\le\;
           \sum_d \max\bigl(q_d \cdot \max_d,\; q_d \cdot \min_d\bigr).

       Both envelopes are required: with a signed query, :math:`q_d \max_d`
       alone is an upper bound only where :math:`q_d \ge 0`, and is a *lower*
       bound everywhere else. This is the same bound
       :class:`GQAQuestSparseAttention` uses, applied here to the
       heavy-hitter score.

    The two terms are combined as
    :math:`w_{\mathrm{avg}} \cdot s_{\mathrm{avg}} + w_{\mathrm{peak}} \cdot
    s_{\mathrm{peak}}`. They are not on the same scale — the peak term is an
    upper bound over the page and is therefore systematically larger than the
    mean term — so the weights are exposed rather than fixed at 1.

    Recent pages are always retained through the framework's
    ``page_reserved_eos`` mechanism, matching H2O's recent window.

    Cache layout
    ------------
    .. code-block:: python

        {
            "centroids": (1, head_dim),   # mean of keys in the page
            "max":       (1, head_dim),   # element-wise max envelope
            "min":       (1, head_dim),   # element-wise min envelope
        }

    Parameters
    ----------
    w_avg : float, optional
        Weight of the average-relevance term. Default ``1.0``.
    w_peak : float, optional
        Weight of the peak-relevance term. Default ``1.0``.

    References
    ----------
    .. [zhang2023h2o]
       Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen,
       Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian,
       Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen.
       *H2O: Heavy-Hitter Oracle for Efficient Generative Inference
       of Large Language Models*. arXiv:2306.14048, 2023.
    """

    def __init__(self, w_avg: float = 1.0, w_peak: float = 1.0):
        super().__init__()
        # Indexer-side ops (each op must be a separate instance)
        self.gemv_centroid = GeMV()
        self.mul_max = Multiply()               # q * max envelope
        self.mul_min = Multiply()               # q * min envelope
        self.maximum_op = Maximum()             # max(q*max, q*min), element-wise
        self.sum = Sum(dim=2)                   # sum over the feature dim D
        self.add = Add(alpha=w_avg, beta=w_peak)
        self.output_func = topK()

        # Cache-side ops
        self.reduction_mean = CMean(dim=1)
        self.reduction_max = CMax(dim=1)
        self.reduction_min = CMin(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Score pages by average relevance plus bounded peak relevance.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor ``[B, H_q, D]``.
        o : torch.Tensor
            Output sparse page indices ``[S_sparse, 1, 1]``.
        cache : Dict[str, torch.Tensor]
            Indexer view: ``"centroids"``, ``"max"`` and ``"min"``, each
            ``[S, 1, D]``.
        ctx : ContextBase
            Runtime context.

        Notes
        -----
        1. ``q_mean = mean(q, dim=H_q)`` → ``[B, 1, D]``
        2. ``s_avg = GeMV(q_mean, centroids)`` → ``[S, 1, 1]``
        3. ``env = Maximum(q_mean * max, q_mean * min)`` → ``[S, 1, D]``
        4. ``s_peak = Sum(env, dim=2)`` → ``[S, 1, 1]``
        5. ``score = w_avg * s_avg + w_peak * s_peak`` → ``[S, 1, 1]``
        6. ``topK(score, o)``
        """
        q_mean = q.mean(dim=1, keepdim=True)
        s_avg = self.gemv_centroid(q_mean, cache["centroids"], ctx=ctx)

        qk_max = self.mul_max(q_mean, cache["max"], ctx=ctx)
        qk_min = self.mul_min(q_mean, cache["min"], ctx=ctx)
        env = self.maximum_op(qk_max, qk_min, ctx=ctx)
        s_peak = self.sum(env, ctx=ctx)

        score = self.add(s_avg, s_peak, ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Refresh the per-page centroid and max/min envelopes from the keys.

        All three are overwriting reductions over ``cache["k"]``, so a page
        recycled from a finished request is fully reinitialised here.
        """
        self.reduction_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for the centroid and the max/min envelopes.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            ``"centroids"``, ``"max"`` and ``"min"``, each ``(1, head_dim)``.
        """
        return {
            "centroids": (1, head_dim),
            "max": (1, head_dim),
            "min": (1, head_dim),
        }


@register("gqa_h2o_sparse_attention")
class GQAH2OSparseAttention(vFlow):
    r"""
    GQA-style page-level Heavy-Hitter Oracle (H2O) sparse attention flow.

    Same scoring as :class:`H2OSparseAttention`, but every query head is
    scored independently and the per-head scores are aggregated with a max
    over the query group instead of the queries being averaged first.

    Both terms are reduced to a single scalar per page *before* they are
    combined, so that the weighted sum operates on two quantities of the same
    rank. Unlike :class:`GQABlockSparseAttention` no softmax is applied to
    either branch: normalising only one of the two terms would put them on
    incomparable scales, and the relative weighting is already exposed
    through ``w_avg`` / ``w_peak``.

    Cache layout
    ------------
    .. code-block:: python

        {
            "centroids": (1, head_dim),
            "max":       (1, head_dim),
            "min":       (1, head_dim),
        }

    Parameters
    ----------
    w_avg : float, optional
        Weight of the average-relevance term. Default ``1.0``.
    w_peak : float, optional
        Weight of the peak-relevance term. Default ``1.0``.
    """

    def __init__(self, w_avg: float = 1.0, w_peak: float = 1.0):
        super().__init__()
        # Indexer-side ops
        self.gemm_centroid = GeMM()
        self.max_avg = Max(dim=2)               # over the query-head axis
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        self.sum = Sum(dim=2)                   # over the feature dim D
        self.max_peak = Max(dim=1)              # over the query-head axis
        self.add = Add(alpha=w_avg, beta=w_peak)
        self.output_func = topK()

        # Cache-side ops
        self.reduction_mean = CMean(dim=1)
        self.reduction_max = CMax(dim=1)
        self.reduction_min = CMin(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Score pages per query head, then aggregate over the query group.

        Notes
        -----
        1. ``s_avg = GeMM(q, centroids)`` → ``[S, 1, H_q]``,
           ``avg = Max(s_avg, dim=2)`` → ``[S, 1, 1]``
        2. ``env = Maximum(q * max, q * min)`` → ``[S, H_q, D]``,
           ``s_peak = Sum(env, dim=2)`` → ``[S, H_q, 1]``,
           ``peak = Max(s_peak, dim=1)`` → ``[S, 1, 1]``
        3. ``score = w_avg * avg + w_peak * peak`` → ``[S, 1, 1]``
        4. ``topK(score, o)``
        """
        s_avg = self.gemm_centroid(q, cache["centroids"], ctx=ctx)
        avg = self.max_avg(s_avg, ctx=ctx)

        qk_max = self.mul_max(q, cache["max"], ctx=ctx)
        qk_min = self.mul_min(q, cache["min"], ctx=ctx)
        env = self.maximum_op(qk_max, qk_min, ctx=ctx)
        s_peak = self.sum(env, ctx=ctx)
        peak = self.max_peak(s_peak, ctx=ctx)

        score = self.add(avg, peak, ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Refresh the per-page centroid and max/min envelopes from the keys.
        """
        self.reduction_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for the centroid and the max/min envelopes.
        """
        return {
            "centroids": (1, head_dim),
            "max": (1, head_dim),
            "min": (1, head_dim),
        }


@register("h2o_token_sparse_attention")
class H2OTokenSparseAttention(vFlow):
    r"""
    Token-level H2O sparse attention with cross-step score accumulation.

    Implements the H2O heavy-hitter criterion :cite:`zhang2023h2o`
    (arXiv:2306.14048) at **token granularity**, using two mechanisms:

    1. **page_size = 1** — one token per page, so page-level ops are
       token-level ops. Enforced in :meth:`create_cache`.
    2. **A** :class:`Save` **in** :meth:`forward_indexer` — writes the
       updated running score back into the paged cache, which is what makes
       state survive across decode steps. ``forward_cache`` runs once per
       page and cannot accumulate.

    Per decode step:

    1. ``attn = GeMV(q_mean, cache["k"])`` — pre-softmax logits per token.
    2. ``Softmax(attn, dim=0)`` — normalise over the request's tokens, giving
       the actual attention probabilities. H2O accumulates attention
       *probabilities*, not raw logits; a running sum of raw logits is
       unbounded, sign-indefinite, and not the quantity the paper defines.
    3. ``score = attn + decay * cache["hh_score"]`` — accumulate.
    4. ``Save(score → cache["hh_score"])`` — persist.
    5. ``topK(score, o)`` — select the heavy hitters.

    Recent tokens are retained through ``page_reserved_eos`` (H2O's recent
    window) and initial tokens through ``page_reserved_bos``.

    Why the score decays
    --------------------
    The KV pool is allocated in bfloat16, so ``hh_score`` accumulates in
    bfloat16 (~8 bits of mantissa). An undecayed running sum grows without
    bound, and once it does, each new increment is small relative to the
    accumulated value and is lost to rounding: with attention probabilities
    over a few hundred tokens the accumulator stops changing after roughly
    500 decode steps, freezing the ranking at whatever it was then. Any token
    that becomes important later can never be selected.

    A decay factor keeps the accumulator at a bounded steady state of
    ``1 / (1 - decay)`` times a typical increment, which keeps increments
    representable. The default ``0.984375`` is ``1 - 2**-6`` — exactly
    representable in bfloat16, an effective window of ~64 decode steps, and
    the longest window that still reproduces the top-k of an exact
    (float64) accumulator in simulation. Set ``decay=1.0`` to recover the
    plain undecayed sum, with the caveat above.

    Requirements
    ------------
    - ``page_size=1`` (asserted)
    - ``disable_radix_cache=True`` — prefix sharing would let one request's
      accumulated scores leak into another through shared pages. This cannot
      be checked from inside the flow; it must be set on the server.

    Format flow
    -----------
    ::

        GeMV(q_BATCHED, k_PAGED)          -> attn_RAGGED   [S, 1, 1]
        Softmax(attn_RAGGED, dim=0)                        (in place)
        Add(attn_RAGGED, hh_score_PAGED)  -> score_RAGGED  [S, 1, 1]
        Save(score_RAGGED -> cache["hh_score"]_PAGED)
        topK(score_RAGGED -> o)

    Parameters
    ----------
    decay : float, optional
        Multiplier applied to the previous accumulated score. Default
        ``0.984375``. See "Why the score decays".
    scale : float, optional
        Softmax temperature. Defaults to ``head_dim ** -0.5`` (the standard
        attention scale), resolved in :meth:`create_cache`.

    Notes
    -----
    The design of the token-level variant — using :class:`Save` inside
    ``forward_indexer`` for cross-step accumulation, together with
    ``page_size=1`` and a disabled radix cache — was suggested by
    Zhuoming Chen.

    References
    ----------
    .. [zhang2023h2o]
       Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen,
       Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian,
       Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen.
       *H2O: Heavy-Hitter Oracle for Efficient Generative Inference
       of Large Language Models*. arXiv:2306.14048, 2023.
    """

    def __init__(self, decay: float = 0.984375, scale: Optional[float] = None):
        super().__init__()
        self._scale = scale
        # Indexer-side ops
        self.gemv = GeMV()                          # q . k^T per token
        self.softmax = Softmax(dim=0, scale=1.0)    # scale resolved in create_cache
        self.add = Add(alpha=1.0, beta=decay)       # attn + decay * old_score
        self.save = Save()                          # persist to the paged cache
        self.output_func = topK()                   # select heavy hitters

        # Cache-side ops
        self.zero_init = CFill(alpha=0.0)           # reset a page on (re)allocation

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Normalise attention, accumulate into the running score, persist, select.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor ``[B, H_q, D]`` (BATCHED).
        o : torch.Tensor
            Output sparse page indices ``[S_sparse, 1, 1]``.
        cache : Dict[str, torch.Tensor]
            Indexer view (PAGED): ``cache["k"]`` ``[S, 1, D]`` (page_size=1)
            and ``cache["hh_score"]`` ``[S, 1, 1]``.
        ctx : ContextBase
            Runtime context.
        """
        # 1. Mean-pool query heads -> [B, 1, D]
        q_mean = q.mean(dim=1, keepdim=True)

        # 2. Attention logits for every cached token -> [S, 1, 1] RAGGED
        attn = self.gemv(q_mean, cache["k"], ctx=ctx)

        # 3. Normalise over this request's tokens: logits -> probabilities
        self.softmax(attn, ctx=ctx)

        # 4. score = attn + decay * old_cumulative -> [S, 1, 1] RAGGED
        #    Add dispatches (RAGGED, PAGED) -> RAGGED
        score = self.add(attn, cache["hh_score"], ctx=ctx)

        # 5. Persist the updated score back into the paged cache (RAGGED -> PAGED)
        self.save(score, cache["hh_score"], ctx=ctx)

        # 6. Select the heavy hitters; page_reserved_eos covers the recent window
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Zero the accumulated score for every page written in this step.

        The KV pool is zeroed once at allocation and never again: a page
        freed by a finished request keeps its contents and is handed to the
        next request as-is. Without this reset a new request would inherit
        the previous one's accumulated heavy-hitter scores. ``forward_cache``
        runs exactly once per page, right after its key/value are written,
        which is precisely when the score must start from zero.
        """
        self.zero_init(cache["hh_score"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare the cumulative-score cache and resolve the softmax scale.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            ``"hh_score"`` with inner shape ``(1, 1)`` — one scalar running
            score per token.
        """
        assert page_size == 1, (
            f"{type(self).__name__} is token-level and requires page_size=1, "
            f"got page_size={page_size}. Use h2o_sparse_attention for "
            f"page-level H2O."
        )
        if self._scale is None:
            self.softmax.scale = head_dim ** -0.5
        else:
            self.softmax.scale = self._scale
        return {
            "hh_score": (1, 1),
        }


@register("gqa_h2o_token_sparse_attention")
class GQAH2OTokenSparseAttention(vFlow):
    r"""
    GQA-aware token-level H2O sparse attention with cross-step accumulation.

    Same as :class:`H2OTokenSparseAttention`, but each query head is scored
    independently with :class:`GeMM` and the per-head attention probabilities
    are aggregated with a max over the query group before accumulation.

    Aggregating with a max means a token is kept if *any* head in the group
    attends to it strongly, which is the conservative choice for GQA: the
    group shares one KV cache, so evicting a token hurts every head in it.
    The alternative — tracking one score per head — would multiply the
    ``hh_score`` cache by the group size and is left out here.

    Requirements
    ------------
    - ``page_size=1`` (asserted)
    - ``disable_radix_cache=True`` (see :class:`H2OTokenSparseAttention`)

    Format flow
    -----------
    ::

        GeMM(q_BATCHED, k_PAGED)              -> attn_RAGGED     [S, 1, H_q]
        Softmax(attn_RAGGED, dim=0)                              (in place)
        Max(attn_RAGGED, dim=2)               -> attn_agg_RAGGED [S, 1, 1]
        Add(attn_agg_RAGGED, hh_score_PAGED)  -> score_RAGGED    [S, 1, 1]
        Save(score_RAGGED -> cache["hh_score"]_PAGED)
        topK(score_RAGGED -> o)

    Parameters
    ----------
    decay : float, optional
        Multiplier applied to the previous accumulated score. Default
        ``0.984375``. See :class:`H2OTokenSparseAttention`.
    scale : float, optional
        Softmax temperature. Defaults to ``head_dim ** -0.5``.

    Notes
    -----
    The design of the token-level variant was suggested by Zhuoming Chen;
    see :class:`H2OTokenSparseAttention`.
    """

    def __init__(self, decay: float = 0.984375, scale: Optional[float] = None):
        super().__init__()
        self._scale = scale
        # Indexer-side ops
        self.gemm = GeMM()                          # q . k^T per head
        self.softmax = Softmax(dim=0, scale=1.0)    # scale resolved in create_cache
        self.max_op = Max(dim=2)                    # aggregate over query heads
        self.add = Add(alpha=1.0, beta=decay)       # attn + decay * old_score
        self.save = Save()                          # persist to the paged cache
        self.output_func = topK()                   # select heavy hitters

        # Cache-side ops
        self.zero_init = CFill(alpha=0.0)           # reset a page on (re)allocation

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Per-head attention, group aggregation, accumulation, persist, select.
        """
        # 1. Per-head attention logits -> [S, 1, H_q] RAGGED
        attn = self.gemm(q, cache["k"], ctx=ctx)

        # 2. Normalise over this request's tokens, per head
        self.softmax(attn, ctx=ctx)

        # 3. Aggregate over the query group -> [S, 1, 1] RAGGED
        attn_agg = self.max_op(attn, ctx=ctx)

        # 4. score = attn_agg + decay * old_cumulative -> [S, 1, 1]
        score = self.add(attn_agg, cache["hh_score"], ctx=ctx)

        # 5. Persist
        self.save(score, cache["hh_score"], ctx=ctx)

        # 6. Select
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Zero the accumulated score for every page written in this step.

        See :meth:`H2OTokenSparseAttention.forward_cache`.
        """
        self.zero_init(cache["hh_score"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare the cumulative-score cache and resolve the softmax scale.
        """
        assert page_size == 1, (
            f"{type(self).__name__} is token-level and requires page_size=1, "
            f"got page_size={page_size}. Use gqa_h2o_sparse_attention for "
            f"page-level H2O."
        )
        if self._scale is None:
            self.softmax.scale = head_dim ** -0.5
        else:
            self.softmax.scale = self._scale
        return {
            "hh_score": (1, 1),
        }


# For agent developers!
# The ops are not reusable, even if they have the same semantic meaning. Internally, they will initialize different memory buffer.
# For example, in Quest attention, we need to define two multiply operators.
# In the entire flow (including forward_cache/forward_indexer), native torch Ops are only allowed to apply to q in forward_indexer. For other tensors, please use vortex_torch ops in indexer/ and cache/.

# In forward indexer, q can be viewed as [1, H_q, D] or [B, H_q, D] (B=1) and cache["xxx"] can be viewed as [S, r, c] (r, c defined in create_cache) logically.
# In forward cache, the cache["xxx"] is viewed as [B, r, c]  (r, c defined in create_cache) logically.
# In forward cache, each page is computed only once if page_id appears in loc. During the entire computation, each page id will appear in loc only once. Thus, users cannot accumulate tensors through forward_cache.
# Thus, all the tensors have 3 dimensions. Reduce operators (Mean, Max, Min, etc) will always keep the dims.

# Tips 1: GeMM(x, y) = yx^t, which might be different from typical definitions.
# Tips 2: Except cache["k"], cache["v"] can also be used in forward_cache to collect information.