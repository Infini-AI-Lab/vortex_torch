from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch

from ..abs import ContextBase
from ..utils import resolve_dtype


class vFlowMLA(ABC):
    r"""
    Base class for **MLA** sparse-attention flows (DeepSeek-V2/V3-style, e.g.
    DeepSeek-V2-Lite, GLM-4.7-Flash).

    Parallel to :class:`~vortex_torch.flow.flow.vFlow`, but for the compressed
    **latent** KV layout. Where the MHA flow stores per-head ``"k"``/``"v"``, an
    MLA flow stores a single shared, fused latent per token:

    - ``"latent"`` — the decode KV cache row, inner shape
      ``(block_size, kv_lora_rank + qk_rope_head_dim)`` (e.g. 512+64 = 576),
      laid out as ``[ kv_c | k_pe ]``. **Auto-injected** by :meth:`initialize`;
      subclasses must not declare it.

    This single buffer serves everything: :meth:`forward_cache` reduces a
    centroid from it, :meth:`forward_indexer` scores against that centroid, and
    the trtllm_mla decode kernel reads it directly — no duplicated KV.

    The query reaching :meth:`forward_indexer` is the **fused absorbed query**
    ``q = [ q_nope_out | q_pe ]`` of inner dim ``kv_lora_rank + qk_rope_head_dim``
    (the backend concatenates the absorbed latent-space query with the RoPE
    query). Scoring ``q`` against a ``latent`` centroid therefore yields the
    **full** decode logit ``⟨q_nope,kv_c⟩ + ⟨q_pe,k_pe⟩`` from one dot product.
    """

    def __init__(self):
        super().__init__()
        self.block_size = None
        self.kv_lora_rank = None
        self.qk_rope_head_dim = None
        self.latent_dim = None
        self.kv_cache_dtype = None
        self.q_data_type = None
        self.intermediate_dtype = None
        self.cache_meta_info = None
        self.token_ratio = None

    # ------------------------------------------------------------------ #
    # abstract API
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
        Score the cached pages from the fused absorbed query and emit the sparse
        page set into ``o`` (must end in ``topK`` / ``approxTopK``).

        Shapes
        ------
        - ``q`` : ``[B, H, latent_dim]`` — fused query ``[q_nope_out | q_pe]``
          (latent_dim = kv_lora_rank + qk_rope_head_dim).
        - ``cache["latent"]`` : ``[S, 1, latent_dim]`` (page-packed, single head).
        - ``o`` : ``[S_sparse, 1, 1]`` integer page ids, written in place.

        ``q`` is per-head (``H``); the latent KV is a single shared head, so
        implementations aggregate over ``H`` (e.g. ``Mean(dim=1)``) before
        scoring against the per-page latent centroid.
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
        Refresh per-page aux state from the freshly written latent. ``cache["latent"]``
        (``[B, 1, latent_dim]``) is auto-provided; compute summaries (e.g. a
        ``CMean`` centroid over ``latent`` per page) into the aux tensors declared
        by :meth:`create_cache`.
        """
        pass

    @abstractmethod
    def create_cache(
        self,
        block_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> Dict[str, Tuple[int, int]]:
        r"""
        Declare inner shapes ``(r, c)`` for the aux cache tensors (everything
        **except** ``"latent"``, which is reserved and injected by
        :meth:`initialize`). E.g.
        ``{"centroids": (1, kv_lora_rank + qk_rope_head_dim)}``.
        """
        pass

    # ------------------------------------------------------------------ #
    # framework helpers
    # ------------------------------------------------------------------ #
    def get_cache_meta_info(
        self,
    ) -> Dict[str, Tuple[Tuple[int, int], torch.dtype]]:
        return self.cache_meta_info

    def get_token_ratio(self) -> float:
        return self.token_ratio

    def get_aux_token_ratio(self) -> float:
        """Token ratio counting only HBM-resident cache fields (host-KV mode)."""
        return self.aux_token_ratio

    def initialize(
        self,
        block_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        kv_cache_dtype: Union[torch.dtype, str],
        q_data_type: Union[torch.dtype, str],
        intermediate_dtype: Union[torch.dtype, str] = torch.bfloat16,
    ):
        r"""Set dtypes/dims, then build ``cache_meta_info`` by injecting the
        reserved fused ``"latent"`` field alongside the aux fields from
        :meth:`create_cache`."""
        self.block_size = block_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.latent_dim = kv_lora_rank + qk_rope_head_dim
        self.kv_cache_dtype = resolve_dtype(kv_cache_dtype)
        self.q_data_type = resolve_dtype(q_data_type)
        self.intermediate_dtype = resolve_dtype(intermediate_dtype)
        self.token_ratio = 0.0

        raw = self.create_cache(block_size, kv_lora_rank, qk_rope_head_dim)
        assert "latent" not in raw, "create_cache must not declare 'latent' key"
        assert "k" not in raw and "v" not in raw and "kv_c" not in raw and "k_pe" not in raw, (
            "MLA flows use the single fused 'latent' field"
        )
        raw["latent"] = (block_size, self.latent_dim)

        base_bytes = block_size * self.latent_dim * torch._utils._element_size(self.kv_cache_dtype)
        self.cache_meta_info = {}
        total_bytes = 0
        aux_bytes = 0
        for key, (r, c) in raw.items():
            dtype = self.kv_cache_dtype if key == "latent" else self.intermediate_dtype
            nbytes = r * c * torch._utils._element_size(dtype)
            total_bytes += nbytes
            if key != "latent":
                aux_bytes += nbytes
            self.cache_meta_info[key] = ((r, c), dtype)
        self.token_ratio = total_bytes / base_bytes
        #: Ratio counting only the HBM-resident fields, for the host-KV mode where
        #: the fused latent lives in pinned host memory (see flow.py).
        self.aux_token_ratio = aux_bytes / base_bytes
