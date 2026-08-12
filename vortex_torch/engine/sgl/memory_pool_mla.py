"""
MLA (Multi-head Latent Attention) variant of the vortex sparse-attention KV
pool — for DeepSeek-V2/V3-style models (DeepSeek-V2-Lite, GLM-4.7-Flash).

**Subclasses sglang's `MLATokenToKVPool`** so it IS-A MLA pool (chunked-prefix
cache + every sglang MLA code path accept it) and inherits the canonical fused
latent buffer `kv_buffer` (`[size+page, 1, kv_lora_rank+qk_rope_head_dim]`,
token-major) plus `set_mla_kv_buffer` / `get_key_buffer`.

On top of that it layers the **vortex** machinery: per-page aux tensors (e.g. a
latent centroid) and a compiled `forward_cache` that refreshes them from the
latent on each write. The vortex cache kernel addresses `cache["latent"]` by
`block_id*block_size*latent_dim` (raw pointer, constexpr dims) — which equals
the token-major flat layout of `kv_buffer` for any blocks-per-page — so the
latent field is just `kv_buffer[layer]` (no copy, no second KV cache).
"""
import logging
from typing import List, Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, unwrap_write_loc

from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache import Context
from vortex_torch.cache.compiler.compile import compile as compile_cache
from vortex_torch.flow.flow_mla import vFlowMLA
from .config import cfg as _vortex_cfg
from .host_kv import HostKVCache, set_host_latent

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024

#: See memory_pool._HOST_KV_STAGING_FRACTION.
_HOST_KV_STAGING_FRACTION = 0.25


class VortexMLACachePool(MLATokenToKVPool):

    supports_fused_set_kv_buffer = False

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        sparse_attention: vFlowMLA,
        model_runner,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        # Allocate the canonical MLA latent buffer (kv_buffer) + set
        # kv_lora_rank/qk_rope_head_dim/kv_cache_dim.
        MLATokenToKVPool.__init__(
            self, size, page_size, dtype, kv_lora_rank, qk_rope_head_dim,
            layer_num, device, enable_memory_saver, start_layer, end_layer,
        )

        # --- vortex setup ---
        self.sparse_attention = sparse_attention
        self.ctx = Context()
        self.block_size = model_runner.block_size
        assert self.page_size % self.block_size == 0, (
            "Page size must be a multiple of block size for block-sparse attention"
        )
        self.num_blocks_per_page = self.page_size // self.block_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        # Single shared KV head; the cache Context reads parent.head_num/head_dim.
        self.head_num = 1
        self.head_dim = self.kv_cache_dim
        # Pages for the aux/centroid buffers (one centroid row per block).
        self.num_pages = (self.size + self.page_size + self.page_size - 1) // self.page_size + 1

        # Host-resident latent KV (see engine/sgl/host_kv.py). MLA has a single
        # fused ``latent`` field instead of separate K/V, and it is allocated by
        # the *parent* sglang class, so hosting it means replacing those tensors
        # after construction rather than choosing a device at allocation time.
        # The aux/centroid buffers stay on the GPU for the same reason as MHA: the
        # indexer scores every cached block every step.
        self.vortex_cfg = _vortex_cfg(model_runner)
        self.host_kv_gb = float(getattr(self.vortex_cfg, "host_kv_gb", 0.0) or 0.0)
        self.host_kv = self.host_kv_gb > 0.0
        self.host_kv_policy = getattr(self.vortex_cfg, "host_kv_policy", "lru") or "lru"
        self.host_kv_caches: List["HostKVCache"] = []
        self._prefix_src = None
        self._prefix_n = 0
        if self.host_kv:
            self._rehost_latent(model_runner)

        self.cache_meta_info = self.sparse_attention.get_cache_meta_info()
        self._create_aux_buffers()      # only the non-"latent" fields
        self._compile(model_runner)     # trace + compile forward_cache

        cache_size = self.get_cache_size_bytes()
        logger.info(
            f"MLA KV Cache allocated. #tokens: {size}, "
            f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
            f"Cache size: {cache_size / GB:.2f} GB"
        )
        self.mem_usage = cache_size / GB
        assert self.dtype == torch.bfloat16, (
            f"MLA pool currently supports only bf16 KV (got {self.dtype}); "
            f"fp8 latent path is a follow-up."
        )

    # ------------------------------------------------------------------ #
    # host-resident latent KV
    # ------------------------------------------------------------------ #
    def _rehost_latent(self, model_runner) -> None:
        """Move the latent ``kv_buffer`` to pinned host memory and size the pools.

        The parent already allocated ``kv_buffer`` on the GPU, so this frees those
        tensors and re-allocates pinned host ones. Re-sizing ``self.size`` first is
        deliberate: sglang picked it to fit HBM, which is the wrong bound once the
        latent lives on the host, and it must match what is allocated or the
        scheduler admits requests with no backing storage.

        The one MLA-specific wrinkle: ``kv_buffer`` is token-major
        ``[size + page, 1, kv_cache_dim]`` while :class:`HostKVCache` addresses
        blocks as ``[num_blocks, block_size, dim]``. Those are the same bytes in
        the same order (which is exactly why the vortex cache kernel can treat the
        flat latent as block-addressed), so a ``view`` suffices — and the *same
        storage* is handed to both, so a write through ``kv_buffer`` is visible to
        the fetch kernel with no synchronisation.
        """
        elt = torch._utils._element_size(self.store_dtype)
        bytes_per_token = self.layer_num * self.kv_cache_dim * elt
        tokens = int(self.host_kv_gb * GB) // bytes_per_token
        tokens = (tokens // self.page_size) * self.page_size
        if tokens < self.page_size:
            raise ValueError(
                f"host_kv_gb={self.host_kv_gb} is too small for this model: one page "
                f"of {self.page_size} tokens needs "
                f"{bytes_per_token * self.page_size / GB:.3f} GB "
                f"({self.layer_num} layers x latent dim {self.kv_cache_dim})."
            )
        # ``size`` arrives already correct: sglang derived it from the HBM budget
        # via ``integration.kv_cell_size``, which under host KV counts only the
        # device-resident aux fields, and sized the allocator / req_to_token_pool
        # from the same number. So this only CLAMPS to the user's host_kv_gb --
        # rewriting it upward would desync those other pools, and the runner is
        # read-only during construction to prevent that.
        if tokens < self.size:
            logger.info(
                "vortex host MLA latent: clamping %d tokens to %d to fit "
                "host_kv_gb=%.2f", self.size, tokens, self.host_kv_gb,
            )
            self.size = tokens
            self.num_pages = (
                (self.size + self.page_size + self.page_size - 1) // self.page_size + 1
            )

        rows = self.size + self.page_size
        # Drop the GPU allocation before taking the host one, so peak footprint
        # never holds both.
        del self.kv_buffer
        torch.cuda.empty_cache()
        self.kv_buffer = [
            torch.zeros((rows, 1, self.kv_cache_dim),
                        dtype=self.store_dtype, device="cpu", pin_memory=True)
            for _ in range(self.layer_num)
        ]

        # Size the block view from the PLANNER's page count, not from ``rows``.
        # ``num_pages`` carries a trailing guard page (sglang's padded slot 0), so a
        # block table entry can reach ``num_pages * blocks_per_page - 1``, which is
        # 1-2 blocks PAST ``rows // block_size``. Handing HostKVCache the smaller
        # figure sizes ``slot_of`` / ``claim_gen`` too short, and the fetch kernel
        # then indexes them out of bounds — an illegal access (or silent corruption
        # of whatever follows) that only shows up under load. Verified out of range
        # for every size/page combination checked.
        blocks_per_page = self.page_size // self.block_size
        blocks = max(rows // self.block_size, self.num_pages * blocks_per_page)
        # Grow the host buffer to match, so those ids address real memory.
        need_rows = blocks * self.block_size
        if need_rows > rows:
            self.kv_buffer = [
                torch.zeros((need_rows, 1, self.kv_cache_dim),
                            dtype=self.store_dtype, device="cpu", pin_memory=True)
                for _ in range(self.layer_num)
            ]
            rows = need_rows
        pool_blocks = self._host_kv_pool_blocks(model_runner)
        for li in range(self.layer_num):
            flat = self.kv_buffer[li].view(blocks, self.block_size, self.kv_cache_dim)
            # MLA has one fused latent rather than separate K and V; pass it as both
            # so the shared cache logic is reused verbatim. The V staging buffer is
            # redundant, so ``fetch``'s second return value is simply ignored by the
            # MLA backends.
            self.host_kv_caches.append(
                HostKVCache(flat, flat, pool_blocks, self.device, fused=True,
                            policy=self.host_kv_policy)
            )
        logger.info(
            f"vortex host MLA latent cache: "
            f"{sum(t.element_size() * t.numel() for t in self.kv_buffer) / GB:.2f} GB "
            f"pinned host ({rows} tokens x {self.layer_num} layers), "
            f"{pool_blocks}-block GPU staging pool per layer"
        )

    def _host_kv_pool_blocks(self, model_runner) -> int:
        """Staging blocks per layer — sized from the per-step selection budget.

        Same reasoning as the MHA pool: only the blocks selected on one step need
        to be resident, and the capacity bound needs one spare slot beyond the
        worst-case demand.
        """
        override = int(getattr(self.vortex_cfg, "host_kv_pool_blocks", 0) or 0)
        if override > 0:
            return override
        sa = model_runner.server_args
        rows = int(model_runner.req_to_token_pool.size)   # MLA: a single KV head
        static = (int(sa.vortex_topk_val) + int(sa.vortex_block_reserved_bos)
                  + int(sa.vortex_block_reserved_eos))
        seq = (model_runner.model_config.context_len if sa.vortex_max_seq_lens < 0
               else sa.vortex_max_seq_lens)
        max_blocks = (seq + self.block_size - 1) // self.block_size
        ratio = float(getattr(sa, "vortex_topk_ratio", 0.0) or 0.0)
        per_row = min(max(static, int(max_blocks * ratio)), max_blocks)

        # Same clamp as the MHA pool: the worst-case "every schedulable request
        # resident at full budget" figure is unaffordable, and the pool is a cache,
        # so a smaller one costs hit rate rather than correctness (overflow is
        # counted). MLA stages one fused latent buffer instead of K and V, so a
        # given byte budget buys twice the blocks.
        want = HostKVCache.required_capacity(rows, per_row)
        block_numel = self.block_size * self.kv_cache_dim
        elt = torch._utils._element_size(self.store_dtype)
        free, _ = torch.cuda.mem_get_info()
        afford = HostKVCache.affordable_capacity(
            block_numel, elt, self.layer_num,
            int(_HOST_KV_STAGING_FRACTION * free), fused=True,
        )
        floor = HostKVCache.required_capacity(1, per_row)   # one request, 1 KV head
        cap = max(floor, min(want, afford))
        if cap < want:
            logger.info(
                "vortex host MLA staging pool: %d blocks/layer (worst-case demand "
                "%d would need %.1f GB across %d layers; capped to %.1f GB)",
                cap, want, want * block_numel * elt * self.layer_num / GB,
                self.layer_num, cap * block_numel * elt * self.layer_num / GB,
            )
        return cap

    def fetch_latent(self, layer_id: int, table, *, row_lens=None, indptr=None,
                     num_rows: int, max_per_row: int):
        """Stage the selected latent blocks; returns ``(latent, table_to_use)``."""
        if not self.host_kv:
            return self.get_fused_latent_buffer(layer_id), table
        k, _v, remapped = self.host_kv_caches[layer_id - self.start_layer].fetch(
            table, row_lens=row_lens, indptr=indptr,
            num_rows=num_rows, max_per_row=max_per_row,
        )
        # ``_v`` aliases ``k`` (fused latent). ``remapped`` is a pool-owned table:
        # the caller's is shared across layers and must not be rewritten.
        return k, remapped

    def host_kv_tick(self, block_tables=None) -> None:
        """Advance the staging generation, and (once) reserve the remap buffer.

        ``block_tables`` is the full sparse block table. Reserving here — from the
        per-step planner, outside any captured region — guarantees ``fetch`` never
        allocates, which inside a cuda graph would be captured into the graph.
        """
        if not self.host_kv:
            return
        for c in self.host_kv_caches:
            c.tick()
            if block_tables is not None:
                c.reserve_remap(block_tables)

    # ------------------------------------------------------------------ #
    # vortex aux buffers + compilation
    # ------------------------------------------------------------------ #
    def _create_aux_buffers(self):
        """Allocate the per-block aux tensors (everything except 'latent', which
        is the inherited kv_buffer)."""
        rows = self.num_pages * self.num_blocks_per_page
        self.aux = [
            {
                name: torch.zeros((rows, shape[0], shape[1]), dtype=dt, device=self.device)
                for name, (shape, dt) in self.cache_meta_info.items()
                if name != "latent"
            }
            for _ in range(self.layer_num)
        ]

    def _layer_cache(self, layer_id: int) -> dict:
        """The {name: tensor} dict the compiled cache / indexer consume:
        'latent' is the inherited kv_buffer (token-major flat == block layout),
        the rest are the aux tensors."""
        li = layer_id - self.start_layer
        cache = {"latent": self.kv_buffer[li]}
        cache.update(self.aux[li])
        return cache

    def _compile(self, model_runner) -> None:
        """Trace the sparse-attention forward_cache on zero-sized dummies and
        compile it (mirrors VortexCachePool._compile, but 'latent' is provided
        by kv_buffer at runtime rather than self-allocated)."""
        self.ctx.create(self, model_runner)
        self.ctx.profile()

        def register(vt, name):
            self.ctx.tensor_list.append(vt)
            self.ctx.output_tensor_to_op_list.append(None)
            self.ctx.tensor_id_to_tensor_name_map[vt.tensor_id] = name

        with torch.no_grad():
            loc_dummy = torch.empty((0,), dtype=torch.int64, device=self.device)
            cache_dummy = {}
            for i, (name, (shape, cache_dtype)) in enumerate(self.cache_meta_info.items()):
                vt = as_vtensor(
                    torch.zeros((0, shape[0], shape[1]), dtype=cache_dtype, device=self.device),
                    FORMAT.PAGED, tensor_id=i,
                )
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")
            self.sparse_attention.forward_cache(cache=cache_dummy, loc=loc_dummy, ctx=self.ctx)

        self.compiled_cache = compile_cache(self.ctx)()
        self.ctx.summary()
        self.ctx.execute()

    def get_cache_size_bytes(self) -> int:
        # Device bytes only: this feeds ``mem_usage``, which sglang reads as HBM
        # occupancy. Under host KV the latent is host-resident, so counting it
        # would overstate GPU use by the whole KV cache; the GPU staging pools are
        # counted in its place.
        if self.host_kv:
            total = sum(c.nbytes_device() for c in self.host_kv_caches)
        else:
            total = sum(t.element_size() * t.numel() for t in self.kv_buffer)
        for layer_aux in self.aux:
            for t in layer_aux.values():
                total += t.element_size() * t.numel()
        return total

    # ------------------------------------------------------------------ #
    # KV write — inherited scatter into kv_buffer, then refresh aux
    # ------------------------------------------------------------------ #
    def _refresh_aux(self, layer, loc: torch.Tensor):
        """Recompute per-page aux (centroids) from the just-written latent."""
        if layer.layer_id in self.layers_skip:
            return
        # The (once-compiled) cache pipeline runs for every decode layer's aux
        # refresh; pass the active global layer id as the EXPLICIT trailing
        # ``cur_layer`` arg so per-layer-weight cache ops (e.g. the
        # LearnedDescriptor Parameters) select the active layer's baked slice.
        # The arg defaults to 0 in the generated forward(), so centroid flows
        # that don't read it are unaffected.
        # Host-resident latent: the blocks covering these tokens may be resident in
        # the staging pool holding their PRE-write contents, and the local window
        # re-selects exactly the newest block every step — so serving the cached
        # copy would attend over stale (usually zero) latent. Evict them before the
        # aux refresh; both write paths (set_kv_buffer / set_mla_kv_buffer) reach
        # here, so this covers prefill and decode.
        if self.host_kv:
            li = layer.layer_id - self.start_layer
            # MLA's latent is token-major with a single KV head, so a token's block
            # is simply ``pos // block_size`` — no head interleave to undo, unlike
            # the MHA pool. ``page_size=block_size`` and one head make
            # ``invalidate_locs``'s address arithmetic reduce to exactly that.
            self.host_kv_caches[li].invalidate_locs(
                loc.to(torch.int64), self.block_size, 1
            )
        self.compiled_cache.forward(
            self._layer_cache(layer.layer_id), loc.to(torch.int64), ctx=self.ctx,
            cur_layer=layer.layer_id,
        )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        # Absorb-path write (trtllm_mla backend): separate [kv_c], [k_pe].
        loc_t, _, _ = unwrap_write_loc(loc)     # may be a KVWriteLoc bundle
        if self.host_kv:
            # Upstream's writer is a JIT'd CUDA kernel that asserts its destination
            # is on cuda:0, so it refuses the pinned host latent outright ("Device
            # mismatch: expected cuda:0 but got cpu"). vortex's Triton equivalent
            # just dereferences the pointer, which works for pinned host memory.
            set_host_latent(
                self.kv_buffer[layer.layer_id - self.start_layer],
                loc_t.to(torch.int64), cache_k_nope, cache_k_rope,
            )
        else:
            super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
        self._refresh_aux(layer, loc_t)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor = None,
    ):
        # Fused-path write (triton backend prefill + decode via TritonAttnBackend,
        # which calls set_kv_buffer with the fused [kv_c | k_pe] as cache_k). Must
        # refresh centroids here too, else prefill-token centroids stay stale and
        # the indexer selects bad blocks under sparsity.
        #
        # ``loc`` may be a ``KVWriteLoc`` bundle rather than a tensor (the triton
        # MLA backend passes one). ``super().set_kv_buffer`` unwraps it internally,
        # which is why the pre-existing path never had to — but the host-KV writer
        # and the aux refresh both index with it directly, so unwrap once here.
        loc_t, _, _ = unwrap_write_loc(loc)
        if self.host_kv:
            # Same reason as set_mla_kv_buffer: the upstream fused write lands in a
            # device-asserting kernel. Split the fused [kv_c | k_pe] back into the
            # two halves the host writer takes.
            k_f = cache_k.view(-1, 1, self.kv_cache_dim)
            set_host_latent(
                self.kv_buffer[layer.layer_id - self.start_layer],
                loc_t.to(torch.int64),
                k_f[..., : self.kv_lora_rank],
                k_f[..., self.kv_lora_rank :],
            )
        else:
            super().set_kv_buffer(layer, loc, cache_k, cache_v)
        self._refresh_aux(layer, loc_t)

    # ------------------------------------------------------------------ #
    # accessors for the vortex indexer / sparse decode
    # ------------------------------------------------------------------ #
    def get_cache(self, layer_id: int) -> dict:
        """{'latent': kv_buffer, 'centroids': aux, ...} consumed by the indexer."""
        return self._layer_cache(layer_id)

    def get_fused_latent_buffer(self, layer_id: int) -> torch.Tensor:
        """kv_buffer viewed page-major [num_pages, page_size, 576] for the
        trtllm_mla decode kernel."""
        return self.kv_buffer[layer_id - self.start_layer].view(
            -1, self.page_size, self.kv_cache_dim
        )

    # ------------------------------------------------------------------ #
    # PD disaggregation: rebuild aux on the decode side
    # ------------------------------------------------------------------ #
    def rebuild_aux(self, loc: torch.Tensor):
        if loc is None or loc.numel() == 0:
            return
        loc = loc.to(torch.int64)
        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            if layer_id in self.layers_skip:
                continue
            self.compiled_cache.forward(
                self._layer_cache(layer_id), loc, ctx=self.ctx, cur_layer=layer_id,
            )
