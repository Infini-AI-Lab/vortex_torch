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
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache import Context
from vortex_torch.cache.compiler.compile import compile as compile_cache
from vortex_torch.flow.flow_mla import vFlowMLA

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


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
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
        self._refresh_aux(layer, loc)

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
        super().set_kv_buffer(layer, loc, cache_k, cache_v)
        self._refresh_aux(layer, loc)

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
