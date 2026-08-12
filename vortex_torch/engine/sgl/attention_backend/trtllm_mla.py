from __future__ import annotations

"""
Vortex **standalone** sparse-attention backend for MLA models (DeepSeek-V2/V3,
GLM-4.7-Flash) on the trtllm MLA decode kernel.

Mirrors the structure of the MHA `trtllm.py` `VortexTRTLLMBackend` — it extends
sglang's base `AttentionBackend` (NOT the dense MLA backend) and manages its own
vortex tensors (`ctx.metadata`, `batch_table`, `qo_indptr`, workspaces) and the
indexer compile. Differences from the MHA backend:
  - single shared KV head (num_kv_heads=1, no head-fold);
  - KV is the fused latent `cache["latent"]` from `VortexMLACachePool`;
  - query is the fused absorbed pair `q = [q_nope_out | q_pe]` (concatenated here),
    passed as a single `q` to the indexer;
  - decode = `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` over the
    sparse block table the indexer produces.

Prefill is always dense (no sparsity branch), delegated to the composed
`TRTLLMMLABackend`; the cuda-graph capture/replay metadata is delegated too.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from vortex_torch.indexer.utils_sglang import get_decode_planner_trtllm

from vortex_torch.engine.sgl.compat import token_to_kv_pool
from .base import VortexMLABackendBase


from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
except Exception:  # mocked in docs / CPU envs
    trtllm_batch_decode_with_kv_cache_mla = None


@dataclass
class MLADecodeMetadata:
    # Index 0 = dense path; index 1 = sparse path (refreshed per layer).
    block_tables: List[torch.Tensor]
    seq_lens: List[torch.Tensor]
    bs: int


_mla_workspace_buffer = None


class VortexTRTLLMMLABackend(VortexMLABackendBase):
    """Standalone vortex sparse MLA backend (no inheritance from sglang's
    dense MLA backend)."""

    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__()
        self._init_mla_geometry(model_runner)
        sa = model_runner.server_args
        assert sa.page_size in (32, 64), (
            f"trtllm_mla requires page_size 32 or 64, got {sa.page_size}"
        )

        mc = model_runner.model_config
        self.qk_nope_head_dim = mc.qk_nope_head_dim   # 128 V2-Lite / 192 GLM-4.7-Flash
        # flashinfer's trtllm-gen MLA decode kernel asserts qk_nope_head_dim==128
        # but never uses it (head dims come from the 576-d fused query/kv layout);
        # pin the kernel arg to 128 so GLM-4.7-Flash (real 192) passes the assert.
        self.decode_qk_nope_head_dim = 128

        max_bs = model_runner.req_to_token_pool.size

        # Vortex tensors (managed here, mirroring trtllm.py) ----------------
        self.batch_table = torch.zeros(
            (sa.max_prefill_tokens,), dtype=torch.uint16, device=self.device
        )
        # prefill kv-indptr / qo-indptr (num_kv_heads=1)
        self.kv_indptr_prefill = torch.zeros((max_bs + 1,), dtype=torch.int32, device=self.device)
        self.qo_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=self.device),
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=self.device),
        ]
        self.max_blocks_per_seq = (self.max_context_len + self.block_size - 1) // self.block_size

        global _mla_workspace_buffer
        if _mla_workspace_buffer is None:
            _mla_workspace_buffer = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
        self.workspace_buffer = _mla_workspace_buffer

        # Planner (same factory as the MHA backend; num_kv_heads=1 in ctx) ----
        self.plan_decode = get_decode_planner_trtllm(sa.vortex_schedule_policy)

        # Dense MLA helper (COMPOSITION, not inheritance) — used only for the
        # non-vortex parts: prefill (always dense), cuda-graph capture/replay,
        # and dense decode on layers_skip layers. The sparse decode path below
        # is fully standalone (vortex ctx / batch_table / indexer).
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
        self._dense = TRTLLMMLABackend(model_runner)

        self._compile_indexer(model_runner)

        self.forward_metadata: Optional[MLADecodeMetadata] = None

    def forward_decode(
        self,
        q: torch.Tensor,                 # q_nope_out  [bs, H, kv_lora_rank]
        k: torch.Tensor,                 # kv_c
        v: torch.Tensor,                 # unused (MLA value == latent)
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,   # q_pe
        k_rope: Optional[torch.Tensor] = None,   # k_pe
        **kwargs,
    ):
        # Skipped layers run dense — delegate to the composed dense helper (it
        # writes the latent + decodes). No sparsity, no double write.
        if layer.layer_id in self.layers_skip:
            return self._dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache,
                q_rope=q_rope, k_rope=k_rope, **kwargs,
            )

        # Sparse path -------------------------------------------------------
        # 1) write the new token's latent into the single fused cache["latent"].
        if save_kv_cache and k is not None:
            token_to_kv_pool(forward_batch).set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc.to(torch.int64), k, k_rope
            )

        md = self.ctx.metadata
        # 2) indexer fills the sparse block table (topk middle); plan_decode
        #    prefilled BOS/EOS + sparse_seqlens. Query = fused [q_nope_out | q_pe].
        query = torch.cat([q, q_rope], dim=-1).contiguous()
        cache = self.vortex_cache(layer.layer_id)
        self.compiled_indexer.forward(
            q=query, o=md.sparse_block_tables, cache=cache, ctx=self.ctx,
        )

        # 3) MLA decode over the selected pages, on the fused latent.
        bs = q.shape[0]  # decode batch (one token/request); slice the preallocated metadata
        block_tables = md.sparse_block_tables[:bs]
        _pool = token_to_kv_pool(forward_batch)
        if getattr(_pool, "host_kv", False):
            # Host-resident latent: stage the selected blocks first. Reading the
            # pinned host latent directly does not fail, it just runs slowly enough
            # to trip sglang's 300 s forward watchdog. Full table in, sliced after,
            # so the remap buffer keeps one address across captured batch sizes.
            latent, full_remapped = _pool.fetch_latent(
                layer.layer_id, md.sparse_block_tables,
                row_lens=md.sparse_seqlens[:bs],
                num_rows=bs, max_per_row=block_tables.shape[1],
            )
            # The staging pool is BLOCK-granular ([capacity, block_size, dim]) while
            # this kernel indexes pages, so the two must coincide. base.py only
            # asserts page_size % block_size == 0, which is weaker.
            assert self.page_size == self.block_size, (
                f"vortex_host_kv_gb with trtllm_mla needs page_size == "
                f"vortex_block_size (staging is block-granular but the trtllm MLA "
                f"kernel indexes pages); got {self.page_size} vs {self.block_size}"
            )
            kv_cache = latent.view(-1, self.block_size, self.kv_cache_dim)
            block_tables = full_remapped[:bs]
        else:
            kv_cache = _pool.get_fused_latent_buffer(layer.layer_id)
        k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
        bmm1_scale = layer.scaling * k_scale
        o = trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1),                 # [bs, 1, H, 576]
            kv_cache=kv_cache.unsqueeze(1),           # [num_pages, 1, page, 576]
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.decode_qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=md.sparse_seqlens[:bs],
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
        )
        return o.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

