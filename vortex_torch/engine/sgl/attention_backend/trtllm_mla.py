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

from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.indexer import Context, MetaData
from vortex_torch.indexer.compiler.compile import compile as compile_indexer
from vortex_torch.indexer.utils_sglang import get_decode_planner_trtllm

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from vortex_torch.engine.sgl.compat import (
    GraphMetadataArgs,
    attention_backend_base,
    capture_dense,
    publish_pools,
    replay_dense,
    token_to_kv_pool,
)
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


class VortexTRTLLMMLABackend(*attention_backend_base()):
    """Standalone vortex sparse MLA backend (no inheritance from sglang's
    dense MLA backend)."""

    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__()
        # sglang >= 0.5.16 reads the KV / req pools off the *backend*
        # (forward_context.get_token_to_kv_pool); publish them here.
        publish_pools(self, model_runner)
        sa = model_runner.server_args
        assert sa.page_size in (32, 64), (
            f"trtllm_mla requires page_size 32 or 64, got {sa.page_size}"
        )

        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device

        # Latent geometry — single shared KV head.
        mc = model_runner.model_config
        self.kv_lora_rank = mc.kv_lora_rank
        self.qk_rope_head_dim = mc.qk_rope_head_dim
        self.qk_nope_head_dim = mc.qk_nope_head_dim   # 128 V2-Lite / 192 GLM-4.7-Flash
        # flashinfer's trtllm-gen MLA decode kernel asserts qk_nope_head_dim==128
        # but never uses it (head dims come from the 576-d fused query/kv layout);
        # pin the kernel arg to 128 so GLM-4.7-Flash (real 192) passes the assert.
        self.decode_qk_nope_head_dim = 128
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim   # 576
        # indexer Context.create reads parent.head_dim — for MLA the indexer
        # query is the fused [q_nope_out | q_pe] of width kv_cache_dim.
        self.head_dim = self.kv_cache_dim
        self.num_qo_heads = mc.num_attention_heads // 1                 # tp by sglang
        self.num_kv_heads = 1                                           # MLA: one latent head
        self.group_size = self.num_qo_heads
        self.q_data_type = model_runner.dtype
        # the model reads attn_backend.data_type (e.g. _fuse_rope_for_trtllm_mla);
        # bf16 here ⇒ no fp8 in-kernel rope fusion (the path we support for now).
        self.data_type = model_runner.kv_cache_dtype

        self.page_size = sa.page_size
        self.block_size = sa.vortex_block_size
        assert self.page_size % self.block_size == 0
        self.num_blocks_per_page = self.page_size // self.block_size
        self.layers_skip = sa.vortex_layers_skip
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
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

        # Indexer compile (ctx.metadata buffers live on the context) ---------
        self.sparse_attention = model_runner.sparse_attention   # a vFlowMLA
        self.ctx = Context()
        self._compile(model_runner)

        self.forward_metadata: Optional[MLADecodeMetadata] = None

    # ------------------------------------------------------------------ #
    # indexer compilation (single fused query "q"; standalone-verified path)
    # ------------------------------------------------------------------ #
    def _compile(self, model_runner) -> None:
        device = model_runner.device
        indexer = self.sparse_attention.forward_indexer

        self.ctx.create(self, model_runner)
        self.ctx.metadata = MetaData.preallocate(self.ctx, device=device)
        self.ctx.assert_created()
        self.ctx.profile()

        def register(vt, name):
            self.ctx.tensor_list.append(vt)
            self.ctx.output_tensor_to_op_list.append(None)
            self.ctx.tensor_id_to_tensor_name_map[vt.tensor_id] = name

        def make_dummy(shape, fmt, tid, tdtype=None, zeros=False):
            factory = torch.zeros if zeros else torch.empty
            return as_vtensor(
                factory(shape, device=device, dtype=tdtype or self.q_data_type),
                fmt, tensor_id=tid,
            )

        with torch.no_grad():
            # fused absorbed query of inner dim kv_cache_dim (= [q_nope_out | q_pe])
            q_dummy = make_dummy((0, self.group_size, self.kv_cache_dim), FORMAT.BATCHED, 0)
            register(q_dummy, "q")
            o_dummy = make_dummy((0, 1, 1), FORMAT.RAGGED, 1)
            register(o_dummy, "o")
            cache_dummy = {}
            for i, (name, (shape, cdt)) in enumerate(
                self.sparse_attention.get_cache_meta_info().items()
            ):
                vt = make_dummy((0, shape[0], shape[1]), FORMAT.PAGED, 2 + i, tdtype=cdt, zeros=True)
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")
            indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)

        self.compiled_indexer = compile_indexer(self.ctx)()
        self.ctx.summary()
        self.ctx.execute()

    # ------------------------------------------------------------------ #
    # per-batch metadata
    # ------------------------------------------------------------------ #
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Dense metadata (block_kv_indices etc.) for prefill + skipped-layer
        # dense decode — delegated to the composed dense MLA helper.
        self._dense.init_forward_metadata(forward_batch)
        # Vortex sparse-decode metadata: plan_decode fills ctx.metadata.
        if forward_batch.forward_mode.is_decode_or_idle():
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx,
            )

    # cuda-graph: delegate the dense graph machinery to the composed helper
    # (it owns the block_kv_indices buffers for prefill/skipped-layer decode),
    # and also fill the vortex sparse metadata (ctx.metadata) via plan_decode so
    # captured/replayed decode graphs route correctly through the indexer +
    # sparse decode. plan_decode writes the (fixed-address) preallocated buffers.
    def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf=None):
        self._dense.init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    def init_forward_metadata_capture_cuda_graph(
        self, bs, num_tokens, req_pool_indices, seq_lens, encoder_lens,
        forward_mode, spec_info,
    ):
        capture_dense(
            self._dense,
            GraphMetadataArgs(
                bs=bs, req_pool_indices=req_pool_indices, seq_lens=seq_lens,
                forward_mode=forward_mode, encoder_lens=encoder_lens,
                spec_info=spec_info,
            ),
        )
        if forward_mode.is_decode_or_idle():
            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices, ctx=self.ctx,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens,
        forward_mode, spec_info, seq_lens_cpu,
    ):
        replay_dense(
            self._dense,
            GraphMetadataArgs(
                bs=bs, req_pool_indices=req_pool_indices, seq_lens=seq_lens,
                forward_mode=forward_mode, encoder_lens=encoder_lens,
                spec_info=spec_info, seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=seq_lens_sum,
            ),
        )
        if forward_mode.is_decode_or_idle():
            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices, ctx=self.ctx,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self._dense.get_cuda_graph_seq_len_fill_value()

    # ------------------------------------------------------------------ #
    # decode (sparse for non-skipped layers; dense otherwise)
    # ------------------------------------------------------------------ #
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
        kv_cache = token_to_kv_pool(forward_batch).get_fused_latent_buffer(layer.layer_id)
        k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
        bmm1_scale = layer.scaling * k_scale
        o = trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1),                 # [bs, 1, H, 576]
            kv_cache=kv_cache.unsqueeze(1),           # [num_pages, 1, page, 576]
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.decode_qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=md.sparse_block_tables[:bs],
            seq_lens=md.sparse_seqlens[:bs],
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
        )
        return o.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    # ------------------------------------------------------------------ #
    # prefill — always dense (no sparsity), delegated to the dense helper
    # ------------------------------------------------------------------ #
    def forward_extend(self, *args, **kwargs):
        return self._dense.forward_extend(*args, **kwargs)
