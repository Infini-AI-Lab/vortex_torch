from __future__ import annotations

"""
Vortex sparse-attention backend for MLA models on the **Triton** decode kernel.

This is the GLM-4.7-Flash-capable sibling of ``VortexTRTLLMMLABackend``. The
trtllm MLA kernels are geometry-locked to DeepSeek-R1/V2 (prefill asserts
192/128; decode FMHA rejects GLM's 20 heads); the flashinfer MLA backend
mis-handles GLM's `qk_nope=192 / v_head=256`. Triton handles arbitrary MLA
geometry, so the sparse decode here uses a custom **block-table** Triton kernel
(`triton_mla_kernel.decode_blocktable_mla`) fed the same
`get_decode_planner_trtllm` metadata (2D sparse_block_tables + sparse_seqlens)
the trtllm backend uses.

Structure mirrors ``trtllm_mla.py``:
  - extends sglang's base ``AttentionBackend`` (no inheritance from a dense MLA
    backend); manages its own vortex ``ctx``/metadata + indexer compile;
  - **composes** sglang's ``TritonAttnBackend`` (``self._dense``) for the
    non-vortex parts: prefill (always dense), cuda-graph capture/replay, and
    dense decode on ``layers_skip`` layers;
  - single shared KV head; KV is the fused latent from ``VortexMLACachePool``.

Calling convention difference vs the trtllm backend: ``triton`` is **not** in
``FORWARD_ABSORB_CORE_ATTENTION_BACKENDS``, so the model fuses the absorbed
query/key itself and calls ``forward_decode(q, k, v)`` with
``q = [q_nope_out | q_pe]`` (`[tokens, H, 576]`), ``k = [kv_c | k_pe]``
(`[tokens, 1, 576]`), ``v = kv_c`` (`[tokens, 1, 512]`) — no separate
``q_rope``/``k_rope`` kwargs.
"""
from typing import TYPE_CHECKING, Optional

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

from .triton_mla_kernel import decode_blocktable_mla

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class VortexTritonMLABackend(*attention_backend_base()):
    """Standalone vortex sparse MLA backend on the Triton decode kernel."""

    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__()
        # sglang >= 0.5.16 reads the KV / req pools off the *backend*
        # (forward_context.get_token_to_kv_pool); publish them here.
        publish_pools(self, model_runner)
        sa = model_runner.server_args

        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device

        mc = model_runner.model_config
        self.kv_lora_rank = mc.kv_lora_rank
        self.qk_rope_head_dim = mc.qk_rope_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim   # 576
        # indexer Context.create reads parent.head_dim — the fused absorbed
        # query [q_nope_out | q_pe] has width kv_cache_dim.
        self.head_dim = self.kv_cache_dim
        self.num_qo_heads = mc.num_attention_heads                      # tp handled by sglang
        self.num_kv_heads = 1
        self.group_size = self.num_qo_heads
        self.q_data_type = model_runner.dtype
        self.data_type = model_runner.kv_cache_dtype

        self.page_size = sa.page_size
        self.block_size = sa.vortex_block_size
        assert self.page_size % self.block_size == 0
        # The block-table kernel multiplies page id by block_size; require
        # page == block (one block per page) so a page id maps directly to a
        # contiguous block_size run of latent slots.
        assert self.page_size == self.block_size, (
            "VortexTritonMLABackend requires page_size == vortex_block_size "
            f"(got page_size={self.page_size}, block_size={self.block_size})."
        )
        self.layers_skip = sa.vortex_layers_skip
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # vortex sparse-decode metadata planner (block tables + seqlens).
        self.plan_decode = get_decode_planner_trtllm(sa.vortex_schedule_policy)

        # Dense Triton helper (COMPOSITION) for prefill / skipped-layer decode /
        # cuda-graph. It also owns the dense MLA metadata (token-level kv_indices).
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        self._dense = TritonAttnBackend(model_runner)

        # Indexer compile (ctx.metadata buffers live on the context).
        self.sparse_attention = model_runner.sparse_attention   # a vFlowMLA
        self.ctx = Context()
        self._compile(model_runner)

    # ------------------------------------------------------------------ #
    # indexer compilation (single fused query "q")
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
        self._dense.init_forward_metadata(forward_batch)
        if forward_batch.forward_mode.is_decode_or_idle():
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx,
            )

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
        q: torch.Tensor,                 # fused [q_nope_out | q_pe]  [tokens, H, 576]
        k: torch.Tensor,                 # fused [kv_c | k_pe]        [tokens, 1, 576]
        v: torch.Tensor,                 # kv_c                       [tokens, 1, 512]
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Skipped layers run dense — delegate to the composed Triton helper.
        if layer.layer_id in self.layers_skip:
            return self._dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs,
            )

        H = self.num_qo_heads
        # 1) write the new token's latent into the fused cache["latent"] (+ aux
        #    centroid refresh). The model already fused k = [kv_c | k_pe]; split
        #    it back into the (k_nope, k_rope) the pool's writer expects.
        if save_kv_cache and k is not None:
            k_f = k.view(-1, 1, self.kv_cache_dim)
            kv_c = k_f[..., : self.kv_lora_rank]
            k_pe = k_f[..., self.kv_lora_rank :]
            token_to_kv_pool(forward_batch).set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc.to(torch.int64), kv_c, k_pe,
            )

        md = self.ctx.metadata
        query = q.contiguous().view(-1, H, self.kv_cache_dim)   # [bs, H, 576]

        # 2) indexer fills the sparse block table (topk middle); plan_decode
        #    prefilled BOS/EOS + sparse_seqlens.
        cache = token_to_kv_pool(forward_batch).get_cache(layer.layer_id)
        self.compiled_indexer.forward(
            q=query, o=md.sparse_block_tables, cache=cache, ctx=self.ctx,
        )

        # 3) block-sparse MLA decode in Triton over the fused latent.
        bs = query.shape[0]
        latent = token_to_kv_pool(forward_batch).get_key_buffer(layer.layer_id).view(
            -1, self.kv_cache_dim
        )
        o = decode_blocktable_mla(
            q=query,
            latent=latent,
            block_table=md.sparse_block_tables[:bs],
            seqlens=md.sparse_seqlens[:bs],
            sm_scale=layer.scaling,
            block_size=self.block_size,
            kv_lora_rank=self.kv_lora_rank,
        )
        return o.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    # ------------------------------------------------------------------ #
    # prefill — always dense (no sparsity), delegated to the dense helper
    # ------------------------------------------------------------------ #
    def forward_extend(self, *args, **kwargs):
        return self._dense.forward_extend(*args, **kwargs)
