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

from vortex_torch.indexer.utils_sglang import get_decode_planner_trtllm

from vortex_torch.engine.sgl.compat import token_to_kv_pool
from ..config import cfg as _vortex_cfg
from .base import VortexMLABackendBase


from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from .triton_mla_kernel import decode_blocktable_mla

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class VortexTritonMLABackend(VortexMLABackendBase):
    """Standalone vortex sparse MLA backend on the Triton decode kernel."""

    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__()
        self._init_mla_geometry(model_runner)

        # The block-table kernel multiplies page id by block_size; require
        # page == block (one block per page) so a page id maps directly to a
        # contiguous block_size run of latent slots.
        assert self.page_size == self.block_size, (
            "VortexTritonMLABackend requires page_size == vortex_block_size "
            f"(got page_size={self.page_size}, block_size={self.block_size})."
        )

        # vortex sparse-decode metadata planner (block tables + seqlens).
        self.plan_decode = get_decode_planner_trtllm(
            model_runner.server_args.vortex_schedule_policy
        )

        # Dense Triton helper (COMPOSITION) for prefill / skipped-layer decode /
        # cuda-graph. It also owns the dense MLA metadata (token-level kv_indices).
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        self._dense = TritonAttnBackend(model_runner)

        # Host-resident KV is not supported on this backend. Unlike ``cuda_mla``,
        # this one delegates PREFILL to sglang's stock ``TritonAttnBackend``
        # (``base.forward_extend`` -> ``self._dense.forward_extend``), which reads
        # the latent buffer directly with no staging. With the latent in pinned host
        # memory that becomes a full per-layer prefix gather over PCIe inside the
        # attention kernel, and GLM-4.7-Flash prefills then stall past sglang's
        # 300 s forward watchdog — reproducibly, while the same flow passes at
        # n=20 and passes at n=100 with KV on the GPU (98%).
        #
        # Refusing is deliberate rather than a stopgap: decode staging alone was
        # verified correct here (20/20 with cuda graphs, 10/10 eager), so a
        # half-wired path would look like it works and fail only under load. Fixing
        # it means giving this backend the same ``MLAPrefill``-based extend override
        # cuda_mla has; until then ``cuda_mla`` (the default) is the supported
        # host-KV MLA backend.
        if float(getattr(_vortex_cfg(model_runner), "host_kv_gb", 0.0) or 0.0) > 0.0:
            raise NotImplementedError(
                "vortex_host_kv_gb is not supported with the triton MLA backend: "
                "its prefill is sglang's dense TritonAttnBackend, which reads the "
                "latent directly and would stream it from host memory per layer "
                "(observed: GLM prefill exceeding the 300 s forward watchdog). Use "
                "the default vortex_attention_backend/attn-backend 'cuda_mla' for "
                "host-resident KV, or set vortex_host_kv_gb=0 to keep KV on the GPU."
            )

        self._compile_indexer(model_runner)

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
        cache = self.vortex_cache(layer.layer_id)
        self.compiled_indexer.forward(
            q=query, o=md.sparse_block_tables, cache=cache, ctx=self.ctx,
        )

        # 3) block-sparse MLA decode in Triton over the fused latent.
        bs = query.shape[0]
        # No host-KV staging here: this backend refuses ``vortex_host_kv_gb`` at
        # construction (see __init__), because its prefill is sglang's dense
        # TritonAttnBackend and would stream the latent from host memory per layer.
        # Decode staging was verified working, so re-enabling it is a matter of
        # giving this backend cuda_mla's MLAPrefill-based extend override.
        block_table = md.sparse_block_tables[:bs]
        latent = token_to_kv_pool(forward_batch).get_key_buffer(layer.layer_id).view(
            -1, self.kv_cache_dim
        )
        o = decode_blocktable_mla(
            q=query,
            latent=latent,
            block_table=block_table,
            seqlens=md.sparse_seqlens[:bs],
            sm_scale=layer.scaling,
            block_size=self.block_size,
            kv_lora_rank=self.kv_lora_rank,
        )
        return o.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

