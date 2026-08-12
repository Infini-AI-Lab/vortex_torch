from __future__ import annotations

"""
Vortex sparse-attention backend for MLA models on the **hand-written CUDA**
decode kernel.

Structurally similar to ``VortexTritonMLABackend`` (``triton_mla.py``) — same
``get_decode_planner_trtllm`` metadata (2D ``sparse_block_tables`` +
``sparse_seqlens``), same indexer compile, composes sglang's ``TritonAttnBackend``
for skipped-layer decode / cuda-graph machinery. Two kernels are replaced with
hand-tuned / vendor paths:

* **decode** — the block-sparse MLA decode runs the from-scratch CUDA kernel
  ``cuda_mla_kernel.decode_blocktable_mla_cuda`` (ldmatrix + bf16-packed
  register-O + register softmax + split-KV; see ``cuda_mla/REPORT.md``), instead
  of the Triton ``decode_blocktable_mla``. Geometry-agnostic (NT=16 tiling), so
  any MLA head count works.
* **prefill** — the dense extend runs flashinfer's ragged FA3/cutlass kernel via
  ``mla_prefill.MLAPrefill`` (planned in ``init_forward_metadata``, run per
  layer), instead of ``TritonAttnBackend``'s ``_fwd_kernel``. On B200 (sm100)
  that Triton extend was ~30.7 ms vs flashinfer's ~2.2 ms for the identical
  192/128 MHA prefill — an ~14× kernel speedup that closes a ~4× end-to-end
  RULER gap, *and* it's a correctness fix: the Triton extend cannot read this
  backend's 576-wide latent KV pool (a 192-wide kernel against a 576-wide
  buffer), so the delegated path produced garbage. Chunked prefill (prefix>0) is
  handled by reconstructing per-head k/v from the latent + ``merge_state``.

Calling convention: ``cuda_mla`` is NOT in
``FORWARD_ABSORB_CORE_ATTENTION_BACKENDS``, so for **decode** the model fuses the
absorbed query/key and calls ``forward_decode(q, k, v)`` with
``q = [q_nope_out | q_pe]`` (`[tokens, H, 576]`), ``k = [kv_c | k_pe]``
(`[tokens, 1, 576]`), ``v = kv_c``. For **prefill** a registered dispatch handler
routes every extend batch to the per-head MHA path (q/k/v `[T,H,192/192/128]`),
which ``forward_extend`` serves with ``MLAPrefill``.

Calling convention is the same as the Triton MLA backend: ``cuda_mla`` is NOT in
``FORWARD_ABSORB_CORE_ATTENTION_BACKENDS``, so the model fuses the absorbed
query/key and calls ``forward_decode(q, k, v)`` with ``q = [q_nope_out | q_pe]``
(`[tokens, H, 576]`), ``k = [kv_c | k_pe]`` (`[tokens, 1, 576]`), ``v = kv_c``.
"""
from typing import TYPE_CHECKING

import torch

from vortex_torch.indexer.utils_sglang import get_decode_planner_trtllm

from vortex_torch.engine.sgl.compat import token_to_kv_pool
from .base import VortexMLABackendBase


from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from .cuda_mla_kernel import allocate_mla_buffers, make_mla_decoder
from .mla_prefill import MLAPrefill

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


# ---------------------------------------------------------------------------- #
# Dispatch handler: route the model's MLA attention to the per-head MHA prefill
# for *every* extend batch (not just prefix==0). The default (unregistered)
# backend falls back to the "triton" handler, which sends prefix>0 extends to the
# absorbed-MLA path — but our forward_extend handles the prefix itself (read the
# latent buffer + merge_state, see MLAPrefill), so we must keep the per-head MHA
# path for prefix>0 too. Decode and speculative paths still use absorbed MLA.
# ---------------------------------------------------------------------------- #
def _register_cuda_mla_dispatch() -> None:
    try:
        from sglang.srt.models.deepseek_common.attention_backend_handler import (
            AttentionBackendRegistry,
            _dispatch_mla_subtype,
        )
        from sglang.srt.compilation.piecewise_context_manager import (
            is_in_piecewise_cuda_graph,
        )
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
            AttnForwardMethod,
        )
    except Exception:
        # Older/newer sglang layouts — best-effort; if the registry isn't where we
        # expect, the default triton handler is used (prefix==0 fast path still works).
        return

    def _handle_attention_cuda_mla(attn, forward_batch):
        if is_in_piecewise_cuda_graph():
            return AttnForwardMethod.MLA
        if get_global_server_args().enable_deterministic_inference:
            return _dispatch_mla_subtype(attn, forward_batch)
        if forward_batch.forward_mode.is_extend_without_speculative():
            return AttnForwardMethod.MHA
        return _dispatch_mla_subtype(attn, forward_batch)

    AttentionBackendRegistry.register("cuda_mla", _handle_attention_cuda_mla)


_register_cuda_mla_dispatch()


class VortexCudaMLABackend(VortexMLABackendBase):
    """Standalone vortex sparse MLA backend on the hand-written CUDA decode kernel."""

    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__()
        self._init_mla_geometry(model_runner)
        sa = model_runner.server_args

        # The block-table kernel multiplies page id by block_size; require
        # page == block (one block per page) so a page id maps directly to a
        # contiguous block_size run of latent slots.
        assert self.page_size == self.block_size, (
            "VortexCudaMLABackend requires page_size == vortex_block_size "
            f"(got page_size={self.page_size}, block_size={self.block_size})."
        )
        assert self.block_size in (16, 32, 64), (
            f"VortexCudaMLABackend supports block_size in {{16,32,64}}, got {self.block_size}."
        )

        # vortex sparse-decode metadata planner (block tables + seqlens).
        self.plan_decode = get_decode_planner_trtllm(sa.vortex_schedule_policy)

        # Dense Triton helper (COMPOSITION) for prefill / skipped-layer decode /
        # cuda-graph. It also owns the dense MLA metadata (token-level kv_indices).
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        self._dense = TritonAttnBackend(model_runner)

        self._compile_indexer(model_runner)

        # flashinfer-style plan/run decoders, one per decode batch size (allocated
        # lazily; cuda-graph captures create their bs's decoder at capture time and
        # reuse it on replay — buffers stay at fixed addresses). `plan()` runs once
        # per step in init_forward_metadata*, `run()` per layer in forward_decode.
        #
        # The work-queue + split-reduction scratch (work_*, mid_*) is allocated ONCE
        # here, sized for the MAX decode batch size, and shared across every per-bs
        # decoder (each slices it to its own target_ctas). target_ctas is monotone in
        # bs, so the max-bs buffers cover every smaller bs. Previously each captured
        # cuda-graph bs built a decoder that allocated its own full
        # target_ctas*M*512 fp32 mid_o + work queue => O(#graph-bs) redundant scratch;
        # now there is exactly one copy. max_bs = req_to_token_pool.size (the running
        # request cap, set on ctx during _compile) — a safe upper bound on decode bs.
        self._max_bs = int(self.ctx.max_bs)
        max_blocks = self.ctx.metadata.sparse_block_tables.size(1)
        self._mla_buffers = allocate_mla_buffers(
            self._max_bs, self.num_qo_heads, self.block_size, max_blocks, self.device,
        )
        self._decoders: dict = {}
        self._cur_decoder = None

        # flashinfer-backed dense prefill (FA3/cutlass sm100), replacing the
        # ~8×-slower Triton extend kernel (see mla_prefill.MLAPrefill). plan()
        # runs once per extend batch in init_forward_metadata; it needs the
        # softmax scale + head dims, uniform across MLA layers, so we capture them
        # from the loaded model here (the model's *computed* scaling — avoids
        # re-deriving the yarn mscale). run() executes per layer. Required (no
        # fallback): the composed Triton extend cannot read this backend's
        # 576-wide latent KV pool with a 192-wide kernel, so there is no correct
        # dense alternative.
        self._prefill_has_prefix = False
        self._mla = self._capture_mla_params(model_runner)
        self._prefill = MLAPrefill(self.device)

    def _capture_mla_params(self, model_runner) -> dict:
        """Read the uniform MLA prefill geometry + softmax scale from the loaded
        model so plan() can run in init_forward_metadata (before any layer is
        seen). Reuses the model's own computed ``scaling`` (yarn mscale folded
        in). Raises if no MLA attention module is found."""
        model = getattr(model_runner, "model", None)
        if model is None:
            raise RuntimeError("model not available at backend init")
        for m in model.modules():
            if (
                hasattr(m, "attn_mha")
                and hasattr(m, "scaling")
                and hasattr(m, "qk_nope_head_dim")
                and hasattr(m, "qk_rope_head_dim")
                and hasattr(m, "v_head_dim")
            ):
                return {
                    "scale": float(m.scaling),
                    "num_heads": int(getattr(m, "num_local_heads", self.num_qo_heads)),
                    "qk_head_dim": int(m.qk_nope_head_dim + m.qk_rope_head_dim),
                    "qk_nope_head_dim": int(m.qk_nope_head_dim),
                    "qk_rope_head_dim": int(m.qk_rope_head_dim),
                    "v_head_dim": int(m.v_head_dim),
                    "logit_cap": float(getattr(m.attn_mha, "logit_cap", 0.0) or 0.0),
                }
        raise RuntimeError("no MLA attention module found in model")

    def _decoder_for(self, bs: int):
        dec = self._decoders.get(bs)
        if dec is None:
            assert bs <= self._max_bs, (
                f"decode bs {bs} exceeds max_bs {self._max_bs} the MLA scratch was "
                f"sized for (req_to_token_pool.size)."
            )
            max_blocks = self.ctx.metadata.sparse_block_tables.size(1)
            dec = make_mla_decoder(
                bs, self.num_qo_heads, self.block_size, max_blocks, self._mla_buffers,
            )
            self._decoders[bs] = dec
        return dec

    def _plan(self, seq_lens: torch.Tensor) -> None:
        """Build the load-balanced work queue once for this decode step (shared by
        every layer's run()). sparse_seqlens were just filled by plan_decode."""
        bs = seq_lens.shape[0]
        dec = self._decoder_for(bs)
        dec.plan(self.ctx.metadata.sparse_seqlens[:bs])
        self._cur_decoder = dec

    # ------------------------------------------------------------------ #
    # per-batch hooks (the base owns the eager / capture / replay plumbing)
    # ------------------------------------------------------------------ #
    def _after_plan_decode(self, seq_lens: torch.Tensor) -> None:
        """Build the load-balanced work queue for this decode step.

        Runs after plan_decode, which is what fills the ``sparse_seqlens`` the
        queue is balanced over. The base also advances the host-KV staging
        generation -- once per step, before any layer, because the generation is
        what pins a staging slot for the whole step.
        """
        super()._after_plan_decode(seq_lens)     # advances the host-KV generation
        self._plan(seq_lens)

    def _init_extend_metadata(self, forward_batch) -> None:

        """Plan the flashinfer prefill wrappers once per extend batch.

        Shared by every layer; ``forward_extend`` then just ``run()``s per layer.
        """
        fm = self._dense.forward_metadata
        kv_indices = getattr(fm, "kv_indices", None)
        has_prefix = kv_indices is not None and kv_indices.numel() > 0
        self._prefill_has_prefix = has_prefix
        self._prefill.plan(
            qo_indptr=fm.qo_indptr,
            kv_indptr_prefix=getattr(fm, "kv_indptr", None),
            num_heads=self._mla["num_heads"],
            qk_head_dim=self._mla["qk_head_dim"],
            v_head_dim=self._mla["v_head_dim"],
            qk_nope_head_dim=self._mla["qk_nope_head_dim"],
            qk_rope_head_dim=self._mla["qk_rope_head_dim"],
            kv_lora_rank=self.kv_lora_rank,
            sm_scale=self._mla["scale"],
            q_dtype=self.q_data_type,
            logits_soft_cap=self._mla["logit_cap"],
            has_prefix=has_prefix,
        )

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
        # The (once-compiled) indexer runs for every decode layer; pass the
        # active global layer id as the EXPLICIT trailing ``cur_layer`` arg so
        # per-layer-weight ops (e.g. the Parameter in learned_block_sparse_mla)
        # select the active layer's baked weight slice at runtime. The arg
        # defaults to 0 in the generated forward(), so every other flow that
        # doesn't pass it is unaffected.
        cache = self.vortex_cache(layer.layer_id)
        self.compiled_indexer.forward(
            q=query, o=md.sparse_block_tables, cache=cache, ctx=self.ctx,
            cur_layer=layer.layer_id,
        )

        # 3) block-sparse MLA decode in CUDA over the fused latent. The work queue
        #    was built once for this step by _plan() (in init_forward_metadata); run()
        #    just consumes it with this layer's block table => one plan, all layers.
        bs = query.shape[0]
        block_tables = md.sparse_block_tables[:bs]
        _pool = token_to_kv_pool(forward_batch)
        if getattr(_pool, "host_kv", False):
            # Host-resident latent: stage the blocks the indexer just selected into
            # the GPU pool and rewrite the block table to staging slots. Between the
            # indexer (which names the blocks) and the decoder (which reads them).
            # Full table, sliced after — see the trtllm note: the remap buffer's
            # address must not change between captured batch sizes.
            latent, full_remapped = _pool.fetch_latent(
                layer.layer_id, md.sparse_block_tables,
                row_lens=md.sparse_seqlens[:bs],
                num_rows=bs, max_per_row=block_tables.shape[1],
            )
            block_tables = full_remapped[:bs]
            latent = latent.view(-1, self.kv_cache_dim)
        else:
            latent = _pool.get_key_buffer(layer.layer_id).view(-1, self.kv_cache_dim)
        o = query.new_empty((bs, self.num_qo_heads, self.kv_lora_rank))
        self._cur_decoder.run(
            query, latent, block_tables, o, layer.scaling,
        )
        return o.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    # ------------------------------------------------------------------ #
    # prefill — dense (no sparsity). Fast flashinfer path (FA3/cutlass sm100)
    # via MLAPrefill, functionally identical to TritonAttnBackend's extend; the
    # Triton helper is the fallback when the fast path is unavailable.

    def _host_prefix(self, key_buffer, kv_indices):
        """Gather the host-resident prefix tokens onto the device.

        The reconstruction gathers TOKENS out of the latent with ``index_select``,
        which cannot mix a device index with a host source. So copy the needed
        tokens over first and return ``(compacted_buffer, 0..P-1 indices)``.

        **Not cached.** A previous version cached the gather per extend batch, keyed
        on the prefix length, to avoid repeating it for each of the 47 layers. That
        is wrong: ``forward_extend`` is called per layer *and* per request, so two
        requests whose prefixes happen to be the same length reused the first one's
        gathered KV — no error, just plausible output and RULER accuracy collapsing
        from 100% to ~25%. Keying on the contents instead would need a ``.item()``
        read (a device sync per layer), which costs more than the gather it saves.
        The gather itself is cheap: measured 0.10 ms for a 4k-token prefix, ~4.9 ms
        across all 47 layers, against a step budget of hundreds of ms.

        The destination buffer *is* reused across calls, which is where the
        allocation cost went; only the copy repeats.
        """
        from ..host_kv import gather_host_tokens

        n = kv_indices.numel()
        flat = key_buffer.view(key_buffer.shape[0], -1)
        out = getattr(self, "_host_prefix_buf", None)
        if out is None or out.shape[0] < n or out.shape[1] != flat.shape[1]:
            out = torch.empty((max(n, 1), flat.shape[1]),
                              dtype=flat.dtype, device=kv_indices.device)
            self._host_prefix_buf = out
        gathered = gather_host_tokens(flat, kv_indices, out=out[:n])
        ar = getattr(self, "_host_prefix_ar", None)
        if ar is None or ar.numel() < n:
            ar = torch.arange(max(n, 1), device=kv_indices.device, dtype=kv_indices.dtype)
            self._host_prefix_ar = ar
        return gathered.unsqueeze(1), ar[:n]

    # ------------------------------------------------------------------ #
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        **kwargs,
    ):
        fm = self._dense.forward_metadata
        kv_indices = getattr(fm, "kv_indices", None)
        key_buffer = None
        if self._prefill_has_prefix:
            _pool = token_to_kv_pool(forward_batch)
            key_buffer = _pool.get_key_buffer(layer.layer_id)
            if getattr(_pool, "host_kv", False):
                # The prefix reconstruction gathers TOKENS out of the latent with
                # index_select, which cannot mix a device index with a host source
                # ("index is on cuda:0, different from other tensors on cpu"). Copy
                # just those tokens to the device first and hand the reconstruction
                # a compacted buffer, with the indices rewritten to 0..P-1.
                #
                # Token-granular, so this does NOT go through the block staging
                # pool: a radix prefix is read once here, and promoting whole blocks
                # would move more data than the tokens actually needed.
                #
                # Cached PER STEP, not per layer. The reconstruction runs once per
                # layer over the SAME prefix tokens, so gathering inside the layer
                # loop re-copies the whole prefix 47x on GLM-4.7-Flash (measured
                # 7-21 ms and 0.1-0.4 GB of redundant PCIe traffic per prefill) and
                # allocates two tensors each time. The gather is keyed on the
                # indices' identity + length, which is what changes between steps.
                key_buffer, kv_indices = self._host_prefix(key_buffer, kv_indices)
        return self._prefill.run(
            q, k, v,
            kv_indices_prefix=kv_indices,
            key_buffer=key_buffer,
            kv_b_proj=getattr(layer, "kv_b_proj", None),
        )
