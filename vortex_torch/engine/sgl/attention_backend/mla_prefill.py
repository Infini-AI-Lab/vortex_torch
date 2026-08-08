from __future__ import annotations

"""
Fast dense MLA **prefill** (extend) attention — a flashinfer-backed replacement
for sglang's Triton ``extend_attention_fwd``.

Why this exists
---------------
The vortex MLA backends (``cuda_mla`` / ``triton_mla``) compose sglang's
``TritonAttnBackend`` purely for the dense prefill. Profiling the RULER run on
B200 (sm100) showed the prefill — not the sparse decode — is the throughput
bottleneck: the Triton extend kernel ``_fwd_kernel`` runs **~30.7 ms** versus
the cutlass ``fmhaSm100fKernel`` at **~3.8 ms** for the *identical* problem
(q ``[T,16,192]``, k ``[T,16,192]``, v ``[T,16,128]``, 27 layers). Same shape,
same FLOPs, same KV read — Triton's generic kernel is simply ~8× slower than the
hand-tuned cutlass/FA3 prefill on Blackwell. Over a 100-prompt RULER prefill
(~392K tokens) that is the fixed ~10 s gap that made ``cuda_mla`` ≈4× slower
end-to-end than ``trtllm_mla``.

Functional contract — same as ``extend_attention_fwd``
------------------------------------------------------
``extend_attention_fwd(q_extend, k_extend, v_extend, o, k_buffer, v_buffer,
qo_indptr, kv_indptr, kv_indices, ...)`` computes, per request, the causal
attention of the *extend* queries against ``[prefix-from-buffer ++ extend]``.
``MLAPrefill.forward_extend`` reproduces this in two flashinfer calls + an
LSE merge (the pattern the user pointed to, and exactly what sglang's own
``_chunked_prefix_attn_mha`` / ``merge_state_v2`` does):

1. **Extend part** — ragged self-attention of the extend q/k/v, ``causal=True``.
   Returns ``(o_e, lse_e)`` when a prefix is present, else just ``o_e``.
2. **Prefix part** (only when ``kv_indices`` is non-empty) — the prefix lives in
   the MLA latent pool as ``[P, 1, kv_lora_rank + qk_rope]`` (an MQA latent, not
   per-head k/v). We reconstruct per-head k/v exactly as the model does for
   chunked prefill: ``kv = kv_b_proj(kv_a_normed)`` → ``k_nope | v``, and
   ``k = [k_nope | k_pe]`` with the shared rope key broadcast across heads. The
   stored ``kv_a`` is already ``kv_a_layernorm``'d and ``k_pe`` is post-rope, so
   no extra normalization is applied. Then a ragged **cross**-attention of the
   extend q against the prefix k/v, ``causal=False`` (the whole prefix is
   visible). Returns ``(o_p, lse_p)``.
3. **Merge** — ``flashinfer.merge_state(o_e, lse_e, o_p, lse_p)`` combines the
   two partial softmaxes into the final output.

For RULER / AIME24 the prompts (~3.9K tokens) are far below the chunked-prefill
size, so ``kv_indices`` is empty and only step 1 runs — the hot path that buys
the ~8× kernel speedup. Steps 2–3 make chunked prefill (prefix>0) correct, which
the previous delegate-to-Triton path could not even do for an MLA latent pool
(a 192-wide kernel cannot read a 576-wide latent buffer).

Plan/run: two ragged wrappers (extend self-attn, prefix cross-attn) are
``plan()``'d once per extend batch from the backend's ``init_forward_metadata``
and ``run()`` per layer, since geometry and softmax scale are uniform across the
27 MLA layers. Mirrors the decode ``MLADecoder``'s plan/run split.
"""

from typing import Optional

import torch


class MLAPrefill:
    """flashinfer ragged-KV (FA3/cutlass sm100) dense MLA prefill with prefix
    handled via reconstruct-from-latent + ``merge_state``."""

    def __init__(self, device, *, kv_layout: str = "NHD", workspace_mb: int = 128):
        import flashinfer

        self._fi = flashinfer
        # merge_state lives at flashinfer.merge_state (re-exported from cascade).
        self._merge_state = getattr(flashinfer, "merge_state", None)
        if self._merge_state is None:
            from flashinfer.cascade import merge_state as _ms
            self._merge_state = _ms

        # One reusable float workspace shared by both wrappers (flashinfer
        # recommends ~128 MB for the split-K scratch).
        self._workspace = torch.empty(
            workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device
        )
        Wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper
        self._extend = Wrapper(self._workspace, kv_layout)   # causal self-attn
        self._prefix = Wrapper(self._workspace, kv_layout)   # non-causal cross-attn
        self._has_prefix = False                             # set by plan()

    # ------------------------------------------------------------------ #
    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr_prefix: Optional[torch.Tensor],
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        sm_scale: float,
        q_dtype: torch.dtype,
        logits_soft_cap: float,
        has_prefix: bool,
    ) -> None:
        """Plan both wrappers for this extend batch. Called once per batch from
        the backend's ``init_forward_metadata``; ``run()`` then reuses the plan
        across all (geometry-uniform) layers. Geometry is stashed so ``run()``
        needs only the per-layer tensors."""
        self._H = num_heads
        self._dqk = qk_head_dim
        self._dv = v_head_dim
        self._qk_nope = qk_nope_head_dim
        self._qk_rope = qk_rope_head_dim
        self._kv_lora = kv_lora_rank
        self._q_dtype = q_dtype
        self._has_prefix = has_prefix

        # flashinfer's prefill plan() requires int32 indptrs. sglang widened
        # TritonAttnBackend.qo_indptr from int32 to int64 in 0.5.16 (kv_indptr is
        # still int32), and since we forward the dense backend's buffers straight
        # through, an int64 qo_indptr gets reinterpreted as int32 inside
        # PrefillSplitQOKVIndptr -> bogus offsets ("qo_indptr[i] - qo_indptr[i-1]
        # should be non-negative"). Coerce instead of trusting the dtype.
        if qo_indptr.dtype != torch.int32:
            qo_indptr = qo_indptr.to(torch.int32)
        if kv_indptr_prefix is not None and kv_indptr_prefix.dtype != torch.int32:
            kv_indptr_prefix = kv_indptr_prefix.to(torch.int32)

        # Extend part: keys are the query tokens themselves (kv_indptr == qo_indptr),
        # plain MHA (num_kv_heads == num_qo_heads), causal.
        self._extend.plan(
            qo_indptr,
            qo_indptr,
            num_heads,
            num_heads,
            qk_head_dim,
            head_dim_vo=v_head_dim,
            causal=True,
            sm_scale=sm_scale,
            logits_soft_cap=logits_soft_cap or 0.0,
            q_data_type=q_dtype,
            kv_data_type=q_dtype,
        )
        # Prefix part: extend queries attend the (shorter) cached prefix,
        # fully visible (causal=False), separate kv_indptr.
        if has_prefix:
            self._prefix.plan(
                qo_indptr,
                kv_indptr_prefix,
                num_heads,
                num_heads,
                qk_head_dim,
                head_dim_vo=v_head_dim,
                causal=False,
                sm_scale=sm_scale,
                logits_soft_cap=logits_soft_cap or 0.0,
                q_data_type=q_dtype,
                kv_data_type=q_dtype,
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconstruct_prefix_kv(
        key_buffer: torch.Tensor,      # [N, 1, kv_lora_rank + qk_rope_head_dim]
        kv_indices: torch.Tensor,      # [P] prefix token locations in the pool
        kv_b_proj,                     # ColumnParallelLinear: kv_lora_rank -> H*(qk_nope+v)
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
    ):
        """Rebuild per-head prefix k/v from the stored MLA latent — identical to
        the model's ``_chunked_prefix_attn_mha`` reconstruction."""
        lat = key_buffer.view(key_buffer.shape[0], -1).index_select(0, kv_indices)
        kv_a = lat[:, :kv_lora_rank]                      # already kv_a_layernorm'd
        k_pe = lat[:, kv_lora_rank:]                      # post-rope, shared across heads
        kv = kv_b_proj(kv_a.to(dtype))[0]
        kv = kv.view(-1, num_heads, qk_nope_head_dim + v_head_dim)
        k_nope = kv[..., :qk_nope_head_dim]
        v = kv[..., qk_nope_head_dim:].contiguous()
        k = torch.empty(
            (kv.shape[0], num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=v.dtype,
            device=v.device,
        )
        k[..., :qk_nope_head_dim] = k_nope
        k[..., qk_nope_head_dim:] = k_pe.unsqueeze(1)     # broadcast rope key over heads
        return k, v

    # ------------------------------------------------------------------ #
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        # prefix (chunked-prefill) inputs — only consulted when the batch was
        # planned with has_prefix=True:
        kv_indices_prefix: Optional[torch.Tensor] = None,
        key_buffer: Optional[torch.Tensor] = None,
        kv_b_proj=None,
    ) -> torch.Tensor:
        """Run the planned prefill for one layer. Returns ``[T, num_heads *
        v_head_dim]`` — same layout as ``TritonAttnBackend.forward_extend``."""
        H, dqk, dv = self._H, self._dqk, self._dv
        qv = q.contiguous().view(-1, H, dqk)
        kv_ = k.contiguous().view(-1, H, dqk)
        vv = v.contiguous().view(-1, H, dv)

        if not self._has_prefix:
            o = self._extend.run(qv, kv_, vv, return_lse=False)
            return o.reshape(-1, H * dv)

        # Extend (causal) + prefix (non-causal) partial softmaxes, merged by LSE.
        o_e, lse_e = self._extend.run(qv, kv_, vv, return_lse=True)
        k_p, v_p = self._reconstruct_prefix_kv(
            key_buffer, kv_indices_prefix, kv_b_proj, H,
            self._qk_nope, self._qk_rope, dv, self._kv_lora, self._q_dtype,
        )
        o_p, lse_p = self._prefix.run(qv, k_p, v_p, return_lse=True)
        o, _ = self._merge_state(o_e, lse_e, o_p, lse_p)
        return o.reshape(-1, H * dv)
