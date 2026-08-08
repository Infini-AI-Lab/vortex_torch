"""Version shims for the handful of sglang APIs that moved between releases.

vortex tracks a vendored sglang (``third_party/sglang/v*``). Upstream
occasionally relocates a symbol or reshapes an interface vortex implements;
rather than scatter ``try/except ImportError`` at every call site, resolve it
once here.

Keep this module import-light: it is pulled in from attention-backend module
scope, which runs inside the spawned scheduler worker.
"""
from __future__ import annotations

from typing import Optional

import torch


def get_attention_tp_size() -> int:
    """Attention tensor-parallel world size.

    * sglang <= 0.5.9 — ``sglang.srt.layers.dp_attention.get_attention_tp_size``
    * sglang >= 0.5.16 — the function was removed; the value now lives on the
      parallel context as ``runtime_context.get_parallel().attn_tp_size``.
    """
    try:
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_size as _legacy,
        )
    except ImportError:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().attn_tp_size
    return _legacy()


def token_to_kv_pool(forward_batch):
    """The KV pool for this forward.

    * sglang <= 0.5.9 — carried on the batch as ``forward_batch.token_to_kv_pool``.
    * sglang >= 0.5.16 — the field was dropped from ``ForwardBatch``; the pool is
      reached through ``forward_context.get_token_to_kv_pool()``, which reads
      ``get_attn_backend().token_to_kv_pool``.

    vortex's backends therefore publish ``self.token_to_kv_pool`` at init (see
    ``bind_kv_pool``) so the upstream accessor resolves for them too; this helper
    just prefers whichever of the two is available.
    """
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is not None:
        return pool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    return get_token_to_kv_pool()


def bind_kv_pool(backend, model_runner) -> None:
    """Publish the KV pool on ``backend`` for sglang >= 0.5.16.

    ``forward_context.get_token_to_kv_pool()`` (used by upstream code such as
    ``models/utils.py::enable_fused_set_kv_buffer``) reads
    ``get_attn_backend().token_to_kv_pool``, so every backend must expose it.
    Harmless on 0.5.9, where nothing reads the attribute.
    """
    backend.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    backend.req_to_token_pool = getattr(model_runner, "req_to_token_pool", None)


def is_draft_extend(forward_mode) -> bool:
    """True when this forward is a speculative draft-extend of either generation.

    vortex supports neither, so its backends assert on this.

    * sglang <= 0.5.9 — ``ForwardMode.is_draft_extend(include_v2=False)`` plus a
      separate ``is_draft_extend_v2()``.
    * sglang >= 0.5.16 — ``is_draft_extend`` was removed; only
      ``is_draft_extend_v2`` remains (v1 draft-extend is gone), and the v2 mode
      is also reachable through ``is_extend(include_draft_extend_v2=True)``.
    """
    legacy = getattr(forward_mode, "is_draft_extend", None)
    if legacy is not None:
        return bool(legacy())
    v2 = getattr(forward_mode, "is_draft_extend_v2", None)
    return bool(v2()) if v2 is not None else False


def dense_capture_cuda_graph(
    dense,
    *,
    bs,
    num_tokens,
    req_pool_indices,
    seq_lens,
    encoder_lens,
    forward_mode,
    spec_info,
    seq_lens_cpu=None,
) -> None:
    """Drive a *wrapped upstream* backend's cuda-graph CAPTURE metadata init.

    vortex's MLA backends wrap a stock sglang backend (``self._dense``, e.g.
    ``TritonAttnBackend``) for the dense layers and forward these calls to it.
    On sglang >= 0.5.16 that backend no longer has
    ``init_forward_metadata_capture_cuda_graph`` — it has
    ``init_forward_metadata_out_graph(fb, in_capture=True)`` instead — so pick
    whichever the wrapped object actually implements.
    """
    legacy = getattr(dense, "init_forward_metadata_capture_cuda_graph", None)
    if legacy is not None:
        legacy(
            bs, num_tokens, req_pool_indices, seq_lens, encoder_lens,
            forward_mode, spec_info,
        )
        return
    dense.init_forward_metadata_out_graph(
        _fb_view(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        ),
        in_capture=True,
    )


def dense_replay_cuda_graph(
    dense,
    *,
    bs,
    req_pool_indices,
    seq_lens,
    seq_lens_sum,
    encoder_lens,
    forward_mode,
    spec_info,
    seq_lens_cpu=None,
) -> None:
    """REPLAY-side twin of :func:`dense_capture_cuda_graph`."""
    legacy = getattr(dense, "init_forward_metadata_replay_cuda_graph", None)
    if legacy is not None:
        legacy(
            bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens,
            forward_mode, spec_info, seq_lens_cpu,
        )
        return
    dense.init_forward_metadata_out_graph(
        _fb_view(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_sum=seq_lens_sum,
        ),
    )


def _fb_view(
    *,
    bs,
    req_pool_indices,
    seq_lens,
    seq_lens_cpu,
    encoder_lens,
    forward_mode,
    spec_info,
    seq_lens_sum=None,
):
    """Minimal ForwardBatch-like view for a 0.5.16 ``_out_graph`` call.

    Mirrors the fields ``build_replay_fb_view`` supplies in
    ``model_executor/runner/decode_cuda_graph_runner.py``; vortex only drives
    single-token decode graphs, so ``num_tokens == bs`` and there is no padding.
    """
    from types import SimpleNamespace

    if seq_lens_sum is None and seq_lens is not None:
        seq_lens_sum = int(seq_lens.sum())
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=forward_mode,
        actual_forward_mode=forward_mode,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=seq_lens_sum,
        encoder_lens=encoder_lens,
        spec_info=spec_info,
        num_padding=0,
        out_cache_loc=None,
    )


def _graph_abi_is_split() -> bool:
    """True on the sglang releases whose ``AttentionBackend`` drives cuda-graph
    metadata through the 3-method contract
    (``init_forward_metadata`` / ``_out_graph`` / ``_in_graph``) instead of the
    legacy ``init_forward_metadata_{capture,replay}_cuda_graph`` pair.

    sglang 0.5.16 removed the legacy pair from the ABC *and* stopped calling it
    from the graph runners, so a backend that only implements the old methods
    silently never initializes its graph metadata.
    """
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    return hasattr(AttentionBackend, "init_forward_metadata_out_graph")


class LegacyCudaGraphABIMixin:
    """Adapts vortex's legacy cuda-graph metadata hooks onto sglang >= 0.5.16.

    Vortex's backends implement the pre-0.5.16 pair:

        init_forward_metadata_capture_cuda_graph(bs, num_tokens, req_pool_indices,
            seq_lens, encoder_lens, forward_mode, spec_info)
        init_forward_metadata_replay_cuda_graph(bs, req_pool_indices, seq_lens,
            seq_lens_sum, encoder_lens, forward_mode, spec_info, seq_lens_cpu)

    0.5.16 replaces both with a single call that the runner makes at capture
    (``in_capture=True``) and again before every replay (``in_capture=False``):

        init_forward_metadata_out_graph(forward_batch, in_capture=False)

    Both call sites hand over a ForwardBatch-like object carrying every field
    the legacy signatures need (see ``build_replay_fb_view`` in
    ``model_executor/runner/decode_cuda_graph_runner.py``), so this mixin just
    unpacks it and dispatches. ``init_forward_metadata_in_graph`` stays the
    base no-op: vortex records no graph-recordable metadata ops — its planning
    is host-side and runs in the out-graph phase.

    Mix in FIRST (before ``AttentionBackend``) so this override wins over the
    base implementation. On sglang <= 0.5.9 the mixin is inert: the runner calls
    the legacy methods directly and never calls ``_out_graph``.
    """

    def init_forward_metadata_out_graph(
        self,
        forward_batch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        encoder_lens = getattr(forward_batch, "encoder_lens", None)
        spec_info = getattr(forward_batch, "spec_info", None)

        if in_capture:
            # Capture: vortex's decode graphs are one token per sequence, which
            # is what the legacy assert `bs == num_tokens` encodes.
            self.init_forward_metadata_capture_cuda_graph(
                bs=bs,
                num_tokens=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                encoder_lens=encoder_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            return

        self.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=getattr(forward_batch, "seq_lens_sum", None),
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=getattr(forward_batch, "seq_lens_cpu", None),
        )


def attention_backend_base():
    """Base classes for a vortex attention backend.

    Returns a tuple to unpack into a class statement so the cuda-graph ABI
    adapter is only interposed on the sglang releases that need it::

        class VortexFooBackend(*attention_backend_base()):
            ...
    """
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    if _graph_abi_is_split():
        return (LegacyCudaGraphABIMixin, AttentionBackend)
    return (AttentionBackend,)
