"""Reaching the KV / req-to-token pools across sglang releases.

sglang 0.5.16 stopped carrying the pools on ``ForwardBatch`` and started
reading them off the *attention backend* instead
(``forward_context.get_token_to_kv_pool`` → ``get_attn_backend().token_to_kv_pool``).
That cuts both ways for vortex, so there are two halves here:

* :func:`token_to_kv_pool` — how vortex's own code finds the pool.
* :func:`publish_pools` — how vortex lets *upstream* code find it, by exposing
  the attributes the new accessor expects on each backend.

The second half is load-bearing beyond vortex's own reads:
``models/utils.py::enable_fused_set_kv_buffer`` resolves the pool this way to
decide whether the fused RoPE KV-store kernel may run, and that kernel must be
declined for vortex's block-interleaved layout.
"""
from __future__ import annotations


def token_to_kv_pool(forward_batch):
    """The KV pool backing this forward.

    * sglang <= 0.5.9 — carried on the batch as ``forward_batch.token_to_kv_pool``.
    * sglang >= 0.5.16 — dropped from ``ForwardBatch``; reached through
      ``forward_context.get_token_to_kv_pool()``.
    """
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is not None:
        return pool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    return get_token_to_kv_pool()


def vortex_cache(forward_batch, layer_id: int):
    """The vortex per-layer cache dict for ``layer_id`` (a **global** layer id).

    ``VortexCachePool.get_cache`` is vortex's own accessor and upstream pools do
    not forward it. On a hybrid model the pool vortex gets handed is upstream's
    ``HybridLinearKVPool`` wrapper, so reach through to the inner
    ``full_kv_pool`` — translating the global layer id to the dense
    full-attention index the way the wrapper does for every other accessor.
    """
    pool = token_to_kv_pool(forward_batch)
    get_cache = getattr(pool, "get_cache", None)
    if get_cache is not None:
        return get_cache(layer_id)

    inner = getattr(pool, "full_kv_pool", None)
    translate = getattr(pool, "_transfer_full_attention_id", None)
    if inner is None or translate is None:
        raise AttributeError(
            f"{type(pool).__name__} exposes neither `get_cache` nor a "
            "`full_kv_pool` + `_transfer_full_attention_id` pair, so the vortex "
            "per-layer cache cannot be reached through it."
        )
    return inner.get_cache(translate(layer_id))


def publish_pools(backend, model_runner) -> None:
    """Expose the pools on ``backend`` so upstream's accessors resolve.

    Call from every vortex backend's ``__init__``. Inert on sglang <= 0.5.9,
    where nothing reads these attributes off the backend.
    """
    backend.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    backend.req_to_token_pool = getattr(model_runner, "req_to_token_pool", None)
