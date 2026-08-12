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


def resolve_vortex_cache(pool):
    """Bind ``get_cache`` for ``pool`` once, returning ``f(layer_id) -> cache``.

    ``VortexCachePool.get_cache`` is vortex's own accessor, and upstream pools do
    not forward it. On a hybrid model the pool vortex is handed is upstream's
    ``HybridLinearKVPool`` wrapper, so the lookup has to reach the inner
    ``full_kv_pool`` and translate the global layer id to the dense
    full-attention index — the same thing the wrapper does for every other
    accessor.

    Which of those two shapes applies is fixed for the life of the backend, so it
    is decided **here, at init**, and the per-layer forward path calls the bound
    result directly. Probing with ``getattr`` on every call would put host work
    in ``forward_decode``, which runs once per layer per token.

    Vortex's own ``get_cache`` already indexes as ``layer_id - start_layer``, and
    the hybrid inner pool is built with ``start_layer=0`` precisely so the dense
    index the wrapper hands back is the right slot.
    """
    get_cache = getattr(pool, "get_cache", None)
    if get_cache is not None:
        return get_cache

    inner = getattr(pool, "full_kv_pool", None)
    translate = getattr(pool, "_transfer_full_attention_id", None)
    if inner is None or translate is None:
        raise AttributeError(
            f"{type(pool).__name__} exposes neither `get_cache` nor a "
            "`full_kv_pool` + `_transfer_full_attention_id` pair, so the vortex "
            "per-layer cache cannot be reached through it."
        )
    inner_get_cache = inner.get_cache

    def get_cache_via_wrapper(layer_id: int):
        return inner_get_cache(translate(layer_id))

    return get_cache_via_wrapper


def publish_pools(backend, model_runner) -> None:
    """Expose the pools on ``backend`` so upstream's accessors resolve, and bind
    the vortex per-layer cache lookup.

    Call from every vortex backend's ``__init__``. Publishing the pools is inert
    on sglang <= 0.5.9, where nothing reads these attributes off the backend.

    ``backend.vortex_cache(layer_id)`` is resolved here rather than per forward —
    see :func:`resolve_vortex_cache`. It is bound lazily on first use because the
    KV pool is not always attached to the runner yet when a backend is
    constructed; the resolution still happens once, not per layer.
    """
    pool = getattr(model_runner, "token_to_kv_pool", None)
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = getattr(model_runner, "req_to_token_pool", None)

    # The pool may not be attached to the runner yet when a backend is built, so
    # ``backend.token_to_kv_pool`` can be None here. ``vortex_pool()`` re-reads it
    # from the runner on demand, for the hooks that run *outside* a forward context
    # (metadata init / planning) and therefore cannot use
    # ``token_to_kv_pool(forward_batch)``.
    def vortex_pool():
        pool = backend.token_to_kv_pool
        if pool is None:
            pool = getattr(model_runner, "token_to_kv_pool", None)
            backend.token_to_kv_pool = pool
        return pool

    backend.vortex_pool = vortex_pool

    bound = {}

    def vortex_cache(layer_id: int):
        fn = bound.get("fn")
        if fn is None:
            fn = resolve_vortex_cache(backend.token_to_kv_pool)
            bound["fn"] = fn
        return fn(layer_id)

    backend.vortex_cache = vortex_cache
