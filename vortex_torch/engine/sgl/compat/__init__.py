"""Compatibility layer between vortex and the vendored sglang release.

vortex vendors sglang (``third_party/sglang/v*``) and hooks a handful of points
in it. Upstream periodically moves a symbol, reshapes an interface vortex
implements, or changes *when* it calls vortex. Everything needed to absorb that
lives here, so the vendored tree keeps only the genuine ``[VORTEX HOOK]`` call
sites and the rest of vortex reads as if the API never moved.

Each submodule owns one kind of change:

* :mod:`symbols`     — functions/predicates that moved or were renamed.
* :mod:`pools`       — how the KV / req-to-token pools are reached.
* :mod:`cuda_graph`  — the two cuda-graph metadata ABIs, in both directions.
* :mod:`runner_view` — a read-only ``ModelRunner`` overlay for pool construction.

The vendored tree is v0.5.16. Every shim still resolves against the older 0.5.9
layout as well — that costs nothing and is what makes each one self-documenting
about *what* moved — but only 0.5.16 is tested. Keep this package import-light:
it is imported at attention-backend module scope, inside the spawned scheduler
worker.
"""
from vortex_torch.engine.sgl.compat.cuda_graph import (
    GraphMetadataArgs,
    LegacyCudaGraphABIMixin,
    attention_backend_base,
    capture_dense,
    replay_dense,
)
from vortex_torch.engine.sgl.compat.pools import publish_pools, token_to_kv_pool
from vortex_torch.engine.sgl.compat.runner_view import runner_view
from vortex_torch.engine.sgl.compat.symbols import (
    get_attention_tp_size,
    is_draft_extend,
)

__all__ = [
    "GraphMetadataArgs",
    "LegacyCudaGraphABIMixin",
    "attention_backend_base",
    "capture_dense",
    "get_attention_tp_size",
    "is_draft_extend",
    "publish_pools",
    "replay_dense",
    "runner_view",
    "token_to_kv_pool",
]
