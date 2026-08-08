"""A read-only view of ``ModelRunner`` for pool construction.

vortex builds its KV pool from a ``ModelRunner``, reading ~10 attributes off it
(and handing it to ``Context.create``, which reads two more). sglang 0.5.16 moved
*when* that construction happens: the pool is now built inside
``KVCacheConfigurator.configure()``, which runs before the runner is given

* ``max_total_num_tokens``  — the configurator computes and returns it, and
* ``req_to_token_pool``     — built as a local, assigned to the runner after,

and which keeps the layer span on a ``ModelLayerInfo`` struct rather than on the
runner (``num_effective_layers`` / ``start_layer`` / ``end_layer`` are gone from
``ModelRunner`` entirely).

So at the moment vortex needs it, the runner is genuinely incomplete. Rather
than back-fill the missing fields onto the caller's runner — a side effect on an
object we don't own, whose ordering then matters — :func:`runner_view` wraps it
in an overlay that answers the missing attributes and forwards everything else.
The runner is only ever *read* during pool construction (nothing stores it), so
a view is sufficient and keeps the data flow one-directional.
"""
from __future__ import annotations

from typing import Any, Optional


class _RunnerView:
    """Reads ``overrides`` first, then falls through to the wrapped runner."""

    __slots__ = ("_runner", "_overrides")

    def __init__(self, runner: Any, overrides: dict):
        # Bypass our own __setattr__ (writes are rejected below).
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_runner"), name)

    def __setattr__(self, name: str, value):
        raise AttributeError(
            f"{type(self).__name__} is read-only; cannot set {name!r}. "
            "Pool construction must not mutate the ModelRunner."
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        runner = object.__getattribute__(self, "_runner")
        keys = sorted(object.__getattribute__(self, "_overrides"))
        return f"<_RunnerView of {type(runner).__name__} overriding {keys}>"


def runner_view(
    runner: Any,
    *,
    max_total_num_tokens: Optional[int] = None,
    layer_info: Optional[Any] = None,
    req_to_token_pool: Optional[Any] = None,
) -> Any:
    """Return ``runner``, or a read-only overlay when anything is supplied.

    Each keyword fills in one thing the runner cannot answer yet; omit them all
    (the sglang <= 0.5.9 call site) and the runner is handed back untouched, so
    there is no wrapper on the path that doesn't need one.
    """
    overrides: dict = {}
    if max_total_num_tokens is not None:
        overrides["max_total_num_tokens"] = max_total_num_tokens
    if layer_info is not None:
        overrides["num_effective_layers"] = layer_info.num_effective_layers
        overrides["start_layer"] = layer_info.start_layer
        overrides["end_layer"] = layer_info.end_layer
    if req_to_token_pool is not None:
        overrides["req_to_token_pool"] = req_to_token_pool
    if not overrides:
        return runner
    return _RunnerView(runner, overrides)
