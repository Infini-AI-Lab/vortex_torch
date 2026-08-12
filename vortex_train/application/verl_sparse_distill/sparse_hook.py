"""Install vortex sparse attention inside verl's training workers.

verl needs **no source changes** to train with sparse attention. Its config carries
``attn_implementation`` all the way to the model:

    verl/workers/config/model.py:185   AutoConfig.from_pretrained(attn_implementation=…)
    verl/workers/engine/fsdp/transformer_impl.py:257
                                       from_pretrained(config=hf_config)

and transformers dispatches per layer through ``ALL_ATTENTION_FUNCTIONS``. Verified
directly: a registered custom name is invoked once per layer (28/28 on Qwen3-0.6B), and
the only whitelist in ``engine.py`` is a rewrite map for veomni's FA names, which leaves
unknown names alone. So the whole integration is: register the function, then set
``attn_implementation=vortex_sparse`` in the YAML.

The one thing that does not come for free is **where** the registration happens. verl runs
training in Ray worker processes, so registering in the launcher is useless — the workers
import a fresh interpreter and would fall back to whatever the config names. Two ways in,
both provided:

``install_from_env()``
    Reads ``VORTEX_SPARSE_*`` from the environment. Ray propagates the launcher's env to
    workers by default, so exporting the geometry once before ``python -m
    verl.trainer.sft_trainer`` reaches every worker. This is the mechanism the runner
    script uses.
``sitecustomize`` / explicit import
    ``import application.verl_sparse_distill.sparse_hook`` at the top of any module the
    worker loads also works, because :func:`install_from_env` runs on import when
    ``VORTEX_SPARSE_ENABLE=1``.

Nothing here is vortex-specific plumbing that verl could have provided: the registry is
transformers' supported extension point, and this module is the adapter that fills it.
"""
from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def install_from_env() -> bool:
    """Register ``vortex_sparse`` from ``VORTEX_SPARSE_*`` env vars.

    Returns True if it installed, False if disabled. Raises if enabled but the geometry
    is unusable — a run that silently trained dense while claiming to be sparse is the
    failure mode worth being loud about, since it looks like a *result* rather than a
    misconfiguration.

    Recognised variables (defaults match the vortex_torch serving config this
    distillation targets):

    ==============================  =======  ====================================
    ``VORTEX_SPARSE_ENABLE``        0        set 1 to install
    ``VORTEX_SPARSE_ALGO``          block_topk   selection from vortex_train's REGISTRY
    ``VORTEX_SPARSE_TOPK``          16       *learned* top-k, vortex_torch convention
    ``VORTEX_SPARSE_BLOCK_Q``       1        query granularity (1 = per-token)
    ``VORTEX_SPARSE_BLOCK_KV``      64       KV block size
    ``VORTEX_SPARSE_RESERVE_BOS``   1        always-selected leading blocks
    ``VORTEX_SPARSE_RESERVE_LOCAL`` 1        always-selected local window
    ``VORTEX_SPARSE_RESERVE_EOS``   0        see the note in patch.install
    ==============================  =======  ====================================
    """
    if not _env_flag("VORTEX_SPARSE_ENABLE"):
        return False

    # Imported lazily so a dense run never pays the Triton import.
    from application.sparse_finetune_qwen3 import patch

    patch.install(
        algo=os.environ.get("VORTEX_SPARSE_ALGO", "block_topk"),
        topk=int(os.environ.get("VORTEX_SPARSE_TOPK", "16")),
        block_q=int(os.environ.get("VORTEX_SPARSE_BLOCK_Q", "1")),
        block_kv=int(os.environ.get("VORTEX_SPARSE_BLOCK_KV", "64")),
        reserve_bos=int(os.environ.get("VORTEX_SPARSE_RESERVE_BOS", "1")),
        reserve_local=int(os.environ.get("VORTEX_SPARSE_RESERVE_LOCAL", "1")),
        reserve_eos=int(os.environ.get("VORTEX_SPARSE_RESERVE_EOS", "0")),
    )
    print(f"[vortex] sparse attention installed in pid {os.getpid()}: "
          f"{patch.describe()}", flush=True)
    return True


def sparse_stats() -> dict:
    """How often the sparse path actually ran, per process.

    Worth logging at the end of a run: ``vortex_train`` falls back to dense for short or
    padded sequences, so a run can be configured sparse and still execute mostly dense.
    ``dense_reason`` says which guard fired, which distinguishes "the data is short" from
    "batching produced a padded mask".
    """
    from application.sparse_finetune_qwen3 import patch
    return dict(patch.STATS)


# Import-time install, so ``import …sparse_hook`` inside a worker is sufficient.
_INSTALLED = install_from_env()
