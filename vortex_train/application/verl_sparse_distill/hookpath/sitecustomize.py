"""Auto-imported by every interpreter that has this directory on PYTHONPATH.

This is how the sparse attention implementation reaches verl's Ray/torchrun **workers**
without editing verl: python imports ``sitecustomize`` at startup, before any user code,
so the registration is in place by the time verl builds the model.

Ordering is the whole point. transformers validates ``attn_implementation`` against
``ALL_ATTENTION_FUNCTIONS`` inside ``PreTrainedModel.__init__``
(``_check_and_adjust_attn_implementation``), and raises

    ValueError: Specified `attn_implementation="vortex_sparse"` is not supported

if the name is not registered *at that moment*. Registering after the model is built is
too late, and registering only in the launcher never reaches the workers at all.

A stable directory in the repo rather than a mktemp one: a temporary path makes the
mechanism invisible to anyone reading the run script, and it silently no-ops if the
directory is cleaned up between launcher and worker start.
"""
import os
import sys

if os.environ.get("VORTEX_SPARSE_ENABLE") == "1":
    try:
        import application.verl_sparse_distill.sparse_hook  # noqa: F401
    except Exception as exc:
        # Loud, not silent: a run that trains dense while reporting itself sparse looks
        # like a *result*, which is far worse than a crash at startup.
        print(f"[vortex] FAILED to install sparse attention: {exc!r}", file=sys.stderr)
        raise
