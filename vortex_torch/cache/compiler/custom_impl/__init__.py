"""Schedule.S codegen for the cache compiler.

Mirror of :mod:`vortex_torch.indexer.compiler.custom_impl`. Schedule.S
cache ops (today: :class:`~vortex_torch.cache.LearnedDescriptor`) emit a
Python launcher body that reaches the op instance via
``ctx.op_list[<op_id>]`` and runs a ``torch`` computation over the
just-written latent — outside the fused per-block Triton kernel. The
emitted code is backend-agnostic.

Public surface:

  * :func:`get_impl_func(op)` — return the codegen function registered
    for ``op``'s class at ``Schedule.S``.
  * :func:`register_headers(ctx)` — append the module-level headers any
    Schedule.S codegen may need to ``ctx.compilation_header_lines``
    (idempotent — ``interface.generate_interface`` dedups via
    ``dict.fromkeys``).
"""
from .register import get_impl_func


def register_headers(ctx) -> None:
    """Append Schedule.S module-level headers to ``ctx`` (idempotent)."""
    ctx.compilation_header_lines.extend([
        "import torch",
        "import triton",
        "import triton.language as tl",
    ])


__all__ = ["get_impl_func", "register_headers"]
