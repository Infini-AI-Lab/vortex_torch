"""Registry of ``op_class -> codegen-function`` for Schedule.S cache ops.

Mirror of :mod:`vortex_torch.indexer.compiler.custom_impl.register`.
Schedule.S cache codegens emit a Python launcher body that reaches the
op instance via ``ctx.op_list[<op_id>]`` and runs a ``torch`` op over the
freshly written latent. They are backend-agnostic — the emitted body has
no coupling to the fused Schedule.W cache kernel.
"""
from ....utils import Schedule

from ...learned_descriptor import LearnedDescriptor

from .learned_descriptor import generate_learned_descriptor_impl


IMPL_REGISTRY = {
    (LearnedDescriptor, Schedule.S): generate_learned_descriptor_impl,
}


def get_impl_func(op):
    """Resolve a Schedule.S cache op instance to its codegen function.

    Exact class match wins; falls back to an MRO walk so a subclass picks
    up the closest registered ancestor at the same schedule.
    """
    schedule = op.schedule
    cls = op.__class__

    exact = IMPL_REGISTRY.get((cls, schedule))
    if exact is not None:
        return exact

    for parent in cls.__mro__[1:]:
        impl_func = IMPL_REGISTRY.get((parent, schedule))
        if impl_func is not None:
            return impl_func

    raise NotImplementedError(
        f"No Schedule.S cache codegen for op {cls.__name__} "
        f"with schedule {schedule}"
    )
