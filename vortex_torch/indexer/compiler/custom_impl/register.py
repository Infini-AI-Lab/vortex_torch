"""Registry of ``op_class -> codegen-function`` for Schedule.S ops.

Schedule.S codegens emit a Python wrapper body that resolves an underlying
custom kernel via :func:`vortex_torch.custom_ops.find` at module-import
time and calls it on the hot path. They are **backend-agnostic** — the
emitted body has no coupling to the fused Schedule.W kernel, so the same
generator is reused regardless of whether the Schedule.W code is being
emitted by ``triton_impl`` or ``cuda_impl``.

Only ``Schedule.S`` ops live here. Each generator is responsible for its
own backend dispatch (via :func:`indexer.compiler.backend.get_backend`)
and for appending whatever module-level headers / kernel-cache trampolines
it needs to ``ctx.compilation_header_lines``.
"""
from ....utils import Schedule

from ...output_func import topK, approxTopK, Union
from ...select import TopK
from ...scan import Softmax, Normalize, Conv1d
from ...reduce import Reduce
from ...matmul import GeMM
from ...reshape import Reshape
from ...factor_score import FactorScore

from .topk import (
    generate_topk_impl,
    generate_approx_topk_impl,
    generate_block_table_topk_impl,
    generate_union_impl,
)
from .softmax import generate_softmax_impl
from .normalize import generate_normalize_impl
from .conv1d import generate_conv1d_impl
from .reduce_dim0 import generate_reduce_dim0_impl
from .gemm_param import generate_gemm_param_impl
from .reshape_s import generate_reshape_s_impl
from .factor_score import generate_factor_score_impl


IMPL_REGISTRY = {
    (topK,       Schedule.S): generate_topk_impl,
    (approxTopK, Schedule.S): generate_approx_topk_impl,
    # TopK (capital T) — trtllm-only, writes block_table + seqlens.
    (TopK,       Schedule.S): generate_block_table_topk_impl,
    # Union — trtllm-only output op: merges two (bt, sl) pairs into the
    # final sparse_block_tables + sparse_seqlens.
    (Union,      Schedule.S): generate_union_impl,
    # The kernels themselves live under
    # ``vortex_torch/custom_ops/<op>/<backend>/default/kernel.py``; these
    # generate_*_impl functions are launcher-emitters that resolve the
    # kernel via ``custom_ops.find`` at runtime.
    (Softmax,    Schedule.S): generate_softmax_impl,
    (Normalize,  Schedule.S): generate_normalize_impl,
    (Conv1d,     Schedule.S): generate_conv1d_impl,
    # Reduce.dim==0 is the cross-row, RAGGED → BATCHED form. The fused
    # ``dim in {1, 2}`` form is Schedule.W and lives in the
    # corresponding backend's ``triton_impl`` / ``cuda_impl`` registry.
    (Reduce,     Schedule.S): generate_reduce_dim0_impl,
    # GeMM with a batch-shared Vortex.Parameter operand → torch.matmul launcher.
    (GeMM,         Schedule.S): generate_gemm_param_impl,
    # Reshape under padded (non-pow2) inner dims → standalone torch reshape.
    (Reshape,      Schedule.S): generate_reshape_s_impl,
    # FactorScore — learned factorized block scorer (torch launcher).
    (FactorScore,  Schedule.S): generate_factor_score_impl,
}


def get_impl_func(op):
    """Resolve a Schedule.S op instance to its codegen function.

    Exact class match wins (so a subclass like ``approxTopK`` gets its
    own generator instead of inheriting ``topK``'s). Falls back to an
    MRO walk: the closest registered ancestor at the same schedule
    provides the generator.
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
        f"No Schedule.S indexer codegen for op {cls.__name__} "
        f"with schedule {schedule}"
    )
