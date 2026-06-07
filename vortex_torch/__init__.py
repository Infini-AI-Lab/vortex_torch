
# Pin the CUDA arch for ALL of vortex's JIT (planner/prefill/custom-ops/top-k)
# the moment vortex is imported, BEFORE any load_inline runs — single SASS
# gencode instead of SASS+PTX, which roughly halves every cold compile. See
# vortex_torch/_jit_setup.py. Best-effort; respects an explicit user setting.
from ._jit_setup import configure_jit_env as _vx_configure_jit_env
from ._jit_setup import warmup_jit as warmup_jit
from ._jit_setup import clear_stale_jit_locks as clear_stale_jit_locks
_vx_configure_jit_env()

from . import indexer as indexer
from . import cache as cache
from . import flow as flow
from .abs import Parameter as Parameter
from . import abs as abs
from .utils import is_hopper_or_newer, is_hopper
from .version import __version__

__all__ = [
        "indexer",
        "cache",
        "flow",
        "abs",
        "Parameter",
        "is_hopper_or_newer",
        "is_hopper",
        "integration",
        "warmup_jit",
        "clear_stale_jit_locks",
        "__version__",
]


# Eagerly install the ServerArgs flat-kwargs adapter so sgl.Engine(vortex_*=...)
# keeps working (it must be active in the *parent* before ServerArgs is built).
# Light: imports only sglang.srt.server_args, not the engine. Best-effort.
try:
    from .engine.sgl.config import install_serverargs_adapter as _vx_install_adapter
    _vx_install_adapter()
except Exception:
    pass


def __getattr__(name):
    # Lazily expose ``vortex_torch.integration`` (the single sglang-integration
    # module) without making a plain ``import vortex_torch`` pull in sglang/the
    # engine. The sglang hooks call ``vortex_torch.integration.<fn>(...)``; the
    # first such access here imports it (and ``build_sparse_flow`` then registers
    # the vortex attention backends in the spawned worker). See
    # vortex_torch/engine/sgl/integration.py.
    if name == "integration":
        from .engine.sgl import integration
        return integration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


