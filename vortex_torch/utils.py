from enum import Enum
import torch

class _UNSET_T:
    def __repr__(self) -> str: return "UNSET"
UNSET = _UNSET_T()


class Mode(Enum):
    profile = 0
    execute = 1

class Schedule(Enum):
    W = 0
    S = 1
    
class ReduceType(Enum):
    Mean = 0
    Max = 1
    Min = 2
    L2Norm = 3
    Sum = 4
    
class ElementwiseBinaryOpType(Enum):
    Maximum = 0
    Minimum = 1
    Add = 2
    Mul = 3
    WhereEqual = 4
    WhereNotEqual = 5
    WhereGreater = 6
    WhereGreaterEqual = 7
    WhereLess = 8
    WhereLessEqual = 9
    

class ElementwiseOpType(Enum):
    Relu = 0
    Sigmoid = 1
    Silu = 2
    Abs = 3
    Add_Mul = 4
    Log = 5
    Exp = 6


class QuantizationType(Enum):
    BF16 = 0
    FP8_E5M2 = 1
    FP8_E4M3 = 2

INDENT = "    "
def indent_block(text: str, level: int = 1) -> str:
    """
    Indent a multi-line text block by the given indentation level.
    """
    prefix = INDENT * level
    lines = text.splitlines()
    return "\n".join(prefix + line if line.strip() else line for line in lines)


# ---------------------------------------------------------------------------
# Dtype string <-> torch.dtype
#
# Canonical string names used in server args, the verify CLI, and anywhere
# else a dtype is configured by name. FP8 uses the short ``fp8_`` prefix;
# other dtypes use their torch attribute names. Short aliases (``bf16``,
# ``fp16``, ``fp32``, ``fp8_e4m3``) map to the same dtype.
# ---------------------------------------------------------------------------

#: Canonical string -> ``torch.dtype``. Keep this one-to-one; aliases go in
#: :data:`_DTYPE_ALIASES` so tooling that prints the valid set (``choices=``,
#: error messages) can show canonical names without alias noise.
DTYPE_STR_TO_TORCH = {
    "bfloat16":    torch.bfloat16,
    "float16":     torch.float16,
    "float32":     torch.float32,
    "fp8_e5m2":    torch.float8_e5m2,
    "fp8_e4m3fn":  torch.float8_e4m3fn,
}

#: Accepted short-forms; every value must be a key of :data:`DTYPE_STR_TO_TORCH`.
_DTYPE_ALIASES = {
    "bf16":      "bfloat16",
    "fp16":      "float16",
    "fp32":      "float32",
    "fp8_e4m3":  "fp8_e4m3fn",   # torch only ships the ``fn`` variant
}


def resolve_dtype(value, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Resolve a dtype given either a ``torch.dtype`` or a canonical string.

    Accepts ``None`` (returns ``default``), a ``torch.dtype`` (returned as-is),
    a canonical string in :data:`DTYPE_STR_TO_TORCH`, or a short alias in
    :data:`_DTYPE_ALIASES`.
    """
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = _DTYPE_ALIASES.get(value, value)
        if name not in DTYPE_STR_TO_TORCH:
            raise ValueError(
                f"unknown dtype string {value!r}; pick one of "
                f"{sorted(DTYPE_STR_TO_TORCH)} "
                f"(aliases: {sorted(_DTYPE_ALIASES)})"
            )
        return DTYPE_STR_TO_TORCH[name]
    raise TypeError(f"cannot resolve dtype from {type(value).__name__}")


def is_int4_kv(cache_meta_info) -> bool:
    """Is this cache's K/V stored as packed INT4?

    Tested on the DECLARED META, not on a config flag, because the meta is what every consumer
    actually addresses: a backend that reads the flag while the pool was built from a different meta
    (or vice versa) mis-strides by 2x, which reads as plausible-looking wrong data rather than as an
    error. ``uint8`` alone is not sufficient -- fp8 caches are also stored as uint8 -- so the test is
    uint8 K/V whose channel count is HALVED relative to a full-width field, which is exactly the
    packing signature.
    """
    kv = cache_meta_info.get("k")
    if kv is None:
        return False
    (r, c), dtype = kv
    if dtype != torch.uint8:
        return False
    # The scale fields are the unambiguous marker; they exist only for INT4.
    return "k_int4_scale" in cache_meta_info


def is_hopper_or_newer():
    """
    Check if the current CUDA device is Hopper architecture or newer.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # Hopper architecture has compute capability 9.0 and above
    return (major > 9) or (major == 9 and minor >= 0)

def is_hopper():
    """
    Check if the current CUDA device is exactly Hopper architecture.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # Hopper architecture has compute capability 9.0
    return (major == 9 and minor == 0)
