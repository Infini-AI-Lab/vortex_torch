"""Build per-KV-block state: one fused reduction over the block axis.

All of a policy's ``Field``s are produced by **one launch**, not one per field. The
reduction reads each KV block's tokens once and computes every requested statistic
from that single pass, because the loads dominate: at ``block_kv=64, D=128`` a block
is 16 KB and the arithmetic is trivial, so a second launch over ``k`` would roughly
double the cost of state building for no reason.

**Sub-blocks.** A field may keep one summary per run of ``sub_block`` tokens rather
than one per block (the LServe refinement — see
:class:`~vortex_train.flow.spec.Field`). Fields with different ``sub_block`` values
coexist in one launch: the output is padded to the widest field's sub-count, and each
field writes only the slots it owns.

Like the select kernel, this is **generated per field-set** rather than written once
with the field list passed as data. The reason is concrete rather than stylistic:
each field's sub-count is both a loop trip count and a tile width, and Triton needs
both as compile-time constants. A ``tl.static_range`` bounded by a tuple element
indexed inside another ``static_range`` is not recognised as constexpr, and binding it
to a named constexpr fails too because Triton forbids reassigning one across loop
iterations — which is what a second field does. Generating the per-field bodies
sidesteps both and emits only the reductions the policy actually asked for.

Layout: ``[B, Hkv, Nkv, F, NSUB, D]`` fp32, where slot ``f`` is ``fields[f]`` and
``NSUB`` is the max sub-count. fp32 because these are reduction *outputs* feeding
another reduction (the score contraction), and bf16 here would compound two
roundings before anything is compared.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import tempfile

import torch

_PRELUDE = """import triton
import triton.language as tl
"""

_TEMPLATE = '''
@triton.jit
def _state_kernel(
    K, V, STATE,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_sb, stride_sh, stride_sn, stride_sf, stride_su, stride_sd,
    seqlen_kv,
    BLOCK_KV: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """One program per (KV block, batch, kv-head); every field in that one pass."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_d = tl.arange(0, HEAD_DIM)
    base_n = pid_n * BLOCK_KV
    kbase = pid_b * stride_kb + pid_h * stride_kh
    vbase = pid_b * stride_vb + pid_h * stride_vh
    sbase = pid_b * stride_sb + pid_h * stride_sh + pid_n * stride_sn

{BODY}
'''

_IND = " " * 4


def _emit_field(f: int, reduce: str, src: str, nsub: int, n_sub_max: int,
                block_kv: int) -> list[str]:
    """Emit one field's reduction, unrolled over its sub-blocks."""
    sub_len = block_kv // nsub
    ptr, base, s_n, s_d = (("K", "kbase", "stride_kn", "stride_kd") if src == "k"
                           else ("V", "vbase", "stride_vn", "stride_vd"))
    out = [f"{_IND}# field {f}: {reduce} over {src}, {nsub} sub-block(s) of {sub_len}"]
    for u in range(nsub):
        t = f"{f}_{u}"
        out.append(f"{_IND}o{t} = base_n + {u * sub_len} + tl.arange(0, {sub_len})")
        out.append(f"{_IND}m{t} = o{t} < seqlen_kv")
        out.append(
            f"{_IND}x{t} = tl.load({ptr} + {base} + o{t}[:, None] * {s_n}"
            f" + offs_d[None, :] * {s_d}, mask=m{t}[:, None], other=0.0).to(tl.float32)"
        )
        # Tokens past the end of the sequence must not participate. `other=0.0`
        # covers mean/sum; for max/min a zero would beat genuinely negative (resp.
        # positive) data, so those need an explicit sentinel.
        if reduce == "mean":
            out.append(
                f"{_IND}r{t} = tl.sum(x{t}, axis=0) / "
                f"tl.maximum(tl.sum(m{t}.to(tl.float32), axis=0), 1.0)"
            )
        elif reduce == "max":
            out.append(
                f'{_IND}r{t} = tl.max(tl.where(m{t}[:, None], x{t}, float("-inf")), axis=0)'
            )
        elif reduce == "min":
            out.append(
                f'{_IND}r{t} = tl.min(tl.where(m{t}[:, None], x{t}, float("inf")), axis=0)'
            )
        else:
            out.append(f"{_IND}r{t} = tl.sum(x{t}, axis=0)")
        out.append(
            f"{_IND}tl.store(STATE + sbase + {f} * stride_sf + {u} * stride_su"
            f" + offs_d * stride_sd, r{t})"
        )
    # Zero the slots this field does not own. Nothing reads them — each scorer call
    # site is specialised to its own field's sub-count — but they are written rather
    # than left uninitialised so a future misread surfaces as an obvious zero instead
    # of reused garbage. Deliberately NOT -inf: an inf here would turn a misread into
    # a NaN that propagates silently through the score sum.
    for u in range(nsub, n_sub_max):
        out.append(
            f"{_IND}tl.store(STATE + sbase + {f} * stride_sf + {u} * stride_su"
            f" + offs_d * stride_sd, tl.zeros((HEAD_DIM,), dtype=tl.float32))"
        )
    return out


def _emit_body(fields, block_kv: int) -> str:
    n_sub_max = max(ns for _, _, ns in fields)
    parts = [
        "\n".join(_emit_field(f, red, src, ns, n_sub_max, block_kv))
        for f, (red, src, ns) in enumerate(fields)
    ]
    return "\n\n".join(parts)


def generated_source(fields, block_kv: int) -> str:
    """The emitted kernel source. For debugging a policy, and for tests."""
    return _TEMPLATE.replace("{BODY}", _emit_body(fields, block_kv))


_KERNEL_CACHE: dict[tuple, object] = {}


def build_state_kernel(fields, block_kv: int):
    """Generate (or fetch) the state kernel specialised to this field-set."""
    key = (tuple(fields), block_kv)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    src = _PRELUDE + generated_source(fields, block_kv)
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
    cache_dir = pathlib.Path(tempfile.gettempdir()) / "vortex_train_kernels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"state_{digest}.py"
    if not path.exists():
        # Write once, atomically: two processes compiling the same policy must not
        # observe a half-written file.
        tmp = path.with_suffix(f".{id(fields):x}.tmp")
        tmp.write_text(src)
        tmp.replace(path)

    spec = importlib.util.spec_from_file_location(f"vortex_train_state_{digest}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = mod._state_kernel
    _KERNEL_CACHE[key] = kernel
    return kernel


def build_state(
    k: torch.Tensor,
    v: torch.Tensor,
    fields: tuple[tuple[str, str, int], ...],
    *,
    block_kv: int,
) -> torch.Tensor:
    """Return packed state ``[B, Hkv, Nkv, F, NSUB, D]`` in fp32.

    ``fields`` is an ordered tuple of ``(reduce, src, num_sub)``; slot ``f`` in the
    output corresponds to ``fields[f]``, and only its first ``num_sub`` sub-slots
    hold data.
    """
    b, hkv, skv, d = k.shape
    n_kv = (skv + block_kv - 1) // block_kv
    if not fields:
        # A stateless policy (pure position-based scoring) is legitimate; hand the
        # scorer a well-formed empty tensor rather than a None to branch on.
        return torch.zeros((b, hkv, n_kv, 0, 1, d), dtype=torch.float32, device=k.device)

    n_sub = max(f[2] for f in fields)
    state = torch.empty((b, hkv, n_kv, len(fields), n_sub, d),
                        dtype=torch.float32, device=k.device)

    build_state_kernel(fields, block_kv)[(n_kv, hkv, b)](
        k, v, state,
        *k.stride(), *v.stride(), *state.stride(),
        skv,
        BLOCK_KV=block_kv, HEAD_DIM=d,
        num_warps=4,
    )
    return state
