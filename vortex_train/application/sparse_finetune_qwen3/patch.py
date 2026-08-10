"""Swap Qwen3's attention for vortex_train's sparse path.

Registers a ``"vortex_sparse"`` entry in transformers' attention registry, so the model
picks it up via ``attn_implementation="vortex_sparse"`` and nothing in the modeling code
is monkey-patched. That matters for maintainability: the registry is the supported
extension point, and `Qwen3Attention.forward` already routes through it
(``ALL_ATTENTION_FUNCTIONS.get_interface``), passing q/k/v post-RoPE and post-q_norm in
``[B, H, T, D]`` — exactly the layout the kernels want.

Two behaviours worth stating because they are decisions, not defaults:

* **One selection per forward, shared by every layer that asks for the same geometry.**
  A ``SparseAttention`` module is built per (layer, shape) and cached, so the Triton
  compile is paid once rather than 36 times. The *pattern* is still rebuilt per layer —
  it depends on that layer's q/k, so sharing it across layers would be wrong.
* **Padded and short sequences fall back to dense.** Below ``topk * block_kv`` tokens
  every block is selected anyway, so the sparse path would do the same work plus
  scoring; and a batch with padding needs the mask honoured. Both are handled by
  delegating to SDPA, which keeps the comparison honest — a "sparse" run that silently
  fell back would otherwise look fast for the wrong reason.
"""
from __future__ import annotations

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vortex_train.flow.spec import REGISTRY, Budget
from vortex_train.nn import SparseAttention

_MODULES: dict[tuple, SparseAttention] = {}

#: set by :func:`install`; read by the attention function
_CONFIG: dict = {}

#: counters so a run can prove how often the sparse path actually ran
STATS = {"sparse_calls": 0, "dense_calls": 0, "dense_reason": {}}


def _policy(algo: str, topk: int, block_q: int, block_kv: int,
            reserve_bos: int, reserve_local: int, reserve_eos: int):
    """A Selection subclass with this run's geometry, built once and cached by key.

    ``topk`` here is the **total** block budget, reservations included — see
    :func:`install` for the conversion from vortex_torch's additive convention.
    """
    base = REGISTRY[algo]
    return type(
        f"{base.__name__}_k{topk}_bq{block_q}",
        (base,),
        {
            "block_q": block_q,
            "block_kv": block_kv,
            "budget": Budget(
                topk=topk,
                reserve_bos=reserve_bos,
                reserve_local=reserve_local,
                reserve_eos=reserve_eos,
            ),
        },
    )


def _get_module(num_kv_heads: int, scaling: float) -> SparseAttention:
    key = (_CONFIG["algo"], _CONFIG["total_blocks"], _CONFIG["block_q"],
           _CONFIG["block_kv"], _CONFIG["reserve_bos"], _CONFIG["reserve_local"],
           _CONFIG["reserve_eos"], num_kv_heads, round(scaling, 8))
    mod = _MODULES.get(key)
    if mod is None:
        mod = SparseAttention(
            _policy(_CONFIG["algo"], _CONFIG["total_blocks"],
                    _CONFIG["block_q"], _CONFIG["block_kv"],
                    _CONFIG["reserve_bos"], _CONFIG["reserve_local"],
                    _CONFIG["reserve_eos"]),
            num_kv_heads=num_kv_heads,
            softmax_scale=scaling,
        )
        _MODULES[key] = mod
    return mod


def _note_dense(reason: str) -> None:
    STATS["dense_calls"] += 1
    STATS["dense_reason"][reason] = STATS["dense_reason"].get(reason, 0) + 1


def vortex_sparse_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,            # [B, Hq, T, D]
    key: torch.Tensor,              # [B, Hkv, T, D]
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    b, hq, tq, d = query.shape
    hkv, tkv = key.shape[1], key.shape[2]
    dense_below = _CONFIG["total_blocks"] * _CONFIG["block_kv"]

    reason = None
    if tq != tkv:
        # decode / cross attention: this application only trains full sequences
        reason = "tq != tkv"
    elif tq < dense_below:
        # every block would be selected anyway -- sparse would cost strictly more
        reason = f"seqlen < {dense_below}"
    elif attention_mask is not None:
        # A padded batch needs the mask honoured; the sparse kernels take a pattern,
        # not a mask. Handled by using batch size 1 upstream (see train.py), so this
        # is a correctness backstop rather than the common path.
        reason = "attention_mask present"
    elif dropout:
        reason = "attention dropout"

    if reason is not None:
        _note_dense(reason)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    STATS["sparse_calls"] += 1
    attn = _get_module(hkv, scaling if scaling is not None else d ** -0.5)
    out = attn(query, key, value)          # [B, Hq, T, D]
    # transformers expects [B, T, H, D] back from an attention function.
    return out.transpose(1, 2).contiguous(), None


def install(*, algo: str = "block_topk", topk: int = 16,
            block_q: int = 1, block_kv: int = 64,
            reserve_bos: int = 1, reserve_local: int = 1,
            reserve_eos: int = 0) -> None:
    """Register the sparse attention implementation and record this run's geometry.

    ``topk`` is given in **vortex_torch's convention**: the *learned* top-k, with the
    reservations layered on top. vortex_torch computes

        selected = topk_val + reserved_bos + reserved_eos

    whereas ``vortex_train``'s ``Budget.topk`` is the *total* including reservations.
    Passing ``topk`` straight through would therefore make training attend
    ``bos + eos`` fewer blocks than serving — a silent mismatch of exactly the kind
    that invalidates a train/serve comparison. So the total is computed here:

        total_blocks = topk + reserve_bos + reserve_local + reserve_eos

    On the mapping of ``eos``: vortex_torch's ``reserved_eos`` is the *last N blocks of
    the sequence*, which during decode is the recent/local window. In training every
    query position has its own "most recent" block, so the faithful analogue is
    ``reserve_local``, not ``reserve_eos``. ``reserve_eos`` is still exposed for
    completeness but defaults to 0.
    """
    if algo not in REGISTRY:
        raise KeyError(f"unknown selection {algo!r}; have {sorted(REGISTRY)}")
    total = topk + reserve_bos + reserve_local + reserve_eos
    _CONFIG.update(algo=algo, topk=topk, total_blocks=total,
                   block_q=block_q, block_kv=block_kv,
                   reserve_bos=reserve_bos, reserve_local=reserve_local,
                   reserve_eos=reserve_eos)
    ALL_ATTENTION_FUNCTIONS["vortex_sparse"] = vortex_sparse_attention_forward
    reset_stats()


def describe() -> str:
    """One line describing the effective budget, for the run log."""
    c = _CONFIG
    return (f"{c['algo']}: topk_val={c['topk']} + bos={c['reserve_bos']} + "
            f"local={c['reserve_local']} + eos={c['reserve_eos']} = "
            f"{c['total_blocks']} blocks x block_kv={c['block_kv']} = "
            f"{c['total_blocks'] * c['block_kv']} KV tokens, block_q={c['block_q']}")


def reset_stats() -> None:
    STATS["sparse_calls"] = 0
    STATS["dense_calls"] = 0
    STATS["dense_reason"] = {}


def sparse_fraction() -> float:
    tot = STATS["sparse_calls"] + STATS["dense_calls"]
    return STATS["sparse_calls"] / tot if tot else 0.0
