"""Fused score + reserve + top-k: one program per query block, one launch.

This is the kernel the compiler emits, and the reason the compiler exists. The
O(T²) block-score matrix never reaches HBM — 4 GB per layer at 1M tokens if it
did. Only the surviving ``cnt``/``idx`` leave, which is 32-256x smaller.

**Tiling (the thing that makes this scale).** The score *vector* a program holds is
``[Nkv]`` fp32 — 8 KB at 128k tokens, which is fine in registers. The **state tile**
is what bites: sized at the full padded ``Nkv`` it is ``[Nkv, D]`` fp32 = 512 KB per
program at 64k, far past a register file, so every program spills to local memory.
Measured before tiling: selection was 18 ms at 64k (43% of the step) for a
one-``Dot`` policy and 60 ms (72%) for a two-field one, while a policy with no
``Dot`` at all stayed flat at 0.04 ms — which is what localized it to the state
tile rather than to the top-k or the scorer chain.

So the KV axis is walked in tiles of ``TILE_N`` blocks: the state tile is
``[TILE_N, D]`` (32 KB), the whole scorer chain runs per tile, and the tile's scores
are placed into the full-width score vector. The top-k afterwards is still a single
exact pass over all ``Nkv``. Tiles entirely outside the causal region are skipped
without loading, which removes about half the work on average under causal masking.

The scorer chain is **generated as Triton source**, one kernel per policy, rather
than interpreted from a tape at runtime. Two reasons, in order of importance:

1. *Straight-line code.* An interpreted tape would emit every op's body at every
   node and branch on a loaded value each step, so the "fusion" would be nominal:
   the register pressure and instruction count of the whole op set at every
   position. Generated source emits only the ops the policy uses.
2. Triton cannot express the interpreter anyway — a graph walk needs to index a
   list of register tensors by node id, and there is no dynamic list in the
   language.

Cost is one Triton compile per distinct (policy, shape) pair, paid once and cached.

Then, in one pass:

1. **score** — the generated chain, per tile, in registers.
2. **mask + reserve** — forbid non-causal blocks, then force reserved blocks
   (BOS / local / EOS) to the top by score boosting rather than a separate merge.
   Boosting is what makes dedup structurally free: a block that is both reserved
   and high-scoring is one entry either way, so no double-count is possible.
3. **top-k** — iterative argmax over the register vector.

The top-k is ``topk`` passes of argmax rather than a sort. At the budgets that
matter (``topk`` 8-64 against ``Nkv`` up to 2048) that is simpler and faster than a
bitonic sort, and — more importantly — *exactly* specifiable, which is what makes
the index-equivalence test against an independent selector possible. Ties break
toward the lower block index.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import tempfile

import torch
import triton
import triton.language as tl

# Op codes, mirrored in compiler/compile.py. Small and closed on purpose: the
# scorer's expressible set is deliberately narrow (see flow/ops.py).
OP_CONST, OP_DOT, OP_SCALE, OP_NORM = 0, 1, 2, 3
OP_DISTANCE, OP_ADD, OP_SUB, OP_MUL = 4, 5, 6, 7
OP_ENVELOPE = 8

# `group_reduce` codes for OP_DOT
GR_MAX, GR_MEAN, GR_SUM = 0, 1, 2

# A boost far above any real score, so reserved blocks always outrank scored ones
# while relative order among reserved blocks stays meaningful. Finite (not inf) so
# arithmetic on a boosted score cannot produce NaN.
RESERVE_BOOST = 1.0e30

# KV blocks per tile. 64 keeps the fp32 state tile at 32 KB for D=128, which fits
# comfortably; larger tiles regain little because the loop is already
# memory-bound, and smaller ones underfill the reduction.
TILE_N = 64


_PRELUDE = """import triton
import triton.language as tl

from vortex_train.kernels.score_ops import (
    q_summary, score_dot, score_envelope, score_norm,
)
"""

# `{BODY}` becomes the generated per-tile scorer chain, assigning `s_tile`.
_TEMPLATE = '''
@triton.jit
def _select_kernel(
    Q, STATE, CNT, IDX,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_sb, stride_sh, stride_sn, stride_sf, stride_su, stride_sd,
    stride_cb, stride_ch, stride_cm,
    stride_ib, stride_ih, stride_im, stride_ik,
    seqlen_q, seqlen_kv, num_kv_blocks,
    GROUP: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_N: tl.constexpr, NTILES: tl.constexpr,
    TOPK: tl.constexpr, RES_BOS: tl.constexpr, RES_LOCAL: tl.constexpr,
    RES_EOS: tl.constexpr, CAUSAL: tl.constexpr, Q_HOW: tl.constexpr,
    BOOST: tl.constexpr,
):
    """One program per (query block, batch, kv-head). Emits that row of cnt/idx."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)          # kv head: selection is shared by the group
    pid_b = tl.program_id(2)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = pid_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    m_mask = offs_m < seqlen_q
    cnt_m = tl.maximum(tl.sum(m_mask.to(tl.float32), axis=0), 1.0)

    # q is the same for every KV tile, so it is summarised ONCE here rather than
    # re-read per tile (which would be BLOCK_Q x D of redundant traffic per tile).
    qs = q_summary(Q, stride_qb, stride_qh, stride_qm, stride_qd,
                   pid_b, pid_h, offs_m, offs_d, m_mask, cnt_m,
                   GROUP, BLOCK_Q, HEAD_DIM, Q_HOW)

    # Last KV block this query block may see, under causal masking.
    if CAUSAL:
        q_end = (pid_m + 1) * BLOCK_Q - 1 + (seqlen_kv - seqlen_q)
        max_blk = q_end // BLOCK_KV
    else:
        max_blk = num_kv_blocks - 1

    # Scores accumulate as [NTILES, TILE_N] and are reshaped to [BLOCK_N] for the
    # top-k. 2D because a tile's result cannot be scattered into a flat register
    # vector by a runtime index, but it can be selected into a row by a static one.
    scores = tl.full((NTILES, TILE_N), float("-inf"), dtype=tl.float32)

    for t in tl.static_range(NTILES):
        tile_lo = t * TILE_N
        # Skip tiles wholly past the causal edge: no load, no scorer work. Under
        # causal masking this removes ~half the tiles on average.
        if tile_lo <= max_blk:
            offs_n = tile_lo + tl.arange(0, TILE_N)
            valid_n = (offs_n < num_kv_blocks) & (offs_n <= max_blk)

            # ---- generated scorer chain (assigns `s_tile`) -----------------
{BODY}

            s_tile = tl.where(valid_n, s_tile, float("-inf"))

            # reservations, applied per tile while the block ids are in hand
            if RES_BOS > 0:
                s_tile = tl.where(valid_n & (offs_n < RES_BOS), s_tile + BOOST, s_tile)
            if RES_LOCAL > 0:
                lo_local = pid_m - (RES_LOCAL - 1)
                s_tile = tl.where(
                    valid_n & (offs_n <= pid_m) & (offs_n >= lo_local),
                    s_tile + BOOST, s_tile,
                )
            if RES_EOS > 0:
                s_tile = tl.where(
                    valid_n & (offs_n >= num_kv_blocks - RES_EOS), s_tile + BOOST, s_tile
                )

            row = (tl.arange(0, NTILES) == t)[:, None]
            scores = tl.where(row, s_tile[None, :], scores)

    score = tl.reshape(scores, (BLOCK_N,))
    offs_all = tl.arange(0, BLOCK_N)

    # ---- top-k by iterative argmax, low index wins ties -------------------
    kept = 0
    base_i = IDX + pid_b * stride_ib + pid_h * stride_ih + pid_m * stride_im
    for j in tl.static_range(TOPK):
        best = tl.max(score, axis=0)
        cand = tl.where(score == best, offs_all, num_kv_blocks + 1)
        pos = tl.min(cand, axis=0)
        take = best > float("-inf")
        tl.store(base_i + j * stride_ik, tl.where(take, pos, -1).to(IDX.dtype.element_ty))
        kept += tl.where(take, 1, 0)
        score = tl.where(offs_all == pos, float("-inf"), score)   # retire the winner

    tl.store(CNT + pid_b * stride_cb + pid_h * stride_ch + pid_m * stride_cm,
             kept.to(CNT.dtype.element_ty))
'''

_IND = " " * 12  # body sits inside `for t` + `if tile_lo <= max_blk`

_DOT_CALL = (
    "{ind}v{i} = score_dot(qs, STATE, stride_sb, stride_sh, stride_sn, stride_sf,\n"
    "{ind}                 stride_su, stride_sd, pid_b, pid_h, offs_n, offs_d,\n"
    "{ind}                 valid_n, {fslot}, {gred}, GROUP, TILE_N, {nsub})"
)
_ENVELOPE_CALL = (
    "{ind}v{i} = score_envelope(qs, STATE, stride_sb, stride_sh, stride_sn,\n"
    "{ind}                      stride_sf, stride_su, stride_sd, pid_b, pid_h,\n"
    "{ind}                      offs_n, offs_d, valid_n, {fslot}, {fslot2},\n"
    "{ind}                      {gred}, GROUP, TILE_N, {nsub})"
)
_NORM_CALL = (
    "{ind}v{i} = score_norm(STATE, stride_sb, stride_sh, stride_sn, stride_sf,\n"
    "{ind}                  stride_su, stride_sd, pid_b, pid_h, offs_n, offs_d,\n"
    "{ind}                  valid_n, {fslot}, TILE_N, {nsub})"
)


def _emit_body(tape: dict) -> str:
    """Lower the tape to Triton source lines. Pure text; no device interaction."""
    lines: list[str] = []
    for i in range(tape["num_nodes"]):
        op = tape["ops"][i]
        a0, a1 = tape["arg0"][i], tape["arg1"][i]
        if op == OP_DOT:
            lines.append(_DOT_CALL.format(
                ind=_IND, i=i, fslot=tape["fslot"][i], gred=tape["gred"][i],
                nsub=tape["nsub"][i]))
        elif op == OP_ENVELOPE:
            lines.append(_ENVELOPE_CALL.format(
                ind=_IND, i=i, fslot=tape["fslot"][i], fslot2=tape["fslot2"][i],
                gred=tape["gred"][i], nsub=tape["nsub"][i]))
        elif op == OP_NORM:
            lines.append(_NORM_CALL.format(ind=_IND, i=i, fslot=tape["fslot"][i],
                                           nsub=tape["nsub"][i]))
        elif op == OP_DISTANCE:
            lines.append(f"{_IND}v{i} = -(pid_m - offs_n).to(tl.float32)")
        elif op == OP_CONST:
            lines.append(f"{_IND}v{i} = tl.zeros((TILE_N,), dtype=tl.float32) "
                         f"+ {tape['cval'][i]!r}")
        elif op == OP_SCALE:
            lines.append(f"{_IND}v{i} = v{a0} * {tape['cval'][i]!r}")
        elif op == OP_ADD:
            lines.append(f"{_IND}v{i} = v{a0} + v{a1}")
        elif op == OP_SUB:
            lines.append(f"{_IND}v{i} = v{a0} - v{a1}")
        elif op == OP_MUL:
            lines.append(f"{_IND}v{i} = v{a0} * v{a1}")
        else:
            raise AssertionError(f"unknown opcode {op} at tape position {i}")
    lines.append(f"{_IND}s_tile = v{tape['out_node']}")
    return "\n".join(lines)


def generated_source(tape: dict) -> str:
    """The emitted kernel source. For debugging a policy, and for tests."""
    return _TEMPLATE.replace("{BODY}", _emit_body(tape))


_KERNEL_CACHE: dict[tuple, object] = {}


def _tape_key(tape: dict) -> tuple:
    return (
        tuple(tape["ops"]), tuple(tape["arg0"]), tuple(tape["arg1"]),
        tuple(tape["fslot"]), tuple(tape["fslot2"]), tuple(tape["gred"]),
        tuple(tape["cval"]), tuple(tape["nsub"]), tape["out_node"],
    )


def build_select_kernel(tape: dict):
    """Generate (or fetch) the Triton kernel specialised to this policy's tape.

    The generated source is written to a real file rather than ``exec``'d from a
    string, because ``@triton.jit`` reads its own source via
    ``inspect.getsourcelines`` and rejects a function with no file. The side benefit
    is that a generated kernel is inspectable: when a policy misbehaves, the exact
    code that ran is on disk.
    """
    key = _tape_key(tape)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    src = _PRELUDE + generated_source(tape)
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
    cache_dir = pathlib.Path(tempfile.gettempdir()) / "vortex_train_kernels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"select_{digest}.py"
    if not path.exists():
        # Write once, atomically: two processes compiling the same policy must not
        # observe a half-written file.
        tmp = path.with_suffix(f".{id(tape):x}.tmp")
        tmp.write_text(src)
        tmp.replace(path)

    spec = importlib.util.spec_from_file_location(f"vortex_train_select_{digest}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = mod._select_kernel
    _KERNEL_CACHE[key] = kernel
    return kernel


def score_and_select(
    q: torch.Tensor,
    state: torch.Tensor,
    tape: dict,
    *,
    num_kv_blocks: int,
    seqlen_kv: int,
    block_q: int,
    block_kv: int,
    num_kv_heads: int,
    topk: int,
    reserve_bos: int,
    reserve_local: int,
    reserve_eos: int,
    causal: bool,
    q_how: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused scorer + top-k. Returns ``(cnt, idx)`` for a ``SparsePattern``.

    Everything is shape-derived: no value is read back from the device, so the
    launch grid never depends on a device value and the path stays cudagraph-safe.
    """
    b, hq, sq, d = q.shape
    group = hq // num_kv_heads
    m = (sq + block_q - 1) // block_q
    k_eff = min(topk, num_kv_blocks)

    tile_n = min(TILE_N, triton.next_power_of_2(num_kv_blocks))
    # NTILES must be a power of two: the kernel holds scores as a [NTILES, TILE_N]
    # register tile and `tl.full` requires power-of-two extents. Rounding up adds
    # trailing tiles whose blocks are all `>= num_kv_blocks`, so they are masked
    # -inf and the top-k ignores them -- correct, and skipped cheaply by the causal
    # `tile_lo <= max_blk` guard. Without this, 71% of lengths in the 12k-40k range
    # failed to compile (e.g. seqlen 12288 -> n_kv 192 -> NTILES 3).
    n_tiles = triton.next_power_of_2(triton.cdiv(num_kv_blocks, tile_n))
    block_n = n_tiles * tile_n          # padded width the score vector spans

    cnt = torch.empty((b, num_kv_heads, m), dtype=torch.int32, device=q.device)
    idx = torch.full((b, num_kv_heads, m, k_eff), -1, dtype=torch.int32, device=q.device)

    build_select_kernel(tape)[(m, num_kv_heads, b)](
        q, state, cnt, idx,
        *q.stride(), *state.stride(), *cnt.stride(), *idx.stride(),
        sq, seqlen_kv, num_kv_blocks,
        GROUP=group, BLOCK_Q=block_q, BLOCK_KV=block_kv, HEAD_DIM=d,
        BLOCK_N=block_n, TILE_N=tile_n, NTILES=n_tiles,
        TOPK=k_eff, RES_BOS=reserve_bos, RES_LOCAL=reserve_local, RES_EOS=reserve_eos,
        CAUSAL=causal, Q_HOW=0 if q_how == "mean" else 1,
        BOOST=RESERVE_BOOST,
        num_warps=4,
    )
    return cnt, idx
