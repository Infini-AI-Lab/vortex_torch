"""Pure-metadata tensor type used throughout the vortex graph + compiler.

A :class:`vTensor` does **not** carry any real storage. It is a small
descriptor that records exactly what the rest of the system needs to
reason about a tensor at graph-construction and codegen time:

  * ``shape``        — tuple of ints, the *real* logical shape.
  * ``padded_shape`` — tuple of ints, each entry rounded up to the next
                       power of two (only ``shape[1]`` / ``shape[2]`` may
                       differ from ``shape``; ``shape[0]`` is the leading
                       axis and is not subject to Triton block-shape
                       pow2 constraints).
  * ``dtype``        — :class:`torch.dtype`
  * ``device``       — :class:`torch.device` / str / ``None``
  * ``_format``      — :class:`FORMAT` (BATCHED / RAGGED / PAGED)
  * ``tensor_id``    — int, the graph-level identity used by the compiler

Why ``padded_shape``: Triton's block-shape constexprs (``tl.zeros``,
``tl.arange``, ``tl.reshape``, ``tl.make_block_ptr.block_shape``, etc.)
must each be a power of two. Real models (e.g. Qwen3-14B with
``num_attention_heads // num_key_value_heads == 5``) carry tensors whose
``shape[1]`` is **not** a power of two. The compiler emits tile sizes
from ``padded_shape`` while keeping memory-addressing math (strides,
per-row offsets, divisors in ``Mean``) anchored to ``shape``; codegens
add load/store masks that select the real-shape lanes whenever
``padded_shape != shape``. When ``shape`` is already pow2,
``padded_shape == shape`` and no masking is emitted, so power-of-two
models pay no overhead.

It also exposes ``dim()`` for parity with ``torch.Tensor`` so existing
profile-time validation code (``assert x.dim() == 3``) keeps working.

There is intentionally **no** torch op support, no ``__torch_function__``
override, no parent ``torch.Tensor`` class. ``vTensor`` is just metadata.
Real tensors used by the runtime/execute path stay as plain
``torch.Tensor`` instances.
"""

from __future__ import annotations
import torch
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union


class FORMAT(Enum):
    """Tensor storage/layout format.

    Attributes:
        BATCHED: Standard dense batched tensors (e.g., ``[B, N, D]``).
        RAGGED: Ragged tensors with variable-length sequences or elements per batch.
        PAGED: Paged tensors used for large or streaming data split into pages/chunks.
        PARAMETER: A learned constant **shared across the batch** (a
            :class:`~vortex_torch.indexer.Parameter`). Carries no per-request /
            per-page axis; its value is baked into the compiled function. An op
            that receives a PARAMETER operand (e.g. ``GeMM``) runs as a
            standalone ``Schedule.S`` ``torch.matmul`` instead of the fused
            per-workload kernel (so a large weight never enters the tiled kernel).
        SLOTTED: Like PAGED in shape — ``[n_slots, r, c]`` — but addressed
            through a device-side ``block_id -> slot`` map instead of by
            ``block_id`` directly, and sized by **concurrency** rather than by
            context length. This is the REQUEST-bound cache domain: state that
            belongs to a request in flight rather than to a stored page (the
            motivating case is INT4's bf16 staging area for blocks whose scale
            is not yet computable).

            It is a *format*, not a separate set of kernels, and that is the
            whole point: a SLOTTED tensor is an ordinary compiler operand, so a
            flow can read both domains in **one fused kernel** with no extra
            launch. The generated addressing differs by exactly one indirection::

                PAGED    ->  off = block_id * (r * c)
                SLOTTED  ->  slot = load(slot_of_ptr + block_id)
                             off  = max(slot, 0) * (r * c)

            The ``max(slot, 0)`` keeps a miss (``slot == -1``) in bounds; the
            load/store is separately masked so a miss reads zeros and writes
            nothing. Only ``shape[0]`` differs between the two domains, so every
            inner-axis codegen path (padding, masks, dtype casts) is shared
            verbatim.
    """

    BATCHED = 0
    RAGGED = 1
    PAGED = 2
    PARAMETER = 3
    SLOTTED = 4


def _next_pow2(n: int) -> int:
    """Round ``n`` up to the next power of two.

    ``_next_pow2(1) == 1``. For ``n <= 0`` returns ``1`` (defensive; the
    compiler never asks about non-positive dims). For positive pow2
    inputs the result is the input.
    """
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _compute_padded_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    """Return ``shape`` with ``shape[1]`` and ``shape[2]`` rounded up to
    the next power of two. Other axes are passed through unchanged.

    Rationale: Triton block-shape constexprs at axes 1 and 2 of the
    per-workload tile (``[chunk, D_0, D_1]``) must be pow2. The leading
    axis (``shape[0]``) is the ragged/paged buffer count and is sized
    by ``workload_chunk_size`` / ``num_blocks_per_page`` (already pow2
    by config). Padding only the inner two axes keeps the change
    minimal and avoids touching addressing math that depends on
    ``shape[0]``.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        return shape
    padded = list(shape)
    for i in (1, 2):
        if i < len(padded):
            padded[i] = _next_pow2(padded[i])
    return tuple(padded)


class vTensor:
    """Pure-metadata virtual tensor.

    Carries the descriptor fields used by the graph builder, the
    compiler, and the codegen layer. It does not own any GPU / CPU
    memory and intentionally cannot participate in torch ops — code
    that wants to compute on real data should hold a ``torch.Tensor``
    separately and use the ``vTensor`` only for graph bookkeeping.
    """

    __slots__ = ("shape", "padded_shape", "dtype", "device", "_format", "tensor_id")

    shape: tuple
    padded_shape: tuple
    dtype: torch.dtype
    device: Optional[Union[torch.device, str]]
    _format: FORMAT
    tensor_id: int

    def __init__(
        self,
        shape: Sequence[int] = (),
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[torch.device, str]] = None,
        _format: FORMAT = FORMAT.BATCHED,
        tensor_id: int = -1,
        padded_shape: Optional[Sequence[int]] = None,
    ) -> None:
        if not isinstance(tensor_id, int):
            raise TypeError(f"tensor_id must be int, got {type(tensor_id).__name__}")
        if not isinstance(_format, FORMAT):
            raise TypeError(f"_format must be a FORMAT enum, got {type(_format).__name__}")

        # Normalize ``shape`` so ``shape[i]``, ``len(shape)`` and ``tuple(shape)``
        # all behave like ``torch.Tensor.shape``.
        self.shape = tuple(int(s) for s in shape)
        # ``padded_shape`` derives from ``shape`` by default; callers may
        # override only when they're hand-building a tensor from a
        # source that already has a padded view (rare — pickle/copy
        # path uses this).
        if padded_shape is None:
            self.padded_shape = _compute_padded_shape(self.shape)
        else:
            self.padded_shape = tuple(int(s) for s in padded_shape)
        self.dtype = dtype
        self.device = device
        self._format = _format
        self.tensor_id = tensor_id

    # -------- shape helpers --------
    def dim(self) -> int:
        """Number of dimensions; mirrors :meth:`torch.Tensor.dim`."""
        return len(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    def size(self, dim: Optional[int] = None):
        """Mirror of :meth:`torch.Tensor.size`."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def needs_padding(self) -> bool:
        """True iff ``padded_shape != shape`` — i.e. at least one inner
        dim is not already a power of two and codegen must emit
        load/store masks.
        """
        return self.padded_shape != self.shape

    # -------- repr --------
    def __repr__(self) -> str:
        pad = "" if self.padded_shape == self.shape else f", padded={self.padded_shape}"
        return (
            f"vTensor(shape={self.shape}{pad}, dtype={self.dtype}, "
            f"device={self.device}, _format={self._format}, "
            f"tensor_id={self.tensor_id})"
        )

    # -------- pickle / copy --------
    def __reduce__(self):
        return (
            _rebuild_vtensor,
            (self.shape, self.dtype, self.device, self._format, self.tensor_id, self.padded_shape),
        )


def _rebuild_vtensor(shape, dtype, device, _format, tensor_id, padded_shape=None):
    return vTensor(
        shape=shape, dtype=dtype, device=device, _format=_format,
        tensor_id=tensor_id, padded_shape=padded_shape,
    )


# -------- convenience factory --------
def as_vtensor(
    x: Any = None,
    _format: FORMAT = FORMAT.BATCHED,
    tensor_id: int = -1,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> vTensor:
    """Build a :class:`vTensor`.

    Three calling styles, all returning a fresh ``vTensor`` (or, for an
    existing ``vTensor``, the same object re-tagged):

    1. **Re-tag an existing vTensor** — ``as_vtensor(vt, fmt, tid)``
       overwrites ``vt._format`` and ``vt.tensor_id`` in place and
       returns ``vt``. Useful when the caller wants to add an existing
       tensor descriptor to the graph under a fresh id. ``padded_shape``
       is preserved.

    2. **Extract metadata from a torch.Tensor** — ``as_vtensor(real, fmt, tid)``
       reads ``shape``, ``dtype``, ``device`` from ``real`` and returns
       a brand-new ``vTensor``. ``padded_shape`` is derived from
       ``shape``. The original tensor is **not** retained — vTensor is
       pure metadata.

    3. **Direct construction by kwargs** —
       ``as_vtensor(_format=fmt, tensor_id=tid, shape=..., dtype=..., device=...)``.
       Use this when no real torch tensor is available (the common case
       once the compile path is fully virtualized).
    """
    if isinstance(x, vTensor):
        x._format = _format
        x.tensor_id = tensor_id
        return x

    if isinstance(x, torch.Tensor):
        return vTensor(
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            _format=_format,
            tensor_id=tensor_id,
        )

    if x is None:
        return vTensor(
            shape=shape if shape is not None else (),
            dtype=dtype if dtype is not None else torch.bfloat16,
            device=device,
            _format=_format,
            tensor_id=tensor_id,
        )

    raise TypeError(
        f"as_vtensor: cannot convert {type(x).__name__} to vTensor; "
        "pass a torch.Tensor, an existing vTensor, or shape/dtype/device kwargs."
    )


if __name__ == "__main__":
    # Direct construction
    a = vTensor(shape=(2, 3, 4), dtype=torch.bfloat16, device="cuda:0",
                _format=FORMAT.RAGGED, tensor_id=0)
    print("a:", a, "padded:", a.padded_shape, "needs_padding:", a.needs_padding())

    # Pow2 stays unchanged
    p = vTensor(shape=(8, 4, 128), dtype=torch.bfloat16,
                _format=FORMAT.BATCHED, tensor_id=1)
    print("p:", p, "needs_padding:", p.needs_padding())

    # Non-pow2 inner dim rounds up
    q = vTensor(shape=(8, 5, 128), dtype=torch.bfloat16,
                _format=FORMAT.BATCHED, tensor_id=2)
    print("q:", q, "padded:", q.padded_shape, "needs_padding:", q.needs_padding())
    assert q.padded_shape == (8, 8, 128)
