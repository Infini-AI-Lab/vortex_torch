"""``Vortex.Parameter`` — a learned constant operand shared across the batch.

A :class:`Parameter` is a ``vTensor`` with ``_format == FORMAT.PARAMETER``: it
carries a baked weight value (no per-request / per-page axis) plus an optional
per-layer table. It is declared in a flow's ``__init__`` and passed straight to
an existing op (today: :class:`~vortex_torch.indexer.GeMM`). When ``GeMM`` sees a
PARAMETER operand it runs as a standalone ``Schedule.S`` ``torch.matmul`` over
the gathered (per-layer) slice — so a large weight never enters the fused
per-workload Triton kernel.

The value is held on the Python object (reached at runtime via
``ctx.op_list[<gemm_id>]``, the same producer-less-constant trick ``Conv1d``
uses), moved to device + bf16 at compile time (``materialize`` from the
consuming op's ``profile``, before any cuda-graph capture), and the per-layer
row is selected by the explicit ``cur_layer`` argument with a host python lookup
(no ``.item()`` / device sync during capture).
"""
from __future__ import annotations

from typing import Optional

import torch

from .tensor import vTensor, FORMAT


class Parameter(vTensor):
    r"""A batch-shared learned constant ``[L, N, K]`` (per-layer) operand.

    :__init__: ``Parameter(value, layer_lookup=None)`` —
        * ``value`` : ``torch.Tensor`` ``[L, N, K]`` (per-layer) or ``[N, K]``
          (single, treated as ``L=1``). ``K`` is the contraction dim (must match
          the activation operand's last dim); ``N`` is the output dim.
        * ``layer_lookup`` : optional 1-D int tensor of length
          ``max(global_layer_id)+1`` mapping a global layer id → row in
          ``value``; default = identity (row == layer id, clamped).
    """

    __slots__ = ("value", "layer_lookup", "_lookup_cpu")

    def __init__(self, value: torch.Tensor, layer_lookup: Optional[torch.Tensor] = None):
        assert isinstance(value, torch.Tensor), (
            f"Parameter: value must be a torch.Tensor, got {type(value)}"
        )
        if value.dim() == 2:
            value = value.unsqueeze(0)                         # [N,K] -> [1,N,K]
        assert value.dim() >= 3, (
            f"Parameter: value must be [L, ...] (>=3D) or [N,K], got {tuple(value.shape)}"
        )
        # Leading axis is the per-layer axis; the trailing two are the (N, K)
        # contract dims that existing ops' rank/K checks read. Higher-rank
        # values (e.g. per-(layer,head) [L,H,d,r]) keep their full shape on
        # ``self.value``; only the metadata view is collapsed to (0, N, K).
        N, K = value.shape[-2], value.shape[-1]
        # 3D metadata view [0, N, K] so existing ops' rank/K checks pass; the
        # leading 0 marks "no per-request axis" (shared across batch).
        super().__init__(shape=(0, N, K), dtype=value.dtype, device=value.device,
                         _format=FORMAT.PARAMETER, tensor_id=-1)
        self.value = value
        self.layer_lookup = layer_lookup
        self._lookup_cpu: Optional[list] = None

    # ------------------------------------------------------------------ #
    def materialize(self, device, dtype: torch.dtype = torch.bfloat16) -> None:
        """Move the value to ``device`` + ``dtype`` and snapshot the layer lookup
        as a host python list. Call ONCE at compile time (pre cuda-graph capture)
        from the consuming op's ``profile`` — keeps the runtime gather sync-free."""
        self.value = self.value.to(device=device, dtype=dtype).contiguous()
        L = self.value.shape[0]
        if self.layer_lookup is None:
            self._lookup_cpu = list(range(L))
        else:
            self._lookup_cpu = self.layer_lookup.to("cpu", torch.long).tolist()

    def gather(self, cur_layer: int) -> torch.Tensor:
        """Return the ``[N, K]`` slice for global layer ``cur_layer`` (host-side
        int index → a view; no device sync). Clamps out-of-range to row 0."""
        lid = int(cur_layer)
        if self._lookup_cpu is not None and 0 <= lid < len(self._lookup_cpu):
            row = self._lookup_cpu[lid]
        else:
            row = lid
        L = self.value.shape[0]
        if row < 0 or row >= L:
            row = 0 if L == 1 else max(0, min(row, L - 1))
        return self.value[row]                                 # [N, K]
