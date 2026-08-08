"""Bridging the two cuda-graph metadata ABIs an ``AttentionBackend`` can speak.

sglang 0.5.16 replaced the pre-0.5.16 pair

    init_forward_metadata_capture_cuda_graph(bs, num_tokens, req_pool_indices,
        seq_lens, encoder_lens, forward_mode, spec_info)
    init_forward_metadata_replay_cuda_graph(bs, req_pool_indices, seq_lens,
        seq_lens_sum, encoder_lens, forward_mode, spec_info, seq_lens_cpu)

with a single method the graph runner calls once at capture
(``in_capture=True``) and again before every replay (``in_capture=False``):

    init_forward_metadata_out_graph(forward_batch, in_capture=False)

vortex sits on *both* sides of that change, in opposite directions:

* its five backends **implement** the legacy pair, and 0.5.16's runners no
  longer call it — silently, so graph metadata would simply never initialize
  (:class:`LegacyCudaGraphABIMixin` adapts them);
* its MLA backends **call** the legacy pair on a wrapped upstream backend
  (``self._dense``, e.g. ``TritonAttnBackend``) that no longer has it
  (:func:`capture_dense` / :func:`replay_dense` adapt those calls).

Both directions are the same mapping between one flat argument list and one
ForwardBatch-like object, so that mapping is written once, as
:class:`GraphMetadataArgs`, and each adapter picks a direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional


@dataclass(frozen=True)
class GraphMetadataArgs:
    """The per-batch facts a backend needs to (re)build cuda-graph metadata.

    Deliberately the union of the legacy pair's parameters and the subset of
    ``ForwardBatch`` that 0.5.16's runners populate, so it can be built from
    either ABI and rendered into the other.
    """

    bs: int
    req_pool_indices: Any
    seq_lens: Any
    forward_mode: Any
    encoder_lens: Optional[Any] = None
    spec_info: Optional[Any] = None
    seq_lens_cpu: Optional[Any] = None
    seq_lens_sum: Optional[int] = None

    @classmethod
    def from_forward_batch(cls, forward_batch) -> "GraphMetadataArgs":
        """Read the 0.5.16 ForwardBatch-like view the graph runner passes.

        The runner supplies these via ``build_replay_fb_view``
        (``model_executor/runner/decode_cuda_graph_runner.py``); the optional
        fields are absent on some paths, hence ``getattr``.
        """
        return cls(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_mode=forward_batch.forward_mode,
            encoder_lens=getattr(forward_batch, "encoder_lens", None),
            spec_info=getattr(forward_batch, "spec_info", None),
            seq_lens_cpu=getattr(forward_batch, "seq_lens_cpu", None),
            seq_lens_sum=getattr(forward_batch, "seq_lens_sum", None),
        )

    def to_forward_batch(self):
        """Render a ForwardBatch-like view for a 0.5.16 ``_out_graph`` call.

        vortex only drives single-token decode graphs, so ``num_tokens == bs``
        and there is no padding.
        """
        seq_lens_sum = self.seq_lens_sum
        if seq_lens_sum is None and self.seq_lens is not None:
            seq_lens_sum = int(self.seq_lens.sum())
        return SimpleNamespace(
            batch_size=self.bs,
            forward_mode=self.forward_mode,
            actual_forward_mode=self.forward_mode,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_cpu=self.seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=self.encoder_lens,
            spec_info=self.spec_info,
            num_padding=0,
            out_cache_loc=None,
        )


# --------------------------------------------------------------------------- #
# vortex CALLS a wrapped upstream backend (MLA dense layers)
# --------------------------------------------------------------------------- #
def capture_dense(dense, args: GraphMetadataArgs) -> None:
    """Initialize a wrapped upstream backend's metadata for graph CAPTURE."""
    legacy = getattr(dense, "init_forward_metadata_capture_cuda_graph", None)
    if legacy is not None:
        legacy(
            args.bs, args.bs, args.req_pool_indices, args.seq_lens,
            args.encoder_lens, args.forward_mode, args.spec_info,
        )
        return
    dense.init_forward_metadata_out_graph(args.to_forward_batch(), in_capture=True)


def replay_dense(dense, args: GraphMetadataArgs) -> None:
    """Refresh a wrapped upstream backend's metadata before graph REPLAY."""
    legacy = getattr(dense, "init_forward_metadata_replay_cuda_graph", None)
    if legacy is not None:
        legacy(
            args.bs, args.req_pool_indices, args.seq_lens, args.seq_lens_sum,
            args.encoder_lens, args.forward_mode, args.spec_info,
            args.seq_lens_cpu,
        )
        return
    dense.init_forward_metadata_out_graph(args.to_forward_batch())


# --------------------------------------------------------------------------- #
# upstream CALLS vortex's backends
# --------------------------------------------------------------------------- #
class LegacyCudaGraphABIMixin:
    """Serves 0.5.16's ``_out_graph`` from vortex's legacy capture/replay pair.

    Mixed in ahead of ``AttentionBackend`` (see :func:`attention_backend_base`)
    so it overrides the base no-op. ``init_forward_metadata_in_graph`` is left
    as that no-op on purpose: vortex's planning is host-side, so it contributes
    no graph-recordable ops. Never installed on releases that still call the
    legacy pair directly.
    """

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        args = GraphMetadataArgs.from_forward_batch(forward_batch)
        if in_capture:
            # Legacy capture asserts bs == num_tokens; single-token decode graphs.
            self.init_forward_metadata_capture_cuda_graph(
                bs=args.bs,
                num_tokens=args.bs,
                req_pool_indices=args.req_pool_indices,
                seq_lens=args.seq_lens,
                encoder_lens=args.encoder_lens,
                forward_mode=args.forward_mode,
                spec_info=args.spec_info,
            )
            return
        self.init_forward_metadata_replay_cuda_graph(
            bs=args.bs,
            req_pool_indices=args.req_pool_indices,
            seq_lens=args.seq_lens,
            seq_lens_sum=args.seq_lens_sum,
            encoder_lens=args.encoder_lens,
            forward_mode=args.forward_mode,
            spec_info=args.spec_info,
            seq_lens_cpu=args.seq_lens_cpu,
        )


def _runner_drives_out_graph() -> bool:
    """True when this sglang drives graph metadata through ``_out_graph``."""
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    return hasattr(AttentionBackend, "init_forward_metadata_out_graph")


def attention_backend_base() -> tuple:
    """Base classes for a vortex attention backend.

    Unpack into the class statement so the ABI adapter is interposed only on the
    releases that need it::

        class VortexFooBackend(*attention_backend_base()):
            ...
    """
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    if _runner_drives_out_graph():
        return (LegacyCudaGraphABIMixin, AttentionBackend)
    return (AttentionBackend,)
