r"""Trainable per-head block compressor for vortex sparse attention.

A small learned module that compresses a block of KV (the fused MLA latent) into
a compact per-head descriptor used to *score* blocks for sparse selection,
trained to match the real attention of a HuggingFace model. It is a low-rank
bilinear generalization of the centroid block scorer:

    s[h, b] = scaling · (Wqᵀ q_h) · (Wkᵀ centroid_b)

Supervision is computed **on the fly** from a frozen HF MLA model (no traces are
written to disk): :class:`MLASupervision` reconstructs the absorbed query +
latent each forward, and the trainer distills the compressor's per-head block
distribution against the true per-block attention mass.

Components:
  * :class:`CompressorConfig`  — geometry + hyper-params.
  * :class:`BlockCompressor`   — the learned scorer (``torch.nn.Module``).
  * :class:`MLASupervision`    — frozen HF MLA model → (latent, absorbed-q) stream.
  * :mod:`objective`           — targets, distillation loss, coverage/recall eval.
  * :mod:`train`               — ``python -m vortex_torch.compressor.train`` CLI.
"""
from .config import CompressorConfig
from .model import BlockCompressor

__all__ = ["CompressorConfig", "BlockCompressor", "MLASupervision"]


def __getattr__(name):
    # Lazy: importing MLASupervision pulls in transformers; keep the package
    # importable (for the model/objective) without it.
    if name == "MLASupervision":
        from .capture import MLASupervision
        return MLASupervision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
