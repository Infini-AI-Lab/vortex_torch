"""Configuration for the trainable per-head block compressor."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class CompressorConfig:
    """Geometry + hyper-params for a :class:`BlockCompressor`.

    The compressor learns, per (layer, head), a low-rank bilinear block scorer
    that generalizes the centroid scorer. With ``pool="mean"`` the block score is

        s[h, b] = scaling * (Wq[h]ᵀ q[h]) · (Wk[h]ᵀ centroid_b)

    where ``centroid_b = mean_{t∈b} latent_t``. Setting ``proj_dim == latent_dim``
    and ``Wk = Wq = I`` recovers exactly ``scaling * q[h] · centroid_b`` — the
    ``centroid`` baseline. So training starts at (a truncation of) centroid and
    learns to do better.

    For MLA (GLM-4.7-Flash / DeepSeek) there is one shared latent KV "head", so
    ``latent_dim = kv_lora_rank + qk_rope_head_dim`` (576 for GLM) and the heads
    here are the *query* heads.
    """
    latent_dim: int                 # KV/descriptor input width (576 for GLM MLA;
                                    #   == head_dim, e.g. 128, for MHA/GQA)
    num_q_heads: int                # number of query heads (per-head params)
    num_kv_heads: int = 0           # MHA/GQA: KV heads (cache-side params are per
                                    #   (layer, kv_head); 0 = MLA shared latent)
    proj_dim: int = 128             # r = d_c — descriptor rank (compression: r < latent_dim)
    arch: str = "bilinear"          # block-scorer architecture (see model.ARCH_REGISTRY)
    pool: str = "mean"              # within-block pooling: "mean" (linear) or "max"
    num_landmarks: int = 1          # m = b_c — arch="landmark"/"factorized"/"gqa_factorized":
                                    #   descriptors per block (max-scored)
    hidden_dim: int = 0             # arch="mlp": hidden width of the per-head MLP heads
    block_size: int = 16            # tokens per block — sizes the learned token-mixing
                                    # (arch="factorized", which compresses block_size→m too)
    per_layer: bool = True          # separate params per (layer,head) vs shared per head
    num_layers: int = 0             # required when per_layer (number of trained layers)
    tie_qk: bool = False            # share Wq = Wk
    tau_per_channel: bool = False   # gqa_envelope: learn a temperature per CHANNEL
                                    # (vs per landmark) — lets outlier channels pick
                                    # their own envelope sharpness
    index_heads: int = 4            # gqa_lightning: number of ReLU indexer heads (H_I)
    init: str = "identity"          # "identity" (truncated I → centroid warm start),
                                    # "orthogonal", or "lowpass" (gqa_factorized: keep the
                                    # r most rope-coherent channels — see rope_theta)
    rope_theta: float = 1e6         # teacher RoPE base (Qwen3: 1e6); used by init="lowpass"
                                    # to rank channels by mean-pool coherence alpha_i(B)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CompressorConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
