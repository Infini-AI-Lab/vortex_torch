"""On-the-fly MHA/GQA supervision from a frozen HuggingFace model.

MHA twin of :class:`~vortex_torch.compressor.capture.MLASupervision`. Loads a
GQA model (Qwen3 / Qwen2 / Llama) with transformers, wraps each target
attention module so that — every forward — it captures the two tensors the
vortex sparse-MHA selector actually sees at decode:

* ``k`` [T, G, hd]   = post-rope (and post-``k_norm``) per-KV-head keys
* ``q`` [W, H, hd]   = post-rope (and post-``q_norm``) per-query-head queries
                       at the last ``W`` real positions

with ``hd = head_dim``, ``G = num_key_value_heads``, ``H = num_attention_heads``.
Query head ``h`` attends KV head ``h // (H // G)`` (HF ``repeat_kv`` order), so
``scaling · ⟨q[h], k[t, h // gs]⟩`` is the true per-head attention logit — no
absorption step is needed (that was the MLA-specific part). Nothing is written
to disk; tensors are yielded per prompt and consumed immediately.

The per-sequence dicts use the same keys as the MLA stream (``latent`` /
``q_abs`` / ``scaling``) so the trainer and objective are shared: ``latent`` is
simply 3-D ``[T, G, hd]`` here instead of 2-D, and the objective dispatches on
that.
"""
from __future__ import annotations

import functools
import importlib
from typing import Iterable, Optional

import torch

from .capture import MLASupervision


def _is_gqa_attn(module) -> bool:
    # Plain q/k/v projections (Qwen3, Qwen2, Llama, ...); excludes MLA modules
    # (kv_a_proj_with_mqa) which have no k_proj.
    return (all(hasattr(module, a) for a in ("q_proj", "k_proj", "v_proj", "scaling"))
            and not hasattr(module, "kv_a_proj_with_mqa"))


class MHASupervision:
    """Frozen GQA model that yields (per-KV-head K, per-head q) supervision on the fly."""

    # Reuse the MLA streaming machinery verbatim — these only touch shared
    # attributes (tokenizer/model/device/_store/_attn_mask/Wq_pos/targets).
    supervise = MLASupervision.supervise
    stream = MLASupervision.stream
    render = MLASupervision.render
    _install_wrappers = MLASupervision._install_wrappers
    _wrapped_forward = MLASupervision._wrapped_forward

    def __init__(
        self,
        model_name: str,
        layers: Optional[Iterable[int]] = None,
        device: str = "cuda",
        dtype="auto",
        num_query_positions: int = 1,
        gradient_checkpointing: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"          # real tokens at positions 0..T-1
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.device = device
        self.Wq_pos = int(num_query_positions)

        gqa = [(m.layer_idx, m) for m in self.model.modules()
               if _is_gqa_attn(m) and hasattr(m, "layer_idx")]
        gqa.sort(key=lambda kv: kv[0])
        if not gqa:
            raise RuntimeError(f"no GQA/MHA attention modules found in {model_name}")
        want = set(layers) if layers is not None else {i for i, _ in gqa}
        self.targets = [(i, m) for i, m in gqa if i in want]
        if not self.targets:
            raise RuntimeError(f"requested layers {sorted(want)} not among "
                               f"{[i for i, _ in gqa]}")

        a0 = self.targets[0][1]
        cfg = a0.config
        self.head_dim = int(getattr(a0, "head_dim", None) or cfg.head_dim)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
        self.group_size = self.num_q_heads // self.num_kv_heads
        # MLA-compat alias: the "latent" width seen by the scorer is head_dim.
        self.latent_dim = self.head_dim
        self.layer_ids = [i for i, _ in self.targets]
        mod = importlib.import_module(type(a0).__module__)
        self._rope = getattr(mod, "apply_rotary_pos_emb")

        self._store: dict = {}
        self._attn_mask = None
        self._install_wrappers()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _capture(self, attn, layer_idx, hidden, position_embeddings) -> None:
        b, s = hidden.shape[:2]
        hd = self.head_dim

        q = attn.q_proj(hidden).view(b, s, -1, hd)             # [b,s,H,hd]
        k = attn.k_proj(hidden).view(b, s, -1, hd)             # [b,s,G,hd]
        # Qwen3 applies per-head RMSNorm before rope; Llama/Qwen2 have none.
        if getattr(attn, "q_norm", None) is not None:
            q = attn.q_norm(q)
        if getattr(attn, "k_norm", None) is not None:
            k = attn.k_norm(k)
        q = q.transpose(1, 2)                                  # [b,H,s,hd]
        k = k.transpose(1, 2)                                  # [b,G,s,hd]
        cos, sin = position_embeddings
        q, k = self._rope(q, k, cos, sin)                      # post-rope

        scal = float(attn.scaling)
        mask = self._attn_mask                                 # [b,s] bool or None
        entries = []
        for i in range(b):
            idx = (mask[i].nonzero(as_tuple=False).flatten() if mask is not None
                   else torch.arange(s, device=k.device))      # real positions
            Ti = int(idx.numel())
            if Ti == 0:
                entries.append(None)
                continue
            w = min(self.Wq_pos, Ti)
            qpos = idx[-w:]                                    # last real positions
            entries.append({
                # same keys as the MLA stream; "latent" is 3-D [T, G, hd] here.
                "latent": k[i].index_select(1, idx)            # [G,Ti,hd]
                            .transpose(0, 1).contiguous().to(torch.float16),  # [Ti,G,hd]
                "q_abs": q[i].index_select(1, qpos)            # [H,w,hd]
                            .transpose(0, 1).contiguous().float(),            # [w,H,hd]
                "scaling": scal,
            })
        self._store[layer_idx] = entries
