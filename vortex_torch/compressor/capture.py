"""On-the-fly MLA supervision from a frozen HuggingFace model.

Loads an MLA model (GLM-4.7-Flash / DeepSeek-V2/V3) with transformers, wraps each
target attention module so that — every forward — it reconstructs the two tensors
the vortex sparse-MLA selector actually uses:

* ``latent``  [T, d]      = [ kv_a_layernorm(k_c) | rope(k_pe) ]   (shared latent KV)
* ``q_abs``   [Wq, H, d]  = [ q_nope · W_UK | rope(q_pe) ]         (absorbed query)

with ``d = kv_lora_rank + qk_rope_head_dim``. By construction
``⟨q_abs[h], latent_t⟩`` equals the true per-head attention logit, so the trainer
can build exact distillation targets from these alone. Nothing is written to
disk; the tensors are yielded per prompt and consumed immediately.
"""
from __future__ import annotations

import functools
import importlib
from typing import Iterable, Iterator, Optional

import torch


def _is_mla_attn(module) -> bool:
    return all(hasattr(module, a) for a in
               ("kv_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm",
                "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "scaling"))


class MLASupervision:
    """Frozen MLA model that yields (latent, absorbed-query) supervision on the fly."""

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
        # Keep the teacher's weights in their native precision (dtype="auto" → the
        # checkpoint's saved dtype, e.g. bf16 for GLM) — no upcast to fp32.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        # Gradient checkpointing trades compute for activation VRAM during the big
        # teacher forward (a safety net for very long contexts; the forward also
        # runs under no_grad, which already frees per-layer activations).
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.device = device
        self.Wq_pos = int(num_query_positions)

        # Discover MLA attention modules in layer order.
        mla = [(m.layer_idx, m) for m in self.model.modules()
               if _is_mla_attn(m) and hasattr(m, "layer_idx")]
        mla.sort(key=lambda kv: kv[0])
        if not mla:
            raise RuntimeError(f"no MLA attention modules found in {model_name}")
        want = set(layers) if layers is not None else {i for i, _ in mla}
        self.targets = [(i, m) for i, m in mla if i in want]
        if not self.targets:
            raise RuntimeError(f"requested layers {sorted(want)} not among "
                               f"{[i for i, _ in mla]}")

        a0 = self.targets[0][1]
        self.kv_lora_rank = int(a0.kv_lora_rank)
        self.qk_rope_head_dim = int(a0.qk_rope_head_dim)
        self.qk_nope_head_dim = int(a0.qk_nope_head_dim)
        self.v_head_dim = int(a0.v_head_dim)
        self.latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.num_q_heads = int(a0.config.num_attention_heads)
        self.layer_ids = [i for i, _ in self.targets]            # captured layer indices
        # rope fns from the model's own modeling module
        mod = importlib.import_module(type(a0).__module__)
        self._rope = getattr(mod, "apply_rotary_pos_emb")
        self._rope_il = getattr(mod, "apply_rotary_pos_emb_interleave", None)

        self._store: dict = {}
        self._install_wrappers()

    # ------------------------------------------------------------------ #
    def _install_wrappers(self) -> None:
        for layer_idx, attn in self.targets:
            attn._orig_forward = attn.forward
            attn.forward = functools.partial(self._wrapped_forward, attn, layer_idx)

    @torch.no_grad()
    def _wrapped_forward(self, attn, layer_idx, *args, **kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is None and len(args) >= 2:
            pos_emb = args[1]
        self._capture(attn, layer_idx, hidden, pos_emb)
        return attn._orig_forward(*args, **kwargs)

    @torch.no_grad()
    def _capture(self, attn, layer_idx, hidden, position_embeddings) -> None:
        b, s = hidden.shape[:2]
        H = attn.num_heads
        nope, rope, lora = attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.kv_lora_rank
        qk_head_dim = attn.qk_head_dim

        if getattr(attn, "q_lora_rank", None) is None:
            q = attn.q_proj(hidden)
        else:
            q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden)))
        q = q.view(b, s, -1, qk_head_dim).transpose(1, 2)          # [b,H,s,qk_head_dim]
        q_pass, q_rot = torch.split(q, [nope, rope], dim=-1)        # [b,H,s,nope],[b,H,s,rope]

        ckv = attn.kv_a_proj_with_mqa(hidden)
        k_c, k_rot = torch.split(ckv, [lora, rope], dim=-1)         # [b,s,lora],[b,s,rope]
        kv_c = attn.kv_a_layernorm(k_c)                             # [b,s,lora]
        k_rot = k_rot.view(b, 1, s, rope)

        cos, sin = position_embeddings
        if getattr(attn.config, "rope_interleave", False) and self._rope_il is not None:
            q_rot, k_rot = self._rope_il(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = self._rope(q_rot, k_rot, cos, sin)       # post-rope

        # latent = [ kv_c | k_pe ]  (k_pe shared across heads)
        latent = torch.cat([kv_c, k_rot[:, 0]], dim=-1)            # [b,s,d]

        # absorbed query nope part: q_nope · W_UK  (W_UK = kv_b_proj k-rows)
        W = attn.kv_b_proj.weight.view(H, nope + attn.v_head_dim, lora)
        W_UK = W[:, :nope, :]                                       # [H,nope,lora]
        q_nope_abs = torch.einsum("bhsd,hdc->bhsc", q_pass.to(W_UK.dtype), W_UK)  # [b,H,s,lora]
        q_abs = torch.cat([q_nope_abs, q_rot.to(W_UK.dtype)], dim=-1)             # [b,H,s,d]

        w = min(self.Wq_pos, s)
        self._store[layer_idx] = {
            "latent": latent[0].to(torch.float16),                  # [s,d]
            "q_abs": q_abs[0, :, -w:, :].transpose(0, 1).contiguous().float(),  # [w,H,d]
            "scaling": float(attn.scaling),
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def supervise(self, prompt: str, max_tokens: int = 8192) -> dict:
        """Run one forward over ``prompt`` and return {layer_idx: {latent,q_abs,scaling}}."""
        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=max_tokens).input_ids.to(self.device)
        self._store = {}
        self.model(ids, use_cache=False)
        out = self._store
        self._store = {}
        return out

    def render(self, text: str, thinking: bool = False) -> str:
        msg = [{"role": "user", "content": text}]
        try:
            return self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)

    def stream(self, prompts: Iterable[str], render: bool = True,
               max_tokens: int = 8192) -> Iterator[dict]:
        for p in prompts:
            yield self.supervise(self.render(p) if render else p, max_tokens=max_tokens)
