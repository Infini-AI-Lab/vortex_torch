"""Trainable per-head block compressor — flexible, pluggable architectures.

A compressor maps a block's KV (the fused MLA latent) to a per-head descriptor
and scores it against the query. Architectures are registered in
:data:`ARCH_REGISTRY`; add a new one by subclassing :class:`BlockScorer` and
decorating it with ``@register_arch("name")`` — no other code changes. All
scorers share one interface::

    block_logits(q[H, d], latent[T, d], block_size, layer_id, scaling) -> [H, n_blocks]

so the trainer/eval are arch-agnostic (they hand over the raw latent; the scorer
does its own block compression). The default ``bilinear`` arch is a strict
low-rank generalization of the centroid scorer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CompressorConfig

ARCH_REGISTRY: dict = {}


def register_arch(name: str):
    def deco(cls):
        ARCH_REGISTRY[name] = cls
        return cls
    return deco


def _sub_centroids(latent: torch.Tensor, block_size: int, m: int) -> torch.Tensor:
    """``latent`` [T, d] → [n_blocks, m, d]: m sub-block means per block."""
    T, d = latent.shape
    nb = (T + block_size - 1) // block_size
    pad = nb * block_size - T
    Lp = F.pad(latent, (0, 0, 0, pad)).view(nb, block_size, d)         # [nb, bs, d]
    sub = block_size // m
    # pad block_size up to a multiple of m if needed
    if block_size % m:
        Lp = F.pad(Lp, (0, 0, 0, m - block_size % m))
        sub = Lp.shape[1] // m
    return Lp.view(nb, m, sub, d).mean(dim=2)                          # [nb, m, d]


def _block_centroids(latent: torch.Tensor, block_size: int) -> torch.Tensor:
    return _sub_centroids(latent, block_size, 1).squeeze(1)            # [nb, d]


def rope_coherence(d: int, block_size: int, rope_theta: float) -> torch.Tensor:
    r"""Per-channel mean-pool coherence under RoPE, ``alpha[d]``.

    RoPE rotates channel pair ``(i, i + d/2)`` by ``theta_i·t`` with
    ``theta_i = rope_theta^{-2i/d}``. Mean-pooling a block of ``B`` positions
    attenuates that pair's coherent component by the Dirichlet factor

        alpha_i = |sin(B·theta_i/2) / (B·sin(theta_i/2))|

    (≈0 for high-frequency pairs at B=64, ≈1 for low-frequency). Channels are
    laid out HF rotate_half-style: alpha[i] = alpha[i + d/2]."""
    half = d // 2
    i = torch.arange(half, dtype=torch.float64)
    theta = rope_theta ** (-2.0 * i / d)
    B = float(block_size)
    a = torch.abs(torch.sin(B * theta / 2) / (B * torch.sin(theta / 2)))
    a = torch.where(theta < 1e-9, torch.ones_like(a), a)
    return torch.cat([a, a]).float()                                   # [d]


# --------------------------------------------------------------------------- #
class BlockScorer(nn.Module):
    """Base class for block-scoring architectures."""

    def __init__(self, cfg: CompressorConfig):
        super().__init__()
        self.cfg = cfg

    def _proj_init(self, lead, d, r) -> torch.Tensor:
        # "quest" init also needs truncated-identity projections (Wk=Wq=I is
        # required to reproduce Quest exactly; only the envelope temperatures
        # differ from the centroid warm start).
        if self.cfg.init in ("identity", "quest"):
            base = torch.zeros(d, r)
            k = min(d, r)
            base[:k, :k] = torch.eye(k)
            w = base.expand(*lead, d, r).clone() + 0.01 * torch.randn(*lead, d, r)
            return w
        w = torch.randn(*lead, d, r)
        return torch.nn.init.orthogonal_(w.reshape(-1, r)).reshape(*lead, d, r)

    def _lowpass_proj(self, lead, d, r):
        """Channel-selecting init that keeps the r most RoPE-COHERENT channels
        (low-frequency pairs) instead of the first r. Truncating to the first r
        keeps high-frequency, position-unstable channels and breaks at small r
        (see report §RoPE); ranking by mean-pool coherence alpha_i fixes it."""
        alpha = rope_coherence(d, self.cfg.block_size, self.cfg.rope_theta)
        order = torch.argsort(alpha, descending=True)[:r]
        base = torch.zeros(d, r)
        base[order, torch.arange(r)] = 1.0
        return base.expand(*lead, d, r).clone() + 0.01 * torch.randn(*lead, d, r)

    def _slice(self, W, layer_id):
        return W[layer_id] if self.cfg.per_layer else W

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        raise NotImplementedError


@register_arch("bilinear")
class BilinearScorer(BlockScorer):
    r"""s[h,b] = scaling · (Wqᵀ q_h) · (Wkᵀ centroid_b). Centroid = mean pool.
    ``Wk=Wq=I, r=d`` recovers the exact centroid scorer (warm start)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.Wk = nn.Parameter(self._proj_init(lead + (H,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        cent = _block_centroids(latent, block_size)                    # [B, d]
        Wk = self._slice(self.Wk, layer_id); Wq = self._slice(self.Wq, layer_id)
        g = torch.einsum("bd,hdr->hbr", cent.to(Wk.dtype), Wk)         # [H, B, r]
        u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)             # [H, r]
        return scaling * torch.einsum("hr,hbr->hb", u, g)              # [H, B]


@register_arch("landmark")
class LandmarkScorer(BlockScorer):
    r"""m sub-block descriptors per block; block score = max over landmarks.
    Learned generalization of LServe sub-block centroids."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.m = max(1, cfg.num_landmarks)
        self.Wk = nn.Parameter(self._proj_init(lead + (H,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        sub = _sub_centroids(latent, block_size, self.m)               # [B, m, d]
        Wk = self._slice(self.Wk, layer_id); Wq = self._slice(self.Wq, layer_id)
        g = torch.einsum("bmd,hdr->hbmr", sub.to(Wk.dtype), Wk)        # [H, B, m, r]
        u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)            # [H, r]
        s = scaling * torch.einsum("hr,hbmr->hbm", u, g)              # [H, B, m]
        return s.amax(dim=-1)                                          # [H, B]


@register_arch("factorized")
class FactorizedScorer(BlockScorer):
    r"""Learn BOTH compressions: a token-mixing ``Wt`` [block_size, m] collapses
    the block's tokens (block_size → m descriptors, instead of fixed mean pooling)
    and ``Wk`` [d, r] compresses channels. Per block, descriptor

        g[m, r] = (Wtᵀ · L_block) · Wk          L_block ∈ R^{block_size × d}

    and block score = max_m scaling·(Wqᵀ q)·g[m]. ``Wt = 1/block_size, m=1, Wk=I``
    recovers the centroid scorer; ``m>1`` learns multiple within-block landmarks
    with learned (not uniform) token weights.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        self.m = max(1, cfg.num_landmarks)
        self.block_size = cfg.block_size
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        # token-mixing: init to uniform mean (warm start = centroid for every landmark)
        Wt = torch.full(lead + (H, cfg.block_size, self.m), 1.0 / cfg.block_size)
        Wt = Wt + 0.01 * torch.randn_like(Wt)
        self.Wt = nn.Parameter(Wt)
        self.Wk = nn.Parameter(self._proj_init(lead + (H,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        T, d = latent.shape
        nb = (T + block_size - 1) // block_size
        pad = nb * block_size - T
        Lb = F.pad(latent, (0, 0, 0, pad)).view(nb, block_size, d)      # [nb, bs, d]
        Wt = self._slice(self.Wt, layer_id); Wk = self._slice(self.Wk, layer_id)
        Wq = self._slice(self.Wq, layer_id)
        tok = torch.einsum("nsd,hsm->hnmd", Lb.to(Wt.dtype), Wt)        # [H, nb, m, d]
        g = torch.einsum("hnmd,hdr->hnmr", tok, Wk)                     # [H, nb, m, r]
        u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)             # [H, r]
        s = scaling * torch.einsum("hr,hnmr->hnm", u, g)              # [H, nb, m]
        return s.amax(dim=-1)                                          # [H, nb]


@register_arch("gqa_factorized")
class GQAFactorizedScorer(BlockScorer):
    r"""MHA/GQA factorized block compressor — **per (layer, kv_head)** cache
    weights, per (layer, q_head) query weights, GQA-group scoring.

    Per block ``b`` of KV head ``g`` with keys ``K_b ∈ R^{bs×hd}``:

        K_comp[g, b] = (Wt[g]ᵀ · K_b) · Wk[g]            ∈ R^{m×r}   (= [b_c, d_c])
        u[h]        = Wq[h]ᵀ q_h                          ∈ R^{r}
        s[h, b]     = scaling · max_m ⟨u_h, K_comp[g(h), b, m]⟩

    where query head ``h`` reads its GQA group ``g(h) = h // (H/G)``. Identity
    init (``Wt = 1/bs`` uniform, ``Wk = Wq = truncated I``) recovers the
    per-group centroid scorer; ``m=2`` spans Quest-like two-descriptor
    selection with *learned* (not min/max) descriptors. ``tie_qk`` shares
    ``Wq[h] = Wk[g(h)]``.

    ``latent`` is 3-D ``[T, G, hd]`` (per-KV-head keys); returns per-query-head
    logits ``[H, n_blocks]`` — the trainer distills these per head, deployment
    pools per group.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        H, G = cfg.num_q_heads, cfg.num_kv_heads
        assert G > 0, "gqa_factorized needs cfg.num_kv_heads > 0"
        assert H % G == 0, f"num_q_heads {H} not divisible by num_kv_heads {G}"
        d, r = cfg.latent_dim, cfg.proj_dim
        self.G, self.gs = G, H // G
        self.m = max(1, cfg.num_landmarks)
        self.block_size = cfg.block_size
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        # token-mixing per kv head: init uniform mean (centroid warm start)
        Wt = torch.full(lead + (G, cfg.block_size, self.m), 1.0 / cfg.block_size)
        self.Wt = nn.Parameter(Wt + 0.01 * torch.randn_like(Wt))
        if cfg.init == "lowpass":
            # Math-derived warm start: project onto the r most rope-coherent
            # channels (alpha-weighted), q and k matched so dot products align.
            alpha = rope_coherence(d, cfg.block_size, cfg.rope_theta)
            order = torch.argsort(alpha, descending=True)[:r]
            base = torch.zeros(d, r)
            base[order, torch.arange(r)] = alpha[order]
            Wk = base.expand(*(lead + (G,)), d, r).clone() + 0.01 * torch.randn(*lead, G, d, r)
            self.Wk = nn.Parameter(Wk)
            if cfg.tie_qk:
                self.Wq = None
            else:
                qbase = torch.zeros(d, r)
                qbase[order, torch.arange(r)] = 1.0      # q side: select, don't attenuate
                Wq = qbase.expand(*(lead + (H,)), d, r).clone() + 0.01 * torch.randn(*lead, H, d, r)
                self.Wq = nn.Parameter(Wq)
        else:
            self.Wk = nn.Parameter(self._proj_init(lead + (G,), d, r))
            self.Wq = None if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        # latent [T, G, hd]; q [H, hd]
        T, G, d = latent.shape
        nb = (T + block_size - 1) // block_size
        pad = nb * block_size - T
        Kb = F.pad(latent, (0, 0, 0, 0, 0, pad)).view(nb, block_size, G, d)
        Wt = self._slice(self.Wt, layer_id); Wk = self._slice(self.Wk, layer_id)
        tok = torch.einsum("nsgd,gsm->gnmd", Kb.to(Wt.dtype), Wt)      # [G, nb, m, d]
        Kc = torch.einsum("gnmd,gdr->gnmr", tok, Wk)                   # [G, nb, m, r]
        if self.Wq is None:                                            # tie: Wq[h] = Wk[g(h)]
            u = torch.einsum("gud,gdr->gur", q.view(G, self.gs, d).to(Wk.dtype), Wk)
        else:
            Wq = self._slice(self.Wq, layer_id)
            u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq).view(G, self.gs, -1)
        s = scaling * torch.einsum("gur,gnmr->gunm", u, Kc)            # [G, gs, nb, m]
        return s.amax(dim=-1).reshape(G * self.gs, nb)                 # [H, nb]


@register_arch("gqa_envelope")
class GQAEnvelopeScorer(BlockScorer):
    r"""MHA/GQA compressor whose hypothesis class **contains both the centroid
    and Quest** as exact points.

    Two structural ingredients let it express Quest (which neither the linear
    factorized scorer nor a post-pool MLP can):

    1. **Soft-envelope pooling** (nonlinear, per channel). Each of ``m``
       landmarks has a learned scalar temperature ``tau[g,m]``; landmark $m$ of
       KV head $g$ pools the block's keys per channel by a token-softmax at that
       temperature,
       \[ p_{m}[c] = \sum_{t\in b}\mathrm{softmax}_t(\tau_m\,K_{b}[t,c])\,K_b[t,c], \]
       so $\tau\!\to\!+\infty$ gives the channelwise **max**, $\tau\!\to\!-\infty$
       the **min**, and $\tau\!=\!0$ the **mean** (centroid). The descriptor is
       $g_m=W_k^{\top}p_m\in\mathbb{R}^r$.
    2. **Per-channel (inside-sum) max over landmarks** at scoring time,
       \[ s(b,g)=\sum_{h\in\mathrm{group}(g)}\sum_{r}\max_{m}\,u_h[r]\,g_m[r],
          \qquad u_h=W_q[h]^{\top}q_h, \]
       matching Quest's $\sum_d \max(q_d k^{\max}_d, q_d k^{\min}_d)$ rather than
       the linear scorer's max-outside-sum.

    With $m{=}1,\tau{=}0,W_k{=}W_q{=}I$ this is the centroid scorer; with
    $m{=}2,\tau{=}(+\infty,-\infty),W_k{=}W_q{=}I,r{=}d$ it is **exactly Quest**.
    Default init is the centroid warm start ($\tau{=}0$); ``init="quest"`` warm
    starts at Quest instead.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        H, G = cfg.num_q_heads, cfg.num_kv_heads
        assert G > 0 and H % G == 0
        d, r = cfg.latent_dim, cfg.proj_dim
        self.G, self.gs = G, H // G
        self.m = max(1, cfg.num_landmarks)
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        if cfg.init == "quest":
            # Warm start AT Quest: landmark 0 -> channelwise max, 1 -> min, ...
            # tau large enough that the per-channel token-softmax is near-hard
            # (post-QKNorm keys are O(1); ~16 gives a Quest-like envelope that is
            # still trainable — much larger saturates the softmax gradient).
            t0 = torch.tensor([16.0 * (1 if j % 2 == 0 else -1) for j in range(self.m)])
            tau = t0.view(*([1] * len(lead)), 1, self.m).expand(lead + (G, self.m)).clone()
        else:
            tau = torch.zeros(lead + (G, self.m))             # tau=0 -> mean -> centroid
        self.per_channel_tau = bool(getattr(cfg, "tau_per_channel", False))
        if self.per_channel_tau:                              # broadcast tau over channels
            tau = tau.unsqueeze(-1).expand(*tau.shape, d).clone()   # [..,G,m,d]
        self.tau = nn.Parameter(tau + 0.01 * torch.randn_like(tau))
        # init="lowpass": keep the r most RoPE-coherent channels (needed for
        # small r — naive truncation keeps high-freq junk and breaks; see d32).
        proj = self._lowpass_proj if cfg.init == "lowpass" else self._proj_init
        self.Wk = nn.Parameter(proj(lead + (G,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(proj(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        T, G, d = latent.shape
        nb = (T + block_size - 1) // block_size
        pad = nb * block_size - T
        Kb = torch.nn.functional.pad(latent, (0, 0, 0, 0, 0, pad)).view(nb, block_size, G, d)
        tau = self._slice(self.tau, layer_id)                 # [G,m] or [G,m,d]
        Wk = self._slice(self.Wk, layer_id); Wq = self._slice(self.Wq, layer_id)
        # per-channel soft pool: weights over the bs tokens, per (block, kv-head,
        # landmark, channel). Mask padded tail tokens out of the softmax.
        if self.per_channel_tau:
            logits = torch.einsum("nsgd,gmd->gnmsd", Kb.to(tau.dtype), tau)  # [G,nb,m,bs,d]
        else:
            logits = torch.einsum("nsgd,gm->gnmsd", Kb.to(tau.dtype), tau)   # [G,nb,m,bs,d]
        if pad:
            mask = torch.zeros(block_size, device=Kb.device)
            mask[block_size - pad:] = float("-inf")           # only the LAST block is partial
            logits[:, -1] = logits[:, -1] + mask.view(1, 1, block_size, 1)
        w = logits.softmax(dim=3)                             # over tokens s
        pooled = torch.einsum("gnmsd,nsgd->gnmd", w, Kb.to(w.dtype))     # [G,nb,m,d]
        g = torch.einsum("gnmd,gdr->gnmr", pooled, Wk)        # [G,nb,m,r]
        if self.Wq is None:
            u = torch.einsum("gud,gdr->gur", q.view(G, self.gs, d).to(Wk.dtype), Wk)
        else:
            u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq).view(G, self.gs, -1)
        # per-channel (inside-sum) max over landmarks: max_m then sum over r.
        prod = torch.einsum("gur,gnmr->gunmr", u, g)          # [G,gs,nb,m,r]
        s = scaling * prod.amax(dim=3).sum(dim=-1)            # max over m, sum over r
        return s.reshape(G * self.gs, nb)                     # [H, nb]


@register_arch("gqa_lightning")
class GQALightningScorer(BlockScorer):
    r"""DeepSeek-V3.2 lightning-indexer-style block scorer for GQA.

    The descriptor is a SINGLE compressed block key ``c_b = W_k^T·mean(K_b) ∈ R^r``
    (centroid-cheap, r values/block — like the centroid, NOT m landmarks). The
    expressivity lives in the SCORING, a multi-head ReLU (DSA Eq. 1):

        s(b,h) = Σ_{j=1}^{H_I} softplus(a_{h,j}) · ReLU( (W^I_{h,j} q_h) · c_b ),

    group-summed over the GQA group. ReLU is the cheap nonlinearity DeepSeek
    chose for FP8 throughput; the per-head learned gates a_{h,j} weight the
    indexer heads. Init small (near-linear). Tests whether a cheap-descriptor +
    nonlinear-score beats the envelope's expensive-descriptor + max-score at
    iso descriptor cost."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, G = cfg.num_q_heads, cfg.num_kv_heads
        assert G > 0 and H % G == 0
        d, r = cfg.latent_dim, cfg.proj_dim
        self.G, self.gs = G, H // G
        self.HI = max(1, int(getattr(cfg, "index_heads", 4)))
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.Wk = nn.Parameter(self._proj_init(lead + (G,), d, r))           # block key compress
        # per (q-head, indexer-head) query projection d->r, and a gate per head.
        self.WqI = nn.Parameter(self._proj_init(lead + (H, self.HI), d, r))   # [..,H,HI,d,r]
        self.gate = nn.Parameter(torch.zeros(lead + (H, self.HI)))            # softplus(0)=ln2

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        T, G, d = latent.shape
        nb = (T + block_size - 1) // block_size
        pad = nb * block_size - T
        Kb = F.pad(latent, (0, 0, 0, 0, 0, pad)).view(nb, block_size, G, d)
        cnt = torch.full((nb,), float(block_size), device=latent.device)
        if pad:
            cnt[-1] = block_size - pad
        cent = Kb.sum(1) / cnt.view(nb, 1, 1)                  # [nb, G, d] per-kv-head mean
        Wk = self._slice(self.Wk, layer_id)                    # [G,d,r]
        WqI = self._slice(self.WqI, layer_id)                  # [H,HI,d,r]
        gate = F.softplus(self._slice(self.gate, layer_id))    # [H,HI]
        c = torch.einsum("ngd,gdr->ngr", cent.to(Wk.dtype), Wk)            # [nb,G,r]
        u = torch.einsum("hd,hjdr->hjr", q.to(WqI.dtype), WqI)             # [H,HI,r]
        u = u.view(G, self.gs, self.HI, -1)                                 # [G,gs,HI,r]
        cg = c.permute(1, 0, 2)                                             # [G,nb,r]
        act = F.relu(torch.einsum("gujr,gnr->gujn", u, cg))                 # [G,gs,HI,nb]
        gj = gate.view(G, self.gs, self.HI, 1)
        s = scaling * (gj * act).sum(dim=2)                                 # [G,gs,nb]
        return s.reshape(G * self.gs, nb)


@register_arch("gqa_factorized_mlp")
class GQAFactorizedMLPScorer(GQAFactorizedScorer):
    r"""GQA factorized scorer + a zero-init GELU residual on BOTH sides:

        g[g, b] = Wk[g]ᵀ c̃_b + K2[g]ᵀ gelu(K1[g]ᵀ c̃_b)
        u[h]    = Wq[h]ᵀ q_h + Q2[h]ᵀ gelu(Q1[h]ᵀ q_h)

    (c̃_b = the Wt-token-mixed descriptor). Starts exactly at the linear
    scorer (residual ≈ 0 at init) — a controlled test of whether *nonlinear*
    capacity buys anything over the optimal linear compressor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        assert not cfg.tie_qk, "gqa_factorized_mlp: tie_qk unsupported"
        H, G = cfg.num_q_heads, cfg.num_kv_heads
        d, r = cfg.latent_dim, cfg.proj_dim
        hid = cfg.hidden_dim or r
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.K1 = nn.Parameter(0.02 * torch.randn(*lead, G, d, hid))
        self.K2 = nn.Parameter(0.001 * torch.randn(*lead, G, hid, r))
        self.Q1 = nn.Parameter(0.02 * torch.randn(*lead, H, d, hid))
        self.Q2 = nn.Parameter(0.001 * torch.randn(*lead, H, hid, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        T, G, d = latent.shape
        nb = (T + block_size - 1) // block_size
        pad = nb * block_size - T
        Kb = F.pad(latent, (0, 0, 0, 0, 0, pad)).view(nb, block_size, G, d)
        Wt = self._slice(self.Wt, layer_id); Wk = self._slice(self.Wk, layer_id)
        Wq = self._slice(self.Wq, layer_id)
        K1 = self._slice(self.K1, layer_id); K2 = self._slice(self.K2, layer_id)
        Q1 = self._slice(self.Q1, layer_id); Q2 = self._slice(self.Q2, layer_id)
        tok = torch.einsum("nsgd,gsm->gnmd", Kb.to(Wt.dtype), Wt)      # [G, nb, m, d]
        g_lin = torch.einsum("gnmd,gdr->gnmr", tok, Wk)
        g_h = F.gelu(torch.einsum("gnmd,gde->gnme", tok, K1))
        Kc = g_lin + torch.einsum("gnme,ger->gnmr", g_h, K2)           # [G, nb, m, r]
        u_lin = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)
        u_h = F.gelu(torch.einsum("hd,hde->he", q.to(Q1.dtype), Q1))
        u = (u_lin + torch.einsum("he,her->hr", u_h, Q2)).view(G, self.gs, -1)
        s = scaling * torch.einsum("gur,gnmr->gunm", u, Kc)            # [G, gs, nb, m]
        return s.amax(dim=-1).reshape(G * self.gs, nb)                 # [H, nb]


@register_arch("mlp")
class MLPScorer(BlockScorer):
    r"""Nonlinear per-head heads: score = MLPq(q)·MLPk(centroid). Mean pool."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        hid = cfg.hidden_dim or max(2 * r, d // 2)
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        # per-(layer,head) two-layer projections (d→hid→r) for q and k
        self.Wk1 = nn.Parameter(self._proj_init(lead + (H,), d, hid))
        self.Wk2 = nn.Parameter(0.02 * torch.randn(*(lead + (H, hid, r))))
        self.Wq1 = nn.Parameter(self._proj_init(lead + (H,), d, hid))
        self.Wq2 = nn.Parameter(0.02 * torch.randn(*(lead + (H, hid, r))))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        cent = _block_centroids(latent, block_size)                        # [B, d]
        Wk1 = self._slice(self.Wk1, layer_id); Wk2 = self._slice(self.Wk2, layer_id)
        Wq1 = self._slice(self.Wq1, layer_id); Wq2 = self._slice(self.Wq2, layer_id)
        gh = F.gelu(torch.einsum("bd,hde->hbe", cent.to(Wk1.dtype), Wk1))  # [H, B, hid]
        g = torch.einsum("hbe,her->hbr", gh, Wk2)                          # [H, B, r]
        uh = F.gelu(torch.einsum("hd,hde->he", q.to(Wq1.dtype), Wq1))      # [H, hid]
        u = torch.einsum("he,her->hr", uh, Wq2)                            # [H, r]
        return scaling * torch.einsum("hr,hbr->hb", u, g)                  # [H, B]


class BlockCompressor(nn.Module):
    """Wraps the configured :class:`BlockScorer` (``cfg.arch``)."""

    def __init__(self, cfg: CompressorConfig):
        super().__init__()
        if cfg.arch not in ARCH_REGISTRY:
            raise ValueError(f"unknown arch {cfg.arch!r}; have {sorted(ARCH_REGISTRY)}")
        self.cfg = cfg
        self.scorer = ARCH_REGISTRY[cfg.arch](cfg)

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        return self.scorer.block_logits(q, latent, block_size, layer_id, scaling)
