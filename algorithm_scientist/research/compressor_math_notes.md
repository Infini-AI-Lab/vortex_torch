# Block-compression math for Qwen3 attention (QKNorm + RoPE)

Derived 2026-06-07, BEFORE the 16K arch sweep finished — predictions P1–P4
below are pre-registered against it. Verification: `compressor_math_verify.py`.

## Setup

Qwen3 head: `k_t = R_t·(γ_k ⊙ û_t)` with `û_t = u_t/rms(u_t)` (per-head RMSNorm
over head_dim d=128, learned per-channel weight γ_k), then RoPE rotation `R_t`:
HF rotate_half layout pairs channel `(i, i+d/2)`, rotated by angle `θ_i·t`,
`θ_i = Θ^{-2i/d}`, Θ = 1e6. Same for q. Logit `s_t = λ⟨q_T, k_t⟩`, λ = d^{-1/2}.

## 1. QKNorm ⇒ keys live on a thin spherical shell (H1)

`‖k_t‖² = ‖γ⊙û_t‖² = d·Σ_c γ_c² w_{t,c}` with `w_t = û_t²` a probability vector
(rotation R_t is orthogonal — norm unchanged). So `√d·γ_min ≤ ‖k_t‖ ≤ √d·γ_max`,
and to the extent the per-channel energy profile w_t is stable across tokens,
`‖k_t‖ ≈ const = ρ̄`. **Consequences:**
- scoring is *angle-driven*: `s_t ≈ λρ̄‖q‖·cos∠(q,k_t)`;
- the block-centroid norm `‖c_b‖ = ρ̄·‖mean_t k̂_t‖` *is* a within-block angular
  coherence measure — centroid scoring `⟨q,c_b⟩` automatically multiplies
  direction-match by coherence. (Min/max envelopes ignore the shell structure —
  one reason quest over-selects at large blocks.)

Verify: CV of `‖k_t‖` per (layer, kv head) — expect ≪ 1.

## 2. The distillation target is logsumexp, not the mean (Jensen gap)

Block attention mass `M_b ∝ Σ_{t∈b} e^{s_t} = B·e^{s̄_b}·E_t[e^{s_t-s̄_b}]`, so

    log M_b = log B + s̄_b + J_b,   J_b ≈ var_b(s)/2 + ...  (Jensen gap ≥ 0)

- centroid = `s̄_b` exactly (`⟨q, mean k⟩ = mean⟨q,k⟩` — linear);
- quest ≈ `max_b s` (channelwise upper bound), and `max ≥ LSE − log B`;
- the truth interpolates: peaked blocks → max; flat blocks → mean.

**A single linear descriptor (b_c=1) cannot represent var_b** — it is quadratic
in the block's keys, and mean-pooling destroys it. Two max-scored landmarks
(b_c=2) give a piecewise-linear surrogate that encodes spread along learned
directions.
⇒ **P1**: b2d128 > b1d128 (esp. p-coverage, where the Jensen term dominates).
⇒ nonlinear maps APPLIED TO THE POOLED DESCRIPTOR cannot recover var either
(information already destroyed): **P4**: gqa_factorized_mlp ≈ b1d128.

## 3. RoPE ⇒ mean-pooling low-passes the channels (H2)

Per pair i write `z_t = k̃_{t,i} + j·k̃_{t,i+d/2}` (complex); rotation = `e^{jθ_i t}`.
Over a block of B consecutive positions, decompose `z_t = z̄ + δ_t`:

    c_{b,i} = z̄·e^{jθ_i t̄}·D_B(θ_i) + O(σ_δ/√B),
    |D_B(θ_i)| = |sin(Bθ_i/2)/(B·sin(θ_i/2))|   (Dirichlet kernel) = α_i

At B=64, Θ=1e6, d=128: α < 0.3 for pairs 0–13, α > 0.84 for pairs ≥ 16 —
**the top ~22% of frequencies are erased from any mean-pooled descriptor**;
they contribute only O(1/√B) noise. A linear compressor should spend its
channel budget d_c on the coherent subspace.
⇒ **P2**: b1d64 ≈ b1d128 (50 coherent pairs ≈ 100 channels, and content
decorrelation shrinks the effective rank further).
⇒ **P3**: lowpass init (keep r most-coherent channels, α-weighted) ≥ identity
init early in training; converges to the same place given time.

Verify: measured per-channel `ρ_c = E|c_{b,c}|/E|k_{t,c}|` tracks α_c;
trained-Wk per-channel row energy correlates with α (H3).

## 4. Why length generalization is hard for ANY fixed linear scorer

`⟨q_T, c_b⟩ = ⟨q̃, R(t̄_b − T)·(coherent part)⟩`: the score depends on the
relative distance Δ = T − t̄_b through per-pair phases θ_i·Δ. A fixed (Wq, Wk)
pair cannot track Δ-dependent rotation; channels are position-transparent only
when θ_i·Δ_max ≪ 1. At Δ = 16K that is θ ≲ 6e-5 ⇒ pairs i ≳ 50; at Δ = 32K
even fewer. Mid-frequency pairs (14–48) are block-coherent but carry strong
relative-position phase — a compressor trained at 8–16K learns the phase
statistics of THOSE Δ ranges and partially mis-scores at 32K. This explains the
observed partial length generalization (8K-trained: edge at 16K, ≈centroid at
32K) and predicts mixed-length training as the fix (not more 8K data).

## 5. "Optimal compressor" working hypothesis

For block descriptor budget = b_c × d_c values per (block, kv head):
- spend d_c on the RoPE-coherent subspace (≈100 of 128 channels at B=64;
  effective rank lower) — d_c ∈ [64, 96] should be ≈ free;
- spend extra budget on b_c=2 max-scored landmarks (Jensen/var term), not on
  d_c > coherent rank and not on nonlinearity;
- per-KV-head Wk + per-q-head Wq (GQA asymmetry is real: 4 q heads share one
  selection);
- token mixing Wt ≈ uniform within coherent channels is near-optimal for the
  mean part; landmarks specialize it.

Pre-registered ranking prediction at equal training: b2d128 ≥ b1d128 ≈ mlp128 ≈
b1d64 > centroid; b2d64 ≈ b1d128 at HALF the bytes; lowpass-init fastest early.
