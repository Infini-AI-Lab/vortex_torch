# `vortex_torch.compressor` — trainable per-head block compressor

A small **learned** module that compresses a block of KV (the fused MLA latent,
or the per-KV-head keys for MHA/GQA) into a compact **per-head descriptor**
used to *score* blocks for sparse selection — trained to match the real
attention of a HuggingFace model.

It is a low-rank bilinear generalization of the centroid block scorer. With mean
pooling the block score is

```
s[h, b] = scaling · (Wqᵀ q_h) · (Wkᵀ centroid_b)          centroid_b = mean_{t∈b} latent_t
```

so the per-block descriptor is `Wkᵀ·centroid_b` — an `r`-dim *compressed* centroid
(`r = proj_dim < latent_dim`). Setting `proj_dim = latent_dim, Wk = Wq = I`
recovers the exact centroid scorer, so training **starts at centroid and learns to
beat it**. This maps cleanly onto a future vortex op (project latent → mean →
gemm), and its quality is measured by the same **p-coverage / recall@N** metrics
as the `cuda_mla_profile` backend.

## Why "per head" for MLA

GLM/DeepSeek MLA store **one shared latent KV head** per token
(`latent_dim = kv_lora_rank + qk_rope_head_dim`, 576 for GLM). "Per head" here
means **per query head**: each query head gets its own `(Wk, Wq)` (and, by
default, per (layer, head)). Training distills each head's block ranking against
*that head's* true attention; eval reports both per-head selection and the
request-level **pooled** selection the deployed decode actually uses.

## MHA/GQA (`--attn mha`, e.g. Qwen3-4B)

For GQA models each block of K is `[block_size, head_dim]` **per KV head**, and
the compressor (`arch=gqa_factorized`) learns per-(layer, **kv_head**)
`K_comp = (Wtᵀ·K_b)·Wk ∈ R^{b_c × d_c}` (`b_c` ≤ block_size descriptors via
learned token-mixing, `d_c` ≤ head_dim channels) plus an optional per-q-head
query projection `Wq` (`--tie-qk` shares `Wq[h] = Wk[g(h)]`). Scoring is
GQA-grouped: `s(b, g) = Σ_{h∈group g} max_m ⟨Wq[h]ᵀ q_h, K_comp[g, b, m]⟩` —
query head `h` reads KV head `g(h) = h // (H/G)` (HF `repeat_kv` order).
**Centroid = (b_c=1, Wt=1/bs, Wk=I)** and **Quest ≈ the b_c=2 max-scored
corner** of the same family, so eval reports the learned scorer against BOTH
baselines (`cent`/`quest`, pooled per KV group — the selection the deployed
decode would use). Supervision (`capture_mha.MHASupervision`) wraps the frozen
HF attention and captures post-rope (and post-q/k-norm, Qwen3) `k [T, G, hd]` /
`q [W, H, hd]` on the fly — no absorption step, no disk. `b_c` ↔
`--num-landmarks/--bc`, `d_c` ↔ `--proj-dim/--dc`.

```bash
conda activate vortex_v1
# 4-GPU data-parallel (torchrun): prompts sharded per rank, compressor grads
# all-reduced per step, eval sets sharded + reduced. Train on the Jackrong
# reasoning mixture — rows are rendered with the teacher's chat template
# (<think> preserved) and PACKED into exactly --max-tokens sequences.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    -m vortex_torch.compressor.train \
    --model Qwen/Qwen3-4B --attn mha \
    --hf-dataset "Jackrong/GLM-5.1-Reasoning-1M-Cleaned,Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned#General-Distillation" \
    --num-prompts 512 --max-minutes 25 --max-tokens 8192 \
    --bc 1 --dc 128 --block-size 64 --budget-blocks 15 \
    --eval-jsonl ruler8k:examples/ruler/validation_8k.jsonl:8192:8 \
    --eval-jsonl ruler16k:examples/ruler/validation_16k.jsonl:16384:8 \
    --eval-jsonl ruler32k:examples/ruler/validation_32k.jsonl:32768:8 \
    --out result/compressor/qwen3_4b_mha_64to1.pt
```

Single-GPU / RULER-jsonl training works the same way without torchrun
(`--data examples/ruler/validation_8k.jsonl --eval-holdout 16`), but RULER's
haystack is one document source (Paul Graham essays) — fine as an *eval*
gate, too narrow as a *training* distribution.

## On-the-fly supervision (no disk)

Supervision is computed **on the fly** from a frozen HF MLA model — no traces are
written to disk. Each step runs one HF forward; a forward wrapper
(`capture.MLASupervision`) reconstructs, for the target layers:

- `latent[T, d] = [ kv_a_layernorm(k_c) | rope(k_pe) ]`  (the shared latent)
- `q_abs[H, d] = [ q_nope · W_UK | rope(q_pe) ]`        (the absorbed query)

By construction `⟨q_abs[h], latent_t⟩` equals the true per-head attention logit,
so the trainer builds the exact per-block attention-mass target and distills the
compressor's block distribution against it (soft cross-entropy / KL). Only the
tiny compressor weights are saved.

The teacher is kept in its **native precision** (`--dtype auto`, e.g. bf16, no
fp32 upcast), runs under `no_grad`, and has **gradient checkpointing** enabled by
default (`--no-grad-checkpointing` to turn off) to bound activation VRAM on long
contexts.

## Train

```bash
conda activate vortex_glm                 # GLM needs transformers >= 5
export HF_HOME=/raid/catalyst/models/
CUDA_VISIBLE_DEVICES=0 python -m vortex_torch.compressor.train \
    --model zai-org/GLM-4.7-Flash \
    --data examples/ruler/validation_4k.jsonl \
    --num-prompts 32 --epochs 3 \
    --proj-dim 128 --block-size 32 --budget-blocks 64 \
    --recall-n 16,64,128 \
    --out result/compressor/glm.pt
```

Per-epoch it prints distillation loss and selection quality (per-head + pooled
p-coverage and recall@N). Useful flags: `--layers 0,1,23,46` (subset),
`--proj-dim` (descriptor rank / compression), `--num-query-positions N`
(supervise the last N decode positions per prompt), `--tie-qk`, `--lr`.

Output: `result/compressor/glm.pt` (`{state_dict, config, layer_ids}`) +
`glm.pt.json` (the `CompressorConfig`).

## API

```python
from vortex_torch.compressor import CompressorConfig, BlockCompressor, MLASupervision
from vortex_torch.compressor import objective as O

sup  = MLASupervision("zai-org/GLM-4.7-Flash", layers=[0, 23, 46])
cfg  = CompressorConfig(latent_dim=sup.latent_dim, num_q_heads=sup.num_q_heads,
                        proj_dim=128, per_layer=True, num_layers=len(sup.layer_ids))
comp = BlockCompressor(cfg)

for s in sup.stream(prompts):                 # one dict per prompt, on the fly
    for lid, d in s.items():
        A    = O.true_attention(d["q_abs"][-1], d["latent"].float(), d["scaling"])
        tgt  = O.block_mass_targets(A, block_size=32)
        cent = O.block_centroids(d["latent"].float(), 32)
        logits = comp.block_logits(d["q_abs"][-1], cent, layer_pos, d["scaling"])
        loss = O.distill_loss(logits, tgt)    # backward updates only `comp`
```

## Files

| File | Role |
|------|------|
| `config.py` | `CompressorConfig` (geometry + hyper-params). |
| `model.py` | `BlockCompressor` — the learned scorers (`ARCH_REGISTRY`: bilinear / landmark / factorized / mlp / **gqa_factorized**). |
| `capture.py` | `MLASupervision` — frozen HF MLA teacher → on-the-fly (latent, absorbed-q) stream. |
| `capture_mha.py` | `MHASupervision` — frozen HF GQA teacher → on-the-fly (per-KV-head k, per-head q) stream. |
| `objective.py` | centroids/quest baselines, true attention (MLA + GQA), block-mass targets, distillation loss, p-coverage/recall@N eval (all-head or per-group pooling). |
| `train.py` | `python -m vortex_torch.compressor.train` streaming trainer (`--attn auto|mla|mha`). |

## Status / next step

Trains and improves selection quality over the centroid baseline (validated on
GLM-4.7-Flash: e.g. per-head recall@16 0.90→0.97 in a few epochs). The learned
`(Wk, Wq)` are exportable; wiring them into a vortex MLA flow as a cache/indexer
op (project latent → compressed-centroid → scored gemm) is the deployment step.
