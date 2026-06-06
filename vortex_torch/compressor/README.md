# `vortex_torch.compressor` — trainable per-head block compressor

A small **learned** module that compresses a block of KV (the fused MLA latent)
into a compact **per-head descriptor** used to *score* blocks for sparse
selection — trained to match the real attention of a HuggingFace model.

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
    --data examples/ruler/validation.jsonl \
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
| `model.py` | `BlockCompressor` — the learned per-(layer,head) bilinear scorer. |
| `capture.py` | `MLASupervision` — frozen HF MLA teacher → on-the-fly (latent, absorbed-q) stream. |
| `objective.py` | centroids, true attention, block-mass targets, distillation loss, p-coverage/recall@N eval. |
| `train.py` | `python -m vortex_torch.compressor.train` streaming trainer. |

## Status / next step

Trains and improves selection quality over the centroid baseline (validated on
GLM-4.7-Flash: e.g. per-head recall@16 0.90→0.97 in a few epochs). The learned
`(Wk, Wq)` are exportable; wiring them into a vortex MLA flow as a cache/indexer
op (project latent → compressed-centroid → scored gemm) is the deployment step.
