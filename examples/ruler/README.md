# `examples/ruler` — RULER quality gate for vortex sparse attention

[RULER](https://github.com/NVIDIA/RULER)-style **needle-in-a-haystack** retrieval
is the fastest way to tell whether a sparse-attention flow is *structurally
sound*: a flow that drops the page holding the needle answers wrong, so accuracy
on long-context retrieval is a sharp, cheap signal (100 prompts, a few minutes on
one GPU) — much faster than a full AIME/AMC math run.

Use it as a **gate**: a flow that scores well here (≈ ≥ 0.85, dense ≈ 1.0) has
working attention and is worth a real benchmark; a flow that tanks here is broken,
no matter what its throughput looks like.

## Files

| File | What it is |
|------|------------|
| `validation_4k.jsonl` | The RULER eval set — one `{"input": <prompt>, "outputs": [<answer>]}` per line. A run scores a hit when `outputs[0]` is a substring of the model's generation. |
| `run_ruler_mha.py` | **MHA** runner. Boots an in-process `sgl.Engine` with vortex sparsity and scores `validation_4k.jsonl`. Default model: Qwen3-4B. |
| `run_ruler_mla.py` | **MLA** runner (DeepSeek/GLM latent attention). Default model: GLM-4.7-Flash. |
| `sweep_mha.sh` | Sweeps the built-in MHA flows through RULER and prints an accuracy table — 9 flows × 2 indexer backends. |
| `sweep_mla.sh` | Same for the MLA flows — 2 flows on `cuda_mla`. |
| `run_profile_mla.py` | **MLA selection-quality profiler**: drives a short decode with `attention_backend=cuda_mla_profile` and reports per-layer/per-head **p-coverage** + **recall@N**. |
| `ruler_output.jsonl` | Last run's raw generations (overwritten each `run_ruler_mha.py` run; gitignored noise). |

The two runners share a **unified CLI** (`--model`, `--module`, `--block`,
`--topk`, `--layers-skip`, `--indexer-backend`, `--attn-backend`, `--dense`,
`--n`, `--tp`, `--kv-cache-dtype`, `--disable-radix-cache`, `--dump`, …) —
same flags, architecture-appropriate defaults. The two sweep scripts share the
same env-override names (`MODEL`, `PY`, `FLOWS`, `BACKENDS`, `BLOCK`, `TOPK`,
`LAYERS_SKIP`, `EXTRA_ARGS`, `OUT`).

All paths are anchored to this directory, so the scripts run from any cwd.

## Prerequisites

You need an interpreter where `import vortex_torch` (with sglang) works — see the
repo's `/setup-env`. The snippets below assume `vortex_v1` for MHA; **MLA on GLM
needs transformers ≥ 5** (the `vortex_glm` env). Models are read from `HF_HOME`
(this cluster: `/raid/catalyst/models/`).

```bash
conda activate vortex_v1
export HF_HOME=/raid/catalyst/models/
```

## MHA — `run_ruler_mha.py`

```bash
# default: Qwen/Qwen3-4B, gqa_block_sparse_attention, flashinfer indexer,
#          block=page=32, topk=29, layers_skip=[]
CUDA_VISIBLE_DEVICES=0 python examples/ruler/run_ruler_mha.py

# a different model / flow / budget / smaller slice / dense reference
python examples/ruler/run_ruler_mha.py --model Qwen/Qwen3-8B \
    --module lserve_sparse_attention --block 32 --topk 15 --gpu 0 --n 20
python examples/ruler/run_ruler_mha.py --dense
```

Key flags (shared with the MLA runner): `--model`, `--module`, `--block`/`--topk`
(block size == page size / selected blocks), `--layers-skip "0,1"` (dense layers,
default none), `--indexer-backend flashinfer|trtllm` (`TopK`/`Union` flows are
trtllm-only; `topK`/`approxTopK` flows run under either), `--disable-radix-cache`
(REQUIRED for `Save(...)`-based flows, e.g. `running_avg_block_sparse`),
`--dense`, `--n`, `--tp`, `--kv-cache-dtype`, `--dump` (`--help` for all).

MHA-only: `--server-url <url>` (or `RULER_SERVER_URL`) drives an already-running
sglang server over HTTP (`/generate`) instead of building an in-process engine;
the server's launch flags define the config (see `examples/misc/server_launch.sh`).

## MLA — `run_ruler_mla.py`

For latent-attention models (DeepSeek-V2 / GLM). Sparse MLA decode runs on the
`cuda_mla` backend; dense baselines can use `trtllm_mla` or `triton`.

```bash
conda activate vortex_glm           # GLM needs transformers >= 5
export HF_HOME=/raid/catalyst/models/

# sparse (default flow rope_aware_block_sparse_mla on GLM-4.7-Flash)
CUDA_VISIBLE_DEVICES=0 python examples/ruler/run_ruler_mla.py

# a different MLA flow / smaller slice / dense reference
python examples/ruler/run_ruler_mla.py --module lserve_centroid_mla --gpu 0 --n 20
python examples/ruler/run_ruler_mla.py --dense --attn-backend trtllm_mla
```

Same unified flags and defaults as the MHA runner (block=page=32, topk=29; the
known-good ~100%-RULER GLM config used `--topk 61`); MLA-specific defaults are
`--attn-backend cuda_mla` (dense baselines: `trtllm_mla`/`triton`) and
`--indexer-backend trtllm` (`--help` for all).

## Sweep all flows — `sweep_mha.sh` / `sweep_mla.sh`

```bash
examples/ruler/sweep_mha.sh                                  # 18 MHA runs (9 flows × 2 backends)
PY="conda run -n vortex_glm python" examples/ruler/sweep_mla.sh   # 2 MLA runs (GLM env)

# unified knobs (same env names in both):
BLOCK=16 TOPK=15 LAYERS_SKIP="0" examples/ruler/sweep_mha.sh
FLOWS="gqa_block_sparse_attention" BACKENDS="flashinfer" EXTRA_ARGS="--n 20" \
    examples/ruler/sweep_mha.sh
```

Both run one flow per free GPU in waves (re-detecting free GPUs each wave via
`algorithm_scientist/free_gpus.sh`); `sweep_mha.sh` auto-adds
`--disable-radix-cache` for `running_avg_block_sparse`. Each prints:

```
===== RULER MHA flow sweep — accuracy =====
flow                               backend     accuracy
block_sparse_attention             flashinfer  99.0%
...
```

Per-run logs land in `examples/ruler/sweep_results/logs/` (gitignored). Env
overrides (unified across both): `MODEL`, `PY`, `FLOWS`, `BACKENDS` (MHA:
indexer backends; MLA: sglang attention backends), `BLOCK`, `TOPK`,
`LAYERS_SKIP`, `EXTRA_ARGS`, `OUT`.

## Profiling selection quality — `run_profile_mla.py`

RULER accuracy tells you *whether* a flow works; the profiler tells you *how
much attention mass it's leaving on the table*. The `cuda_mla_profile` backend
executes exactly like `cuda_mla` but, for every decoded token, recomputes the
dense attention and accumulates, per layer and per head:

- **p-coverage** — fraction of the full softmax mass landing on the selected KV
  (`Σ_{t∈S} softmax(q·k_t)`); 1.0 = the selection caught all the mass.
- **recall@N** — of the exact top-`N` tokens by score, how many were selected
  (`|topN ∩ S| / N`); `N` is user-specified (comma list).

```bash
conda activate vortex_glm          # GLM needs transformers >= 5
export HF_HOME=/raid/catalyst/models/
CUDA_VISIBLE_DEVICES=0 python examples/ruler/run_profile_mla.py \
    --module rope_aware_block_sparse_mla --n 4 --recall-n 16,64,128
```

It prints a per-layer table (p-cov, recall@N) + overall means and dumps full
per-head detail to `--out` (default `mla_profile.json`). Profiling recomputes
dense attention in PyTorch per layer/token, so it's much slower than `cuda_mla`
and runs **eager** (cuda graph disabled) — keep `--n` small. A math-workload
twin lives at `examples/math/run_profile_mla.py`.

## Interpreting results

- **Dense ≈ 1.0, sparse ≈ 0.98–1.0** at these budgets is healthy; small run-to-run
  jitter (±1–2%) is expected on a 100-prompt set.
- **Accuracy is backend-invariant** — `flashinfer` vs `trtllm` change throughput,
  not correctness. A flow that scores differently across backends usually has a
  backend-specific bug (or you hit run-to-run noise).
- **A flow well below ~0.85** has broken attention: widen `--topk` /
  `vortex_topk_ratio`, check the indexer scoring, or confirm
  `--disable-radix-cache` for `Save`-based flows.
