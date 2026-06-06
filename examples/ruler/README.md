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
| `validation.jsonl` | The RULER eval set — one `{"input": <prompt>, "outputs": [<answer>]}` per line. A run scores a hit when `outputs[0]` is a substring of the model's generation. |
| `run_ruler.py` | **MHA** runner. Boots an in-process `sgl.Engine` with vortex sparsity and scores `validation.jsonl`. Knobs via env vars (below). |
| `run_ruler_mla.py` | **MLA** runner (DeepSeek/GLM latent attention). CLI-driven; defaults reproduce the known-good GLM-4.7-Flash config. |
| `sweep_flows.sh` | Sweeps every built-in flow through RULER and prints an accuracy table — 9 MHA flows × 2 indexer backends + 2 MLA flows. |
| `run_profile_mla.py` | **MLA selection-quality profiler**: drives a short decode with `attention_backend=cuda_mla_profile` and reports per-layer/per-head **p-coverage** + **recall@N**. |
| `ruler_output.jsonl` | Last run's raw generations (overwritten each `run_ruler.py` run; gitignored noise). |

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

## MHA — `run_ruler.py`

```bash
# default: Qwen/Qwen3-4B, gqa_block_sparse_attention, flashinfer indexer
CUDA_VISIBLE_DEVICES=0 python examples/ruler/run_ruler.py

# pick a different model (positional arg 1) and flow
CUDA_VISIBLE_DEVICES=0 VORTEX_MODULE=lserve_sparse_attention \
    python examples/ruler/run_ruler.py Qwen/Qwen3-4B
```

Prints `Ruler Accuracy [<flow>]: NN.NN%`.

| Env var | Default | Meaning |
|---------|---------|---------|
| `VORTEX_MODULE` | `gqa_block_sparse_attention` | Registered flow name from `vortex_torch/flow/algorithms.py`. |
| `VORTEX_ATTENTION_BACKEND` | `flashinfer` | Indexer backend: `flashinfer` or `trtllm`. (`TopK`/`Union` flows are trtllm-only; `topK`/`approxTopK` flows run under either.) |
| `DISABLE_RADIX_CACHE` | `0` | Set `1` for flows whose `forward_indexer` uses `Save(...)` (e.g. `running_avg_block_sparse`) — otherwise sglang's prefix-radix cache corrupts the saved per-request state. |
| `ENABLE_VORTEX_SPARSITY` | `1` | `0` runs **dense** sglang (no sparse path) to confirm the reference accuracy. |
| `RULER_SERVER_URL` | _(unset)_ | If set, drive an already-running sglang server over HTTP (`/generate`) instead of building an in-process engine; the server's launch flags define the config (see `examples/misc/server_launch.sh`). |

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

Key flags: `--model`, `--module`, `--attn-backend`, `--block`/`--topk`,
`--dense`, `--n`, `--tp`, `--kv-cache-dtype`, `--dump` (`--help` for all).

## Sweep all flows — `sweep_flows.sh`

```bash
examples/ruler/sweep_flows.sh            # all: 18 MHA (9 flows × 2 backends) + 2 MLA
examples/ruler/sweep_flows.sh mha        # just the 18 MHA runs
examples/ruler/sweep_flows.sh mla        # just the 2 MLA runs

# MLA part needs the GLM env:
MLA_PY="conda run -n vortex_glm python" examples/ruler/sweep_flows.sh
```

Runs one flow per free GPU in waves (re-detecting free GPUs each wave via
`algorithm_scientist/free_gpus.sh`), auto-sets `DISABLE_RADIX_CACHE=1` for
`running_avg_block_sparse`, then prints:

```
===== RULER flow sweep — accuracy =====
flow                               backend     accuracy
block_sparse_attention             flashinfer  99.00%
...
rope_aware_block_sparse_mla        cuda_mla    100.0%
lserve_centroid_mla                cuda_mla    99.0%
```

Per-run logs land in `examples/ruler/sweep_results/logs/` (gitignored). Override
`MODEL`, `MLA_MODEL`, `MHA_PY`, `MLA_PY`, `BACKENDS`, `HF_HOME`, `OUT` via env.

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
- **A flow well below ~0.85** has broken attention: widen `vortex_topk_val` /
  `vortex_topk_ratio`, check the indexer scoring, or confirm `DISABLE_RADIX_CACHE`
  for `Save`-based flows.
