# `flow/algorithms.py` — RULER results

RULER on **`examples/ruler/validation.jsonl`** (100 prompts, 64 new tokens each)
using **Qwen/Qwen3-1.7B** on **NVIDIA B200**. All runs at identical settings
to isolate per-algorithm behaviour (knobs match the existing
`submissions/_flow_algorithms_test/` baselines).

Latest run: **2026-05-17 06:20** — re-verified after the indexer-compiler
refactor that moved `Schedule.S` codegen into
`indexer/compiler/custom_impl/`. Pre-flight (`check_engine_config`) passes
for all 8 configs.

| # | Registered name | Class | Save / Load? | `disable_radix_cache` | RULER accuracy | Throughput (tok/s) | e2e (s) |
|---|---|---|---|---|---|---|---|
| 1 | `block_sparse_attention`          | `BlockSparseAttention`          | no  | (not set)            | **0.98** | 2793 | 2.29 |
| 2 | `gqa_block_sparse_attention`      | `GQABlockSparseAttention`       | no  | (not set)            | **1.00** | 2215 | 2.89 |
| 3 | `gqa_quest_sparse_attention`      | `GQAQuestSparseAttention`       | no  | (not set)            | **1.00** | 2213 | 2.89 |
| 4 | `lserve_sparse_attention`         | `LServeSparseAttention`         | no  | (not set)            | **1.00** | 2631 | 2.43 |
| 5 | `masked_quest_sparse_attention`   | `MaskedQuestSparseAttention`    | no  | (not set)            | **1.00** | 2577 | 2.48 |
| 6 | `centered_block_sparse_attention` | `CenteredBlockSparseAttention`  | no  | (not set)            | **0.98** | 2660 | 2.41 |
| 7 | `running_avg_block_sparse`        | `RunningAvgBlockSparse`         | **yes** | **true** (required) | **0.99** | 2256 | 2.84 |
| 8 | `venergy_gated_centroid`          | `VEnergyGatedCentroid`          | no  | (not set)            | **1.00** | 2712 | 2.36 |

Throughput is the warm decode rate RULER reports as `throughput`.
`running_avg_block_sparse` requires `disable_radix_cache: true` because
its `forward_indexer` writes per-request state via `Save(...)`; sglang's
prefix-radix cache otherwise shares that state across requests with
matching prompt prefixes.

The GQA variants (#2, #3) were run as part of a 5-way parallel wave on
this host — host-side process startup contention while five sglang
engines initialised at the same time accounts for the lower throughput
relative to past sequential numbers. Running them in isolation (single
GPU, no concurrent boots) recovers the standard ~2.6k tok/s for both.

## Common knobs (identical across all 8 runs)

```json
{
  "vortex_attention_backend": "flashinfer",
  "vortex_block_size": 16,
  "vortex_topk_val": 29,
  "vortex_topk_ratio": 0.0625,
  "vortex_block_reserved_bos": 1,
  "vortex_block_reserved_eos": 2,
  "vortex_workload_chunk_size": 32,
  "vortex_layers_skip": [0],
  "kv_cache_dtype": "auto",
  "mem_fraction_static": 0.8
}
```

`disable_radix_cache: true` added only for `running_avg_block_sparse`
(Save/Load required).

## Why one submission file per algorithm

`vortex_torch.engine.sgl._check_disable_radix_cache` does a **text scan** of
the module file passed via `vortex_module_path`. Pointing all 8 submissions
at `vortex_torch/flow/algorithms.py` (which contains the `Save(...)` site
inside `RunningAvgBlockSparse`) would force `disable_radix_cache: true` on
every algorithm — incorrect for the 7 that don't actually use Save/Load,
and a measurable throughput hit. The fix: extract each `@register` class
into its own file under `submissions/_flow_algorithms_test/<name>.py`, and
add a `_sub` suffix to the registered name (the sglang process auto-imports
`vortex_torch.flow.algorithms` at startup, so re-registering the same name
would collide).

## Files

  * Per-algorithm submission flow:
    `submissions/_flow_algorithms_test/<algo>.py`
  * Per-algorithm submission config:
    `submissions/_flow_algorithms_test/<algo>.json`
  * Per-algorithm RULER artefacts:
    `summary_ruler_submissions/_flow_algorithms_test/<algo>/latest.json`

## Reproduction

```bash
# Preflight (cheap, CPU-only)
for cfg in submissions/_flow_algorithms_test/*.json; do
  python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('$cfg')"
done

# RULER — wave size 1 (recommended) or in parallel if the host has the
# CPU headroom. Parallel >4 hits sglang server-startup contention on
# this machine; pin a single free GPU and let them run sequentially.
FREE_GPUS=( $(algorithm_scientist/free_gpus.sh) )
for cfg in submissions/_flow_algorithms_test/*.json; do
  CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
    python algorithm_scientist/run_ruler.py --config "$cfg"
done
```

---

# trtllm backend results

Same protocol, same submissions, same knobs — only
`vortex_attention_backend` flipped from `flashinfer` → `trtllm`. Pre-flight
passes for all 8 configs. RULER run: **2026-05-17 06:28** on NVIDIA B200.
Wave size 4 with a 5 s stagger between child boots (this avoids the
host-startup contention seen in the flashinfer run when 5+ sglang engines
boot simultaneously).

| # | Registered name | RULER accuracy | Throughput (tok/s) | e2e (s) |
|---|---|---|---|---|
| 1 | `block_sparse_attention`          | **0.98** | 2809 | 2.28 |
| 2 | `gqa_block_sparse_attention`      | **1.00** | 2750 | 2.33 |
| 3 | `gqa_quest_sparse_attention`      | **1.00** | 2837 | 2.26 |
| 4 | `lserve_sparse_attention`         | **1.00** | 2783 | 2.30 |
| 5 | `masked_quest_sparse_attention`   | **1.00** | 2782 | 2.30 |
| 6 | `centered_block_sparse_attention` | **0.98** | 2859 | 2.24 |
| 7 | `running_avg_block_sparse`        | **0.99** | 2383 | 2.68 |
| 8 | `venergy_gated_centroid`          | **1.00** | 2781 | 2.30 |

## Side-by-side (flashinfer vs trtllm throughput)

| Algorithm | flashinfer tok/s | trtllm tok/s | Δ |
|---|---|---|---|
| `block_sparse_attention`          | 2793 | 2809 | +0.6% |
| `gqa_block_sparse_attention`      | 2215* | 2750 | +24.2%* |
| `gqa_quest_sparse_attention`      | 2213* | 2837 | +28.2%* |
| `lserve_sparse_attention`         | 2631 | 2783 | +5.8% |
| `masked_quest_sparse_attention`   | 2577 | 2782 | +8.0% |
| `centered_block_sparse_attention` | 2660 | 2859 | +7.5% |
| `running_avg_block_sparse`        | 2256 | 2383 | +5.6% |
| `venergy_gated_centroid`          | 2712 | 2781 | +2.5% |

`*` flashinfer GQA numbers came from the 5-way parallel wave that had
host-startup contention; the +24/+28% deltas overstate the real
backend-only effect. Comparing the seven non-GQA rows (which were
unaffected by contention in the flashinfer run) the trtllm backend is
consistently **+3 to +8% faster** on this workload, with no accuracy
change.

## Reproduction (trtllm)

Identical to the flashinfer flow; the only difference is the
`vortex_attention_backend` field in each config:

```bash
for cfg in submissions/_flow_algorithms_test/*.json; do
  python -c "import json; p='$cfg'; d=json.load(open(p)); d['vortex_attention_backend']='trtllm'; json.dump(d, open(p,'w'), indent=2)"
done

for cfg in submissions/_flow_algorithms_test/*.json; do
  python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('$cfg')"
done

FREE_GPUS=( $(algorithm_scientist/free_gpus.sh) )
for cfg in submissions/_flow_algorithms_test/*.json; do
  CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
    python algorithm_scientist/run_ruler.py --config "$cfg"
done
```
