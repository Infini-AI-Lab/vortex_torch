# GLM-4.7-Flash · MLA sparse-attention experiments (AIME26, mean@16)

**Driver:** `examples/misc/algo2.sh` → `examples/math/verify_algo.py`
&nbsp;·&nbsp; **Data:** `examples/math/aime26_glm.jsonl`
&nbsp;·&nbsp; **Model:** `zai-org/GLM-4.7-Flash`
&nbsp;·&nbsp; **trials:** 16 &nbsp;·&nbsp; **block=page=16** &nbsp;·&nbsp; **gen ≤ 32768 tok** &nbsp;·&nbsp; **tp=1**

## Plan

| # | module | backend | sparsity | runs |
|---|--------|---------|----------|------|
| 1 | `full_attention` | `flashinfer` (dense MLA) | — (dense baseline) | **1** (topk ignored) |
| 2 | `rope_aware_block_sparse_mla` | `cuda_mla` decode + Triton **tensor-core** indexer, **no layer skip** | topk ∈ {61,93,125,157,253} | 5 |
| 3 | `lserve_centroid_mla` | `cuda_mla` decode + Triton **tensor-core** indexer, **no layer skip** | topk ∈ {61,93,125,157,253} | 5 |

*Why these settings.* `full_attention` is dense, so it cannot use a vortex sparse
backend (those require `enable_vortex_sparsity=True`) — it runs on the dense
flashinfer MLA path as the accuracy ceiling / throughput-floor reference, once.
The two sparse modules run on the geometry-agnostic **`cuda_mla`** hand-CUDA decode;
**`--vortex-use-tensor-core`** turns on the bf16-MMA Triton indexer (hence
`--vortex-impl-backend triton`); **`--vortex-layers-skip`** with no values keeps
*every* layer sparse. The topk sweep traces the accuracy/throughput Pareto.


## Results

*Generated 2026-05-30 02:16:44 from `summary-glm4.7-flash/` (4 run(s)).*

| module | backend | topk | blk | TC | skip | mean@16 | pass@16 | pass@8 | pass@4 | tok/s | out_tok | e2e_s |
|---|---|--:|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| `rope_aware_block_sparse_mla` | cuda_mla | 29 | 64 | ✓ | none | 0.7042 | 0.9000 | 0.8791 | 0.8465 | 4778.5 | 8606843 | 1801.2 |
| `rope_aware_block_sparse_mla` | cuda_mla | 61 | 32 | ✓ | none | 0.7042 | 0.9000 | 0.8871 | 0.8481 | 4830.0 | 8422265 | 1743.7 |
| `rope_aware_block_sparse_mla` | cuda_mla | 61 | 64 | ✓ | none | 0.7396 | 0.9333 | 0.9175 | 0.8871 | 4221.4 | 8383809 | 1986.0 |
| `rope_aware_block_sparse_mla` | cuda_mla | 125 | 32 | ✓ | none | 0.7521 | 0.9333 | 0.9111 | 0.8695 | 4144.7 | 8276051 | 1996.8 |

## Original summaries (raw)

<details><summary><code>GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-30_01-41-34.json</code></summary>

```json
{
  "mean@16": 0.7041666666666667,
  "pass@16": 0.9,
  "pass@4": 0.8465018315018314,
  "pass@8": 0.8791401191401191,
  "total_example": 480,
  "e2e_time": 1801.155065536499,
  "total_tokens": 8606843,
  "throughput": 4778.513057917272,
  "args": {
    "trials": 16,
    "topk_val": 29,
    "topk_ratio": 0.0,
    "block_size": 64,
    "page_size": 64,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-30_00-33-53.json</code></summary>

```json
{
  "mean@16": 0.7041666666666667,
  "pass@16": 0.9,
  "pass@4": 0.8480952380952381,
  "pass@8": 0.8871328671328672,
  "total_example": 480,
  "e2e_time": 1743.732130765915,
  "total_tokens": 8422265,
  "throughput": 4830.022255941693,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-30_02-16-42.json</code></summary>

```json
{
  "mean@16": 0.7395833333333334,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.887051282051282,
  "pass@8": 0.9175446775446775,
  "total_example": 480,
  "e2e_time": 1986.0331807136536,
  "total_tokens": 8383809,
  "throughput": 4221.384154814268,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 64,
    "page_size": 64,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-30_01-09-27.json</code></summary>

```json
{
  "mean@16": 0.7520833333333333,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8694688644688644,
  "pass@8": 0.9111111111111111,
  "total_example": 480,
  "e2e_time": 1996.79616355896,
  "total_tokens": 8276051,
  "throughput": 4144.664914244077,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

