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

*Generated 2026-05-27 21:47:27 from `summary-glm4.7-flash/` (13 run(s)).*

| module | backend | topk | blk | TC | skip | mean@16 | pass@16 | pass@8 | pass@4 | tok/s | out_tok | e2e_s |
|---|---|--:|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| `full_attention` | triton | 253 | 32 | · | [0] | 0.2583 | 0.7000 | 0.5814 | 0.4747 | 1265.8 | 2720623 | 2149.3 |
| `lserve_centroid_mla` | cuda_mla | 61 | 16 | ✓ | none | 0.6771 | 0.8667 | 0.8582 | 0.8192 | 4454.0 | 8734383 | 1961.0 |
| `lserve_centroid_mla` | cuda_mla | 93 | 16 | ✓ | none | 0.7042 | 0.9000 | 0.8661 | 0.8363 | 4183.3 | 8522305 | 2037.2 |
| `lserve_centroid_mla` | cuda_mla | 157 | 16 | ✓ | none | 0.7125 | 0.9000 | 0.8839 | 0.8478 | 4236.4 | 8569544 | 2022.9 |
| `lserve_centroid_mla` | cuda_mla | 253 | 16 | ✓ | none | 0.7479 | 0.9000 | 0.8889 | 0.8637 | 3924.7 | 8466111 | 2157.1 |
| `rope_aware_block_sparse_mla` | triton | 61 | 16 | ✓ | none | 0.6479 | 0.9000 | 0.8754 | 0.8170 | 3121.7 | 8712928 | 2791.1 |
| `rope_aware_block_sparse_mla` | cuda_mla | 61 | 16 | ✓ | none | 0.6646 | 0.9000 | 0.8639 | 0.8106 | 3012.0 | 8608008 | 2857.9 |
| `rope_aware_block_sparse_mla` | triton | 93 | 16 | ✓ | none | 0.2667 | 0.6667 | 0.5876 | 0.4918 | 2979.4 | 3430788 | 1151.5 |
| `rope_aware_block_sparse_mla` | cuda_mla | 93 | 16 | ✓ | none | 0.7146 | 0.9000 | 0.8876 | 0.8538 | 2994.1 | 8441414 | 2819.4 |
| `rope_aware_block_sparse_mla` | triton | 125 | 16 | ✓ | none | 0.2750 | 0.6333 | 0.5685 | 0.4817 | 2169.5 | 2598418 | 1197.7 |
| `rope_aware_block_sparse_mla` | cuda_mla | 125 | 16 | ✓ | none | 0.7250 | 0.9333 | 0.9204 | 0.8797 | 2928.4 | 8420886 | 2875.6 |
| `rope_aware_block_sparse_mla` | cuda_mla | 157 | 16 | ✓ | none | 0.7333 | 0.9333 | 0.9119 | 0.8784 | 2842.0 | 8437797 | 2968.9 |
| `rope_aware_block_sparse_mla` | cuda_mla | 253 | 16 | ✓ | none | 0.7438 | 0.9000 | 0.8912 | 0.8643 | 3884.0 | 8324371 | 2143.2 |

## Original summaries (raw)

<details><summary><code>zai-org_GLM-4.7-Flash_full_attention_16trials_tp1_2026-05-27_19-00-36.json</code></summary>

```json
{
  "mean@16": 0.25833333333333336,
  "pass@16": 0.7,
  "pass@4": 0.47474358974358977,
  "pass@8": 0.5814063714063714,
  "total_example": 480,
  "e2e_time": 2149.259838104248,
  "total_tokens": 2720623,
  "throughput": 1265.8418269238782,
  "args": {
    "trials": 16,
    "topk_val": 253,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "full_attention",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": false,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "cuda",
    "vortex_use_tensor_core": false,
    "vortex_layers_skip": [
      0
    ]
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-07-15.json</code></summary>

```json
{
  "mean@16": 0.6770833333333334,
  "pass@16": 0.8666666666666667,
  "pass@4": 0.8191941391941392,
  "pass@8": 0.8582413882413883,
  "total_example": 480,
  "e2e_time": 1961.034833908081,
  "total_tokens": 8734383,
  "throughput": 4453.966267694255,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-08-29.json</code></summary>

```json
{
  "mean@16": 0.7041666666666667,
  "pass@16": 0.9,
  "pass@4": 0.8363003663003663,
  "pass@8": 0.8660968660968661,
  "total_example": 480,
  "e2e_time": 2037.1976444721222,
  "total_tokens": 8522305,
  "throughput": 4183.347169640134,
  "args": {
    "trials": 16,
    "topk_val": 93,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-45-11.json</code></summary>

```json
{
  "mean@16": 0.7125,
  "pass@16": 0.9,
  "pass@4": 0.8478205128205129,
  "pass@8": 0.8838539238539238,
  "total_example": 480,
  "e2e_time": 2022.85959649086,
  "total_tokens": 8569544,
  "throughput": 4236.351358673607,
  "args": {
    "trials": 16,
    "topk_val": 157,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-47-25.json</code></summary>

```json
{
  "mean@16": 0.7479166666666667,
  "pass@16": 0.9,
  "pass@4": 0.8636813186813187,
  "pass@8": 0.8888603988603988,
  "total_example": 480,
  "e2e_time": 2157.129903793335,
  "total_tokens": 8466111,
  "throughput": 3924.7107858976215,
  "args": {
    "trials": 16,
    "topk_val": 253,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-12-09.json</code></summary>

```json
{
  "mean@16": 0.6479166666666667,
  "pass@16": 0.9,
  "pass@4": 0.817032967032967,
  "pass@8": 0.8753768453768453,
  "total_example": 480,
  "e2e_time": 2791.0943007469177,
  "total_tokens": 8712928,
  "throughput": 3121.6888650692867,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-31-31.json</code></summary>

```json
{
  "mean@16": 0.6645833333333333,
  "pass@16": 0.9,
  "pass@4": 0.8106227106227106,
  "pass@8": 0.8638539238539239,
  "total_example": 480,
  "e2e_time": 2857.8958954811096,
  "total_tokens": 8608008,
  "throughput": 3012.008944626338,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_18-45-09.json</code></summary>

```json
{
  "mean@16": 0.26666666666666666,
  "pass@16": 0.6666666666666666,
  "pass@4": 0.49179487179487175,
  "pass@8": 0.5876275576275576,
  "total_example": 480,
  "e2e_time": 1151.4940140247345,
  "total_tokens": 3430788,
  "throughput": 2979.4232173284277,
  "args": {
    "trials": 16,
    "topk_val": 93,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-30-51.json</code></summary>

```json
{
  "mean@16": 0.7145833333333333,
  "pass@16": 0.9,
  "pass@4": 0.8538461538461539,
  "pass@8": 0.8875602175602175,
  "total_example": 480,
  "e2e_time": 2819.3768174648285,
  "total_tokens": 8441414,
  "throughput": 2994.0708697429395,
  "args": {
    "trials": 16,
    "topk_val": 93,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_18-45-50.json</code></summary>

```json
{
  "mean@16": 0.275,
  "pass@16": 0.6333333333333333,
  "pass@4": 0.48166666666666663,
  "pass@8": 0.5684615384615384,
  "total_example": 480,
  "e2e_time": 1197.7052600383759,
  "total_tokens": 2598418,
  "throughput": 2169.4970262690035,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/math/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-31-48.json</code></summary>

```json
{
  "mean@16": 0.725,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8797435897435898,
  "pass@8": 0.9204428904428904,
  "total_example": 480,
  "e2e_time": 2875.6124556064606,
  "total_tokens": 8420886,
  "throughput": 2928.379999044083,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-33-20.json</code></summary>

```json
{
  "mean@16": 0.7333333333333333,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8783699633699633,
  "pass@8": 0.9119114219114218,
  "total_example": 480,
  "e2e_time": 2968.9209332466125,
  "total_tokens": 8437797,
  "throughput": 2842.041667567412,
  "args": {
    "trials": 16,
    "topk_val": 157,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_21-10-14.json</code></summary>

```json
{
  "mean@16": 0.74375,
  "pass@16": 0.9,
  "pass@4": 0.8643223443223443,
  "pass@8": 0.8912483812483812,
  "total_example": 480,
  "e2e_time": 2143.240613937378,
  "total_tokens": 8324371,
  "throughput": 3884.0114105094244,
  "args": {
    "trials": 16,
    "topk_val": 253,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
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

