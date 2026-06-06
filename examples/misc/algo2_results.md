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

*Generated 2026-05-27 15:24:40 from `summary-glm4.7-flash/` (0 run(s)).*

| module | backend | topk | blk | TC | skip | mean@16 | pass@16 | pass@8 | pass@4 | tok/s | out_tok | e2e_s |
|---|---|--:|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| _(no summaries yet — run `examples/misc/algo2.sh`)_ |||||||||||||

## Original summaries (raw)

