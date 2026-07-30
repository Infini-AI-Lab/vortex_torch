# Q2/Q3: Workload planning, fusion, and native-API comparison

**Response.** We added a controlled end-to-end ablation using Quest. At the
same memory fraction, Vortex is **3.86×/3.30×** faster than a native-API-style
implementation at top-\(k=61/125\). The gains come from radix top-\(k\)
(**1.25×/1.19×**), workload planning (**1.26×/1.29×**), and fusion
(**2.44×/2.14×**). The fully optimized configuration also supports a larger KV
cache; at `mem=0.85`, it reaches **4.40×/3.73×** the native baseline throughput.

| Implementation | Memory | Top-\(k=61\) tok/s (speedup) | Top-\(k=125\) tok/s (speedup) |
|---|---:|---:|---:|
| Native API: padded, unfused, `torch.topk` | 0.60 | 2,676.7 (1.00×) | 2,643.4 (1.00×) |
| + Vortex radix top-\(k\) | 0.60 | 3,352.5 (1.25×) | 3,154.5 (1.19×) |
| + Vortex workload planner | 0.60 | 4,233.5 (1.58×) | 4,064.9 (1.54×) |
| + Vortex fusion and CUDA graphs | 0.60 | **10,329.8 (3.86×)** | **8,717.5 (3.30×)** |
| Fully optimized Vortex (larger KV cache) | 0.85 | **11,770.9 (4.40×)** | **9,856.5 (3.73×)** |

The native baseline implements the same Quest block selection with stock
PyTorch operators and `torch.topk`, constructs a padded request × KV-head ×
chunk worklist, and passes the selected block table to TensorRT-LLM's sparse
attention API. Thus, it measures the API-composition approach requested by the
reviewer, rather than comparing only attention-kernel latency. We do not label
this row as a direct FlexAttention or FlashInfer measurement.

Quest is a useful fusion case because its indexer contains two multiplications,
an elementwise maximum, and two reductions. With fusion disabled, these
operators launch separately and materialize cross-operator intermediates.
Vortex fusion avoids those intermediates. Its planner additionally creates a
compact device-side prefix-sum worklist containing only active ragged chunks;
the naïve baseline pads every request to the longest active sequence and must
run eagerly because that maximum is host-dependent.

**Setup.** Qwen3-4B, AIME24 (30 problems × 16 generations), one NVIDIA B200,
maximum generation length 16,384, block size = page size = 16, TensorRT-LLM
sparse attention, Triton tensor-core indexer, layer skip `[0]`,
`workload_chunk_size=32`, and seed 0. All controlled rows use `mem=0.60`.
The final `mem=0.85` row is reported separately as a deployment point because
the unfused implementation exhausts memory at that fraction. Accuracy remains
comparable across configurations (mean@16 ranges from 0.579 to 0.637).
