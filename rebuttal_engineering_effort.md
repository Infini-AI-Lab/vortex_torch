# Response: Engineering effort

## 1. Conclusion

Vortex reduces adding a sparse-attention method from a serving-engine
integration task to a compact algorithm description. For Quest, the complete
Vortex implementation is approximately 50 lines and requires no
method-specific CUDA/Triton or serving-runtime changes. A native SGLang or vLLM
implementation requires hundreds to over one thousand incremental lines across
kernels, cache management, runtime wiring, and tests.

## 2. Engineering-effort comparison

| Implementation | Algorithm code | Kernels | Runtime integration and tests | Total incremental LOC |
|---|---:|---:|---:|---:|
| **Vortex Quest (measured)** | 37 Python LOC | 0 | 13-line JSON | **~50** |
| **Native SGLang (estimated)** | 30–80 | 350–900 | 200–550 | **580–1,530** |
| **Native vLLM (estimated)** | 30–80 | 350–900 | 250–700 | **630–1,680** |

As a measured native reference, SGLang's Double Sparsity implementation
contains 212 executable LOC in its backend and 941 LOC in its Triton kernels:
**1,153 executable LOC before additional runtime wiring and tests**.

## 3. Counting and implementation details

Quest stores the coordinate-wise minimum and maximum key for each page and
uses them to compute a query-dependent upper bound. In Vortex:

- `create_cache` declares the two page summaries;
- `forward_cache` computes them using `CMin` and `CMax`; and
- `forward_indexer` composes `Multiply`, `Maximum`, `Sum`, `Max`, and `topK`.

The released `GQAQuestSparseAttention` implementation contains 37 executable
Python LOC after excluding comments, blank lines, and docstrings. Its engine
configuration is 13 lines.

The SGLang/vLLM values are estimates because neither engine provides a directly
comparable native Quest implementation. They include the incremental work to
implement cache-update and routing kernels, allocate auxiliary page state,
propagate block tables through attention metadata and the model runner, expose
configuration, and test batching, prefix caching, CUDA graphs, and tensor
parallelism. They assume existing sparse-decode and top-k primitives can be
reused; otherwise, the cost can be higher.
