# Response: Robustness to stochastic early termination

## 1. Conclusion

The end-to-end accuracy is robust across a wide range of stochastic
early-termination settings. For Qwen3-4B, RULER 16K remains 98.0--98.4% through
`tol=0.85` and is 97.0% at `tol=0.95`; AIME24 mean@16 remains stable over the
same range. Llama-3.1-8B likewise remains at 97--98% through `tol=0.95`.
Meanwhile, expected radix rounds decrease by 25% for Qwen and 29% for Llama.
Only `tol=1.0`, which skips refinement, is unstable and should be avoided.

## 2. End-to-end results

`Mean ± std` denotes mean ± standard deviation. Mass-recall std is measured
over randomized fill orders; Qwen RULER std is measured across nine flows.

### Qwen3-4B

| Tolerate ratio | Block recall | Mass recall, mean ± std | Expected rounds | RULER 16K, mean ± std (9 flows) | AIME24 mean@16 | AIME24 pass@16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (exact) | 1.0000 | 11.40% ± 0.00 | 1.980 | 98.1% ± 1.1 | 0.7271 | 0.8333 |
| 0.05 | 0.9997 | 11.41% ± 0.02 | 1.973 | 98.3% ± 1.5 | — | — |
| 0.15 | 0.9946 | 11.45% ± 0.04 | 1.914 | 98.3% ± 1.5 | 0.7229 | 0.8333 |
| 0.25 | 0.9859 | 11.44% ± 0.04 | 1.859 | 98.4% ± 1.3 | 0.6958 | 0.8333 |
| 0.35 | 0.9717 | 11.42% ± 0.04 | 1.805 | 98.3% ± 1.9 | 0.7104 | 0.8667 |
| 0.45 | 0.9623 | 11.40% ± 0.11 | 1.766 | 98.2% ± 1.6 | 0.7021 | 0.8333 |
| 0.55 | 0.9580 | 11.47% ± 0.09 | 1.758 | 98.3% ± 1.9 | 0.6896 | 0.8333 |
| 0.65 | 0.9313 | 11.32% ± 0.09 | 1.695 | 98.0% ± 1.7 | 0.7146 | 0.8333 |
| 0.75 | 0.8917 | 11.31% ± 0.22 | 1.621 | 98.1% ± 1.7 | 0.7167 | 0.8667 |
| 0.85 | 0.8592 | 11.53% ± 0.10 | 1.562 | 98.1% ± 1.5 | 0.7042 | 0.8333 |
| 0.95 | 0.8167 | 11.77% ± 0.37 | 1.492 | 97.0% ± 2.2 | 0.7083 | 0.8333 |

### Llama-3.1-8B-Instruct

| Tolerate ratio | Block recall | Mass recall, mean ± std | Expected rounds | RULER 16K |
|---:|---:|---:|---:|---:|
| 0 (exact) | 1.0000 | 24.58% ± 0.00 | 1.969 | 98% |
| 0.05 | 0.9997 | 24.58% ± 0.00 | 1.961 | 98% |
| 0.15 | 0.9947 | 24.58% ± 0.01 | 1.906 | 98% |
| 0.25 | 0.9869 | 24.57% ± 0.00 | 1.855 | 98% |
| 0.35 | 0.9716 | 24.74% ± 0.14 | 1.797 | 98% |
| 0.45 | 0.9586 | 24.81% ± 0.39 | 1.738 | 98% |
| 0.55 | 0.9428 | 24.02% ± 0.29 | 1.695 | 98% |
| 0.65 | 0.9071 | 23.91% ± 0.44 | 1.617 | 98% |
| 0.75 | 0.8778 | 23.90% ± 0.35 | 1.555 | 98% |
| 0.85 | 0.8256 | 23.19% ± 0.17 | 1.469 | 97% |
| 0.95 | 0.7701 | 22.90% ± 0.21 | 1.391 | 98% |
| 1.0 | 0.4627 | 20.24% ± 0.26 | 1.000 | 0% |

## 3. Experimental details

- Exact top-k is `tol=0`; larger ratios permit the two-pass BF16 radix selector
  to terminate earlier.
- Block recall is measured against exact top-k. Mass recall measures attention
  mass on the selected scored pages, excluding always-retained BOS/EOS pages.
- Because threshold-bin insertion has nondeterministic atomic arrival order,
  mass recall is reported as mean ± std over five randomized fill orders for
  Qwen and three for Llama.
- Qwen RULER uses 16K context, 100 examples per flow, and nine flows; its table
  reports mean ± std across flows. AIME24 uses 30 questions × 16 sampled
  generations. Llama RULER uses one 100-example run per tolerance.
- All runs use the TensorRT-LLM block-table backend, block/page size 32, and
  top-k 29.
