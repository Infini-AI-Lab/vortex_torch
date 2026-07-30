# Response: Sensitivity of approximate radix top-k

## 1. Conclusion

Stochastic early termination is stable over a broad operating range. As the
tolerance increases, block recall and expected radix rounds decrease, but
attention-mass recall and downstream accuracy remain stable through
`tol=0.95`. This provides useful optimization headroom. The only clear failure
case is the degenerate `tol=1.0` endpoint, which skips refinement entirely.

## 2. Experimental results

Values reported as `mean ± std` use the standard deviation (std) across
randomized fill orders for mass recall and across nine flows for Qwen RULER.

### Qwen3-4B

| Tolerance | Block recall | Mass recall, mean ± std | Expected rounds | RULER 16K, mean ± std (9 flows) | AIME24 mean@16 | AIME24 pass@16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (exact) | 1.0000 | 11.40% ± 0.00 | 1.980 | 98.1% ± 1.1 | 0.7271 | 0.8333 |
| 0.15 | 0.9946 | 11.45% ± 0.04 | 1.914 | 98.3% ± 1.5 | 0.7229 | 0.8333 |
| 0.25 | 0.9859 | 11.44% ± 0.04 | 1.859 | 98.4% ± 1.3 | 0.6958 | 0.8333 |
| 0.45 | 0.9623 | 11.40% ± 0.11 | 1.766 | 98.2% ± 1.6 | 0.7021 | 0.8333 |
| 0.65 | 0.9313 | 11.32% ± 0.09 | 1.695 | 98.0% ± 1.7 | 0.7146 | 0.8333 |
| 0.85 | 0.8592 | 11.53% ± 0.10 | 1.562 | 98.1% ± 1.5 | 0.7042 | 0.8333 |
| 0.95 | 0.8167 | 11.77% ± 0.37 | 1.492 | 97.0% ± 2.2 | 0.7083 | 0.8333 |

Through `tol=0.85`, mean RULER stays within 0.4 percentage points of exact
top-k while expected rounds decrease by 21%. At `tol=0.95`, expected rounds
decrease by 25%; mass recall remains within its observed variation, and RULER
remains high.

### Llama-3.1-8B-Instruct

| Tolerance | Block recall | Mass recall, mean ± std | Expected rounds | RULER 16K |
|---:|---:|---:|---:|---:|
| 0 (exact) | 1.0000 | 24.58% ± 0.00 | 1.969 | 98% |
| 0.15 | 0.9947 | 24.58% ± 0.01 | 1.906 | 98% |
| 0.25 | 0.9869 | 24.57% ± 0.00 | 1.855 | 98% |
| 0.45 | 0.9586 | 24.81% ± 0.39 | 1.738 | 98% |
| 0.65 | 0.9071 | 23.91% ± 0.44 | 1.617 | 98% |
| 0.85 | 0.8256 | 23.19% ± 0.17 | 1.469 | 97% |
| 0.95 | 0.7701 | 22.90% ± 0.21 | 1.391 | 98% |
| 1.0 | 0.4627 | 20.24% ± 0.26 | 1.000 | 0% |

At `tol=0.95`, expected rounds decrease by 29% while RULER remains 98%. Only
`tol=1.0`, which eliminates the refinement pass, causes accuracy to collapse.

## 3. Experimental details

- `tol=0` is exact BF16 radix top-k; increasing the tolerance permits earlier
  termination.
- Block recall is the fraction of exact top-k pages retained. Mass recall is
  the attention mass captured on the selected scored pages, excluding the
  always-retained BOS/EOS pages.
- Mass recall is reported as mean ± std over five randomized fill orders for
  Qwen and three seeds for Llama. Qwen RULER is mean ± std across nine routing
  flows. AIME24 and Llama RULER are single benchmark evaluations per setting
  and are therefore reported as point estimates.
- RULER uses 16K context and 100 examples per flow. AIME24 uses Qwen3-4B,
  30 questions, and 16 sampled generations per question.
- All sparse runs use the TensorRT-LLM block-table backend, block/page size 32,
  and top-k 29.
