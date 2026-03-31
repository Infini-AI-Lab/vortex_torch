# TopK Kernel Benchmarking Suite

Standalone benchmarking for Vortex's three topk kernel variants, measuring kernel-level latency isolated from the full SGLang inference pipeline.

## Kernel Variants

| Kernel | Description |
|--------|-------------|
| `naive` | CUB radix sort (bf16 only) |
| `sglang_m0` | Two-stage hierarchical radix sort, no mapping |
| `sglang_m1` | + LUT mapping (requires `--lut-path`) |
| `sglang_m2` | + Quantile mapping (requires `--quantiles-path`) |
| `sglang_m3` | + Power mapping (configurable via `--mapping-power`) |
| `sglang_m4` | + Log mapping |

## Quick Start

```bash
# Activate environment
source /scr/dataset/yuke/xinrui/uv_env/vortex/bin/activate

# Quick single-config test
python benchmarking/bench_topk.py \
  --batch-sizes 8 \
  --seq-lens 4096 \
  --topk-vals 30 \
  --num-kv-heads 2 \
  --repeat 200

# Sweep with histogram analysis
python benchmarking/bench_topk.py \
  --batch-sizes 4 8 16 \
  --seq-lens 2048 4096 8192 \
  --topk-vals 30 64 \
  --num-kv-heads 2 \
  --repeat 100 \
  --histogram

# Full sweep with JSON output
python benchmarking/bench_topk.py \
  --output-json benchmarking/results.json \
  --histogram
```

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-sizes` | 1 4 8 16 32 64 | Batch sizes to sweep |
| `--seq-lens` | 1024 2048 4096 8192 | Sequence lengths to sweep |
| `--topk-vals` | 16 30 64 | TopK values to sweep |
| `--num-kv-heads` | 2 4 8 | KV head counts to sweep |
| `--page-size` | 16 | Tokens per page |
| `--reserved-bos` | 1 | Reserved BOS pages |
| `--reserved-eos` | 2 | Reserved EOS pages |
| `--score-dtype` | bfloat16 | Score tensor dtype (bfloat16 or float32) |
| `--distributions` | normal lognormal uniform | Score distributions to test |
| `--warmup` | 10 | Warmup iterations |
| `--repeat` | 100 | Timed iterations |
| `--mapping-power` | 0.5 | Power parameter for mode=3 |
| `--lut-path` | None | Path to .npy uint8[256] LUT for mode=1 |
| `--quantiles-path` | None | Path to .npy float32[256] quantiles for mode=2 |
| `--output-json` | None | Save results to JSON file |
| `--filter-kernels` | None | Only run specific kernels (e.g., `naive sglang_m0`) |
| `--histogram` | False | Collect bin distribution statistics |

## Histogram Analysis

When `--histogram` is passed, each config additionally runs `topk_profile_histogram` and reports:

- **max/mean ratio**: Peak bin count divided by average (lower = more uniform)
- **Gini coefficient**: Inequality measure of bin distribution (0 = perfectly uniform)
- **nonzero_bins**: How many of the 256 bins received any values

This shows whether mapping modes improve bin uniformity for a given score distribution.

## Output Format

```
TopK Kernel Benchmark Results
GPU: NVIDIA H100 80GB HBM3 | SM count: 132

bs=8 | seq=4096 | topk=30 | heads=2 | pages/seg=256 | dist=normal
  naive               : 0.0420ms (median) +/- 0.0030ms  [min=0.0390, max=0.0510]
  sglang mode=0       : 0.0310ms (median) +/- 0.0020ms  [min=0.0290, max=0.0380]
  sglang mode=3       : 0.0330ms (median) +/- 0.0020ms  [min=0.0300, max=0.0400]
  sglang mode=4       : 0.0320ms (median) +/- 0.0020ms  [min=0.0300, max=0.0390]
  histogram stats    : max/mean=3.99  gini=0.568  nonzero_bins=70/256
```
