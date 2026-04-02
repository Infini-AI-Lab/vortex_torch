# Vortex Torch Examples

End-to-end accuracy evaluation and profiling pipelines for Vortex sparse attention on top of the SGLang inference engine. The scripts in this directory evaluate different TopK kernel variants, mapping functions, KV-cache quantization settings, and external sparse-attention backends on math reasoning benchmarks.

---

## Mapping Functions Reference

The TopK Stage-1 radix histogram uses 256 uint8 bins. A **mapping function** transforms raw attention scores before binning to improve bucket uniformity and reduce tail latency. Set via `--topk-mapping-mode`.

| Mode | Name | Formula | Requires Calibration | Hyperparameter (`--topk-mapping-power`) |
|------|------|---------|---------------------|-----------------------------------------|
| 0 | None | FP16 bit-pattern bucketing | No | — |
| 1 | LUT CDF | `lut[original_bin]` (CDF equalization) | Yes (`--topk-mapping-lut-path`) | — |
| 2 | Quantile | Binary search over 256 float thresholds | Yes (`--topk-mapping-quantiles-path`) | — |
| 3 | Power | `sign(x) * \|x\|^p` | No | `p` (exponent, default 0.5) |
| 4 | Log | `sign(x) * log(\|x\| + 1)` | No | — |
| 5 | Index Cache | Reuse top-k indices from a preceding layer | No | — (see `--index-cache-shared-layers`) |
| 6 | Asinh | `asinh(beta * x)` | No | `beta` (default 0.5) |
| 7 | Log1p | `sign(x) * log1p(alpha * \|x\|)` | No | `alpha` (default 0.5) |
| 8 | Trunc8 | BF16 upper-8-bit bucketing | No | — |

Modes 1 and 2 require an offline calibration step (see `calibrate_topk.py` in `benchmarks/`). Modes 3, 6, and 7 accept a tunable hyperparameter via `--topk-mapping-power`.

---

## Python Scripts

### `verify_algo.py` — End-to-End Accuracy Benchmark

The primary evaluation script. Loads AMC 2023 math problems from `amc23.jsonl`, runs inference via the SGLang engine with Vortex sparse attention, and scores answers using `lighteval`'s extractive-match metric. Reports `mean@N`, `pass@N`, throughput, and memory access cost.

**Usage:**

```bash
python verify_algo.py [OPTIONS]
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--trials` | 2 | Number of trials (each prompt repeated N times) |
| `--topk-val` | 30 | Number of top-k pages to select per segment |
| `--page-size` | 16 | Tokens per KV-cache page |
| `--vortex-module-name` | `gqa_block_sparse_attention` | Sparse attention algorithm module |
| `--model-name` | `Qwen/Qwen3-1.7B` | HuggingFace model identifier |
| `-f`, `--full-attention` | off | Disable sparse attention (full-attention baseline) |
| `--mem` | 0.8 | Static GPU memory fraction for SGLang |
| `--kv-cache-dtype` | `auto` | KV cache dtype: `auto`, `fp8_e5m2`, `fp8_e4m3`, `int8` |
| `--topk-type` | `naive` | TopK kernel: `naive` (CUB radix sort) or `sglang` (fast two-stage radix) |
| `--topk-mapping-mode` | 0 | Mapping function for Stage-1 binning (see table above) |
| `--topk-mapping-power` | 0.5 | Hyperparameter for modes 3/6/7 |
| `--topk-mapping-lut-path` | None | `.npy` uint8[256] LUT for mode 1 |
| `--topk-mapping-quantiles-path` | None | `.npy` float32[256] quantiles for mode 2 |
| `--index-cache-shared-layers` | None | Layer IDs that skip the indexer and reuse a previous layer's indices |

**Fixed engine settings:** `attention_backend=flashinfer`, `vortex_max_seq_lens=12288`, layer 0 skipped, `reserved_bos=1`, `reserved_eos=2`. Sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_new_tokens=8192`.

**Index cache note (mode 5):** When `--topk-mapping-mode 5` is set without `--index-cache-shared-layers`, the script defaults to even layers `[2, 4, 6, ..., 26]` and internally resets the mapping mode to 0 while passing the shared-layer list to the engine.

**Example — full-attention baseline:**

```bash
python verify_algo.py --full-attention --trials 8 --mem 0.7
```

**Example — sglang TopK with power mapping:**

```bash
python verify_algo.py \
  --topk-type sglang \
  --topk-mapping-mode 3 \
  --topk-mapping-power 0.25 \
  --trials 8 --topk-val 30 --mem 0.7
```

**Example — sglang TopK with calibrated LUT:**

```bash
python verify_algo.py \
  --topk-type sglang \
  --topk-mapping-mode 1 \
  --topk-mapping-lut-path calibration/lut.npy \
  --trials 8 --topk-val 30 --mem 0.7
```

---

### `verify_aim24.py` — AIME 2024 Throughput Test (Legacy)

A standalone throughput script that loads AIME 2024 from HuggingFace (`HuggingFaceH4/aime_2024`), builds chat prompts using the Qwen3 tokenizer with `enable_thinking=True`, and repeats each prompt 8 times. Outputs a JSONL file with generation results and timing metadata. Does **not** compute accuracy metrics.

**Usage:**

```bash
python verify_aim24.py
```

All settings are hard-coded (no CLI arguments):

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-0.6B` |
| Page size | 16 |
| Selected pages | 29 |
| Max sequence length | 20480 |
| Module | `block_sparse_attention` |
| Memory fraction | 0.9 |
| Max new tokens | 16384 |
| CUDA graph | Enabled |

---

## Shell Scripts

All shell scripts set `CUDA_VISIBLE_DEVICES` and save timestamped logs to `results/`.

### `verify_algo.sh` — Baseline TopK Comparison (Naive vs SGLang)

Runs `verify_algo.py` with `block_sparse_attention` comparing the `naive` and `sglang` TopK kernels. Each configuration is repeated `REPEAT_COUNT` times (default 3, overridable via environment variable).

```bash
REPEAT_COUNT=5 bash verify_algo.sh
```

### `verify_algo_topk.sh` — Naive vs SGLang Comparison

Similar to `verify_algo.sh` but simpler: runs `naive` TopK and `sglang` TopK back-to-back for `block_sparse_attention`, each with 8 trials.

### `verify_algo_quant.sh` — INT8 KV-Cache Quantization

Tests sparse attention with `--kv-cache-dtype int8` to measure accuracy under quantized KV caches.

```bash
bash verify_algo_quant.sh
```

### `verify_sparse_backends.sh` — External Sparse Attention Backends

Evaluates three external sparse-attention algorithms integrated via the Vortex flow interface:

- `nsa` (Native Sparse Attention)
- `fsa` (Flash Sparse Attention)
- `flash_moba` (Flash MoBA)

```bash
bash verify_sparse_backends.sh
```

### `verify_algo_topk_mapping.sh` — Full Mapping Mode Sweep

Comprehensive sweep across all mapping modes:

1. **Baseline:** `naive` TopK, mode 0
2. **Calibration:** runs `calibrate_topk.py` to generate `lut.npy` and `quantiles.npy` (skipped if files exist)
3. **Mode 1** (LUT CDF) and **Mode 2** (Quantile) with calibrated tables
4. **Modes 0, 3, 4** (no calibration needed) — Power mode uses `--topk-mapping-power 0.5`
5. **Mode 6** (Asinh) — sweeps `beta` in `[0.5, 1.0, 2.0]`
6. **Mode 7** (Log1p) — sweeps `alpha` in `[0.5, 1.0, 2.0]`

```bash
export CUDA_VISIBLE_DEVICES=0
bash verify_algo_topk_mapping.sh
```

### `verify_algo_topk_mapping_new.sh` — Parametric Mapping Sweep (Modes 3, 6, 7)

Focused hyperparameter sweep for the three parametric modes, preceded by an auto-tuning step:

| Mode | Parameter | Sweep Values |
|------|-----------|-------------|
| 3 (Power) | `p` | 0.1, 0.25, 0.75, 0.9 |
| 6 (Asinh) | `beta` | 0.1, 0.5, 1.0, 2.0, 4.0 |
| 7 (Log1p) | `alpha` | 0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0 |

Requires `calibration/raw_histograms.npy` for the auto-tune step.

```bash
export CUDA_VISIBLE_DEVICES=5
bash verify_algo_topk_mapping_new.sh
```

### `verify_algo_topk_mapping_indexcache.sh` — Index Cache (Mode 5)

Tests the index-cache optimization where even-numbered layers `[2, 4, 6, ..., 26]` reuse top-k indices from the nearest preceding full layer, skipping their indexer entirely.

```bash
bash verify_algo_topk_mapping_indexcache.sh
```

### `run_topk_benchmark.sh` — Unified TopK Benchmark Pipeline

The most comprehensive benchmarking script. Three-step pipeline:

1. **Calibrate** — collect real-data histograms + LUT/quantile tables
2. **Kernel bench** — latency + histogram profiling across batch sizes, sequence lengths, and distributions, followed by distribution analysis plots and auto-tuning
3. **E2E accuracy** — full-attention baseline plus every mapping mode

```bash
bash run_topk_benchmark.sh --gpu 5 --trials 8 --model-name Qwen/Qwen3-1.7B
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-name` | `Qwen/Qwen3-1.7B` | HuggingFace model |
| `--topk-val` | 30 | Top-k pages |
| `--trials` | 8 | E2E trial count |
| `--mem` | 0.7 | GPU memory fraction |
| `--gpu` | 5 | CUDA device |
| `--algo` | `block_sparse_attention` | Sparse attention algorithm |
| `--skip-calibrate` | off | Reuse existing calibration |
| `--skip-kernel` | off | Skip kernel-level latency step |
| `--skip-e2e` | off | Skip E2E accuracy step |

### `run_distribution_analysis.sh` — Bucket Distribution Profiling (All Modes)

Three-step pipeline to analyze how each mapping mode affects the 256-bin bucket distribution:

1. **Calibrate** — collect real-data histograms (skippable with `--real-histograms`)
2. **Bench** — histogram profiling with modes 0–8 on `bucket_uniform` and `normal` distributions
3. **Analyze** — generate comparison plots and CSV bucket count tables

```bash
bash run_distribution_analysis.sh --gpu 5
bash run_distribution_analysis.sh --gpu 5 --real-histograms /path/to/raw_histograms.npy
```

### `run_distribution_analysis_new.sh` — Bucket Distribution Profiling (Modes 3, 6, 7)

Same pipeline as above but focused on parametric modes only, with an additional auto-tune step:

1. **Calibrate** (or skip with existing histograms)
2. **Auto-tune** — sweep hyperparameters on synthetic data
3. **Bench** — histogram profiling for modes 3, 6, 7, 8
4. **Analyze** — comparison plots + tables

```bash
bash run_distribution_analysis_new.sh --gpu 5
```

---

## Benchmarks Directory Scripts

The `benchmarks/` directory contains standalone profiling and analysis tools used by the shell pipelines above. These can also be run independently.

### `calibrate_topk.py` — Offline Calibration

Runs the SGLang engine on real prompts from `amc23.jsonl` with histogram collection enabled. Produces three files:

- `lut.npy` — uint8[256] CDF-equalized LUT for mode 1
- `quantiles.npy` — float32[256] quantile breakpoints for mode 2
- `raw_histograms.npy` — raw per-sample 256-bin histograms

```bash
python benchmarks/calibrate_topk.py \
  --model-name Qwen/Qwen3-1.7B \
  --topk-val 30 --mem 0.7 \
  --output-dir calibration/
```

### `bench_topk.py` — Kernel-Level Latency Benchmark

Benchmarks `topk_output` (naive/CUB) and `topk_output_sglang` (fast radix) across configurable sweeps of batch size, sequence length, TopK value, KV heads, and score distributions. Optionally collects 256-bin histogram statistics.

```bash
python benchmarks/bench_topk.py \
  --batch-sizes 4 8 16 \
  --seq-lens 2048 4096 8192 \
  --topk-vals 30 \
  --num-kv-heads 2 \
  --distributions normal lognormal uniform bucket_uniform \
  --histogram \
  --repeat 100 \
  --output-json results.json
```

### `autotune_topk_mapping.py` — Hyperparameter Auto-Tuning

Sweeps hyperparameters for parametric mapping modes (3, 6, 7) using the `topk_profile_histogram` kernel on synthetic data. Ranks configurations by resolution rate, Gini coefficient, max/mean ratio, and nonzero bins.

```bash
python benchmarks/autotune_topk_mapping.py \
  --topk-val 30 --batch-size 4 --seq-len 4096 --num-kv-heads 2 \
  --real-histograms calibration/raw_histograms.npy \
  --output-json autotune_results.json
```

### `analyze_topk_distribution.py` — Visualization and Analysis

Loads profiling data and generates:
- Per-segment 256-bin bar charts
- Heatmaps (segments x bins, log-scale)
- Before/after LUT mapping comparisons
- Mode comparison grouped bar charts (Gini + max/mean)
- Distribution comparison plots across data sources
- CSV bucket count tables

```bash
python benchmarks/analyze_topk_distribution.py \
  --bench-json bench_distribution.json \
  --real-histograms calibration/raw_histograms.npy \
  --output-dir plots/
```

### `profile_topk_distribution.py` — Offline Table Generation

Computes LUT and quantile tables from pre-collected histograms or raw scores without running a model. Outputs a single `.npz` archive.

```bash
python benchmarks/profile_topk_distribution.py \
  --histograms-input raw_histograms.npy \
  --output mapping_tables.npz
```

### `greedy_layer_search.py` — Index Cache Layer Selection

Greedy forward-selection of layers whose indexer can be skipped (index cache). Iteratively adds layers to the shared set as long as accuracy stays above `--threshold` times the baseline.

```bash
cd examples && python ../benchmarks/greedy_layer_search.py \
  --model-name Qwen/Qwen3-1.7B \
  --topk-val 30 \
  --threshold 0.95 \
  --trials 1 \
  --num-layers 28 \
  --mem 0.7
```

---

## Data Files

| File | Description |
|------|-------------|
| `amc23.jsonl` | AMC 2023 math problems with `prompt` and `answer` fields, used by `verify_algo.py` and `calibrate_topk.py` |

---

## Output Structure

Results are saved under `results/` in timestamped directories:

```
results/
├── dist_analysis_YYYYMMDD_HHMMSS/
│   ├── step1_calibrate.log
│   ├── step2_autotune.log / step2_bench.log
│   ├── step3_bench.log / step3_analyze.log
│   ├── step4_analyze.log
│   ├── autotune_results.json
│   ├── bench_distribution.json
│   ├── distribution_comparison_*.png
│   ├── bucket_counts_*.csv
│   └── calibration/
│       ├── lut.npy
│       ├── quantiles.npy
│       └── raw_histograms.npy
├── topk_benchmark_YYYYMMDD_HHMMSS/
│   ├── kernel_latency.json
│   ├── e2e/
│   │   ├── full_attention_baseline.log
│   │   ├── sglang_mode0_none.log
│   │   └── ...
│   └── calibration/
└── *.log  (individual run logs)
```

---

## Quick Start: Typical Workflow

```bash
export CUDA_VISIBLE_DEVICES=0

# 1. Calibrate to generate LUT + quantile tables
python benchmarks/calibrate_topk.py \
  --model-name Qwen/Qwen3-1.7B --topk-val 30 --mem 0.7 \
  --output-dir examples/calibration/

# 2. Run full-attention baseline
python examples/verify_algo.py --full-attention --trials 8 --mem 0.7

# 3. Evaluate sparse attention with different mapping modes
python examples/verify_algo.py \
  --topk-type sglang --topk-mapping-mode 0 --trials 8 --mem 0.7

python examples/verify_algo.py \
  --topk-type sglang --topk-mapping-mode 3 --topk-mapping-power 0.25 \
  --trials 8 --mem 0.7

python examples/verify_algo.py \
  --topk-type sglang --topk-mapping-mode 6 --topk-mapping-power 1.0 \
  --trials 8 --mem 0.7

# 4. Or run the full pipeline in one shot
bash examples/run_topk_benchmark.sh --gpu 0 --trials 8
```
