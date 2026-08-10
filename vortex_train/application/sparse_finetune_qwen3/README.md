# Sparse-attention finetuning of Qwen3-4B on reasoning traces

The first application built on `vortex_train`: finetune `Qwen/Qwen3-4B` on GLM-5.1
reasoning traces with **block-sparse attention in both the forward and the backward
pass**, on 8× B200.

```bash
# 1. verify the templating (CPU only, ~1 min) -- do this first
python -m application.sparse_finetune_qwen3.verify_data --rows 20

# 2. train on 8 GPUs (1000 steps, checkpoint at the end and every 500)
torchrun --nproc_per_node 8 -m application.sparse_finetune_qwen3.train \
    --steps 1000 --topk 16 --block-kv 64 --block-q 1 \
    --reserve-bos 1 --reserve-local 1 \
    --save /scratch/zhuominc/ckpt_1k --save-every 500

# 3. score trained vs untrained in vortex_torch at the SAME budget
python -m application.sparse_finetune_qwen3.evaluate \
    --trained /scratch/zhuominc/ckpt_1k --base Qwen/Qwen3-4B \
    --tasks aime24 aime25 --trials 16

# dense baseline, same data and schedule
torchrun --nproc_per_node 8 -m application.sparse_finetune_qwen3.train \
    --steps 100 --attn sdpa
```

## Configuration

| | value |
|---|---|
| model | `Qwen/Qwen3-4B` — 36 layers, Hq=32, Hkv=8, D=128, GQA group 4 |
| data | `Jackrong/Qwen3.5-reasoning-700x` (≥12288 tok) + `r0b0tlab/qwen3.8-max-distillation-50k` (all) |
| max context | 40960 (the model's own `max_position_embeddings`) |
| selection | `block_topk` — centroid top-k, i.e. vortex's `block_sparse_attention` |
| `block_q` / `block_kv` | **1** / 64 |
| `topk` | 16 learned + 1 BOS + 1 local = **18 blocks = 1152 KV tokens**, context-independent |
| parallelism | FSDP2 (`fully_shard`), 1 sequence per rank, effective batch 8 |

`block_q=1` means **one selection per query token** — no averaging of queries into a
block. It is the accurate end of the tradeoff and the expensive one; see the root
README for the `block_q` sweep.

## Token positions below 2048 are dense, automatically

With `block_q=1` and `block_kv=64`, query token `i` can causally see only
`i//64 + 1` KV blocks, so it selects `min(32, i//64 + 1)`:

| query token | blocks available | selected | regime |
|---|---|---|---|
| 0–63 | 1 | 1 | dense (trivially) |
| 1024 | 17 | 17 | **dense** — fewer than the budget |
| 2047 | 32 | 32 | exactly at the budget |
| 2048+ | 33 … 640 | 32 | **sparse** |

No special-casing is needed: the causal mask already restricts the candidate set, and
the top-k emits `cnt = min(topk, available)` with `-1` padding rather than repeating a
block. Measured means match the closed form exactly (29.417 predicted vs 29.417
measured at 12288 tokens).

The consequence is worth stating rather than hiding: for a 12K sequence,
2048/12288 ≈ **17% of query positions are in the dense regime**, so the realised
speedup is below the FLOP ratio. That fraction shrinks as sequences get longer (5% at
40K).

## Datasets: two schemas, two length regimes

Both are **Qwen-distilled**, which is the point: a first run on `GLM-5.1` traces
regressed AIME24 by −0.34 and AIME25 by −0.25, consistent with pushing Qwen3 toward
another model's style.

| | `Qwen3.5-reasoning-700x` | `qwen3.8-max-distillation-50k` |
|---|---|---|
| schema | `input`/`output` flat strings | `messages` (role/content) |
| system turn | none | **yes** — prescribes the `<think>`/`\boxed{}` format |
| `<think>` in target | 1 | 1 |
| p50 tokens | ~11,171 | ~445 |
| rows ≥ 12k | 46% | **0%** |

Consequences, all handled explicitly:

- **Each dataset gets a registered adapter** (`ADAPTERS` in `data.py`). Inferring the
  field layout from whichever keys exist is how a schema change becomes a silent
  mistraining. Note `Qwen3.5-reasoning-700x` uses `conversation` (singular) where the
  older set used `conversations`.
- **The system prompt is carried through and masked.** It *is* the format contract the
  assistant text satisfies; training the completion without it teaches the model to emit
  that format unprompted.
- **`min_length` is per dataset** (`[12288, 0]`). A single global 12288 would discard
  100% of the short set. The long set exercises sparsity; the short set contributes
  format and answer supervision, and its steps correctly show `sparse% = 0` because
  they are below the 1152-token budget.
- **Streams are interleaved round-robin**, not concatenated — otherwise the LR schedule
  would decay away before the second dataset was ever reached.

## Validation every 100 steps

`--val-every 100` writes a checkpoint plus a `READY` marker; `watch_validate.py` scores
each on AIME24 (16 trials) through the serving engine at the trained budget and appends
to `history.jsonl`, printing a running curve.

```bash
# alongside training, on GPU 0
python -m application.sparse_finetune_qwen3.watch_validate \
    --val-dir /scratch/zhuominc/ckpt_qwen/val --trials 16
```

This exists because **loss is not the objective**: the previous run's loss fell 1.14 →
0.76 the whole way while AIME24 dropped 0.71 → 0.37. A 100-step validation catches that
in ~40 minutes instead of 7 hours.

Scoring is **out of process**, which was not the first design. In-process validation
failed twice, and both failures look like tuning problems but are structural:

1. the seven non-validating FSDP ranks wait in a barrier, and NCCL's 600 s watchdog
   SIGABRTs them as a hung collective;
2. with a longer timeout, the trainer's ~90 GB of *live* parameters are still resident —
   `empty_cache()` frees cached blocks, not live tensors — so the engine's KV allocation
   was OOM-killed (exit −9). Lowering the child's memory fraction only postpones it,
   since the trainer's peak tracks sequence length.

Out of process, training never pauses (measured: 2.6 min for 4 steps vs 24 min when
validation blocked), the engine sees actually-free memory, and a failed validation cannot
take the run down. The cost is that a score lands minutes after its step.

## Run 2 result (Qwen-distilled datasets, 1000 steps, validation every 100)

Loss fell 1.31 → 0.45. AIME24 mean@16, 30 problems, all at the trained budget:

| step | mean@16 | vs base |
|---:|---:|---:|
| base (untrained) | **0.7188** | — |
| 100 | 0.5917 | −0.127 |
| 300 | 0.6229 | −0.096 |
| 500 | 0.6146 | −0.104 |
| 700 | 0.6104 | −0.108 |
| 1000 | 0.5813 | −0.138 |

AIME25: base 0.6062 → step1000 0.5583 (−0.048).

**The finetune still hurts, but far less than the GLM-5.1 run** (−0.10 to −0.14 here
vs −0.34/−0.25 there), which is consistent with the cross-model-distillation diagnosis.
Two things the curve shows that a single end-point measurement could not:

* **The damage happens in the first 100 steps and then plateaus.** Steps 100–1000 sit in
  a 0.048 band — about 1.4 problems out of 30, i.e. noise. So this is not slow drift;
  it is an immediate shift, and 900 of the 1000 steps bought nothing either way.
* **Loss and accuracy move in opposite directions again.** 1.31 → 0.45 while AIME24 sat
  ~0.11 below base. Validating on the task rather than trusting the loss is the whole
  point of the per-100-step check.

### This run's mixture was broken — read the numbers with that in mind

```
Jackrong/Qwen3.5-reasoning-700x : yielded  39   (too_short 40)   <- EXHAUSTED
r0b0tlab/qwen3.8-max-50k        : yielded 961
=> 812 / 1000 steps ran DENSE (seqlen < 1152)
```

The dataset name is literal: ~700 rows total. Sharded 8 ways (`i % world == rank`) that
is ~88 per rank, and the ≥12288 filter keeps about half — ~39 usable examples per rank.
Round-robin drained it in ~78 steps and then fell back to the short stream for the
remaining 922.

So this run trained mostly on 445-token rows and **barely exercised sparse attention**.
It is a valid measurement of *that* finetune, not of the intended long-context sparse
one. Fixing it needs: not sharding a 700-row dataset across ranks, a lower `min_length`
for it (p50 is 11171, so 4096 keeps ~78% instead of ~50%), and cycling it for multiple
epochs.

## Templating — the highest-risk part

Rows already carry `<think>…</think>` inside `output`, and Qwen3's chat template does
**not** add those tags itself (verified against the installed tokenizer, with and
without `enable_thinking`). So the content is passed through verbatim; re-wrapping it
would produce nested `<think><think>` that appears nowhere in pretraining.

```
<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>
|------------------ masked (-100) ------------------|--- supervised ---|
```

Three decisions that `verify_data.py` asserts rather than assumes:

- **The boundary comes from the template, not a token search.** The prompt is
  re-rendered with `add_generation_prompt=True` and its length is the mask boundary, so
  a template change cannot silently shift it. The full rendering is also checked to
  literally start with the prompt rendering.
- **The trailing `\n` after `<|im_end|>` is trimmed.** Qwen3's template emits
  `<|im_end|>\n` as a *turn separator*; supervising that newline teaches the model to
  emit one more token after it has already stopped. The last supervised token is now the
  EOS a sampler stops on.
- **No truncation.** A reasoning trace cut mid-`<think>` teaches the model to abandon
  reasoning without concluding, which is worse than dropping the row. Over-long rows are
  skipped and counted.

## Why `min_length = 12288`

Sparsity only bites above `topk * block_kv` = 2048 tokens. This dataset's median row is
~3000 tokens and only 2.8% exceed 16K, so an unfiltered stream would spend most steps in
the regime where the sparse path *is* dense — the loss would look healthy and the
experiment would measure nothing.

The cost is a keep rate of ~3–6%, which the run log prints so it cannot pass unnoticed.
Set `--min-length 0` to train on the true distribution instead.

## Verified: 100 steps on 8× B200

```
100 steps in 41.2 min
loss: first-5 mean 1.0115 -> last-5 mean 0.7630 (DECREASING)
attention: 72 sparse / 0 dense calls per step (36 layers)
```

| metric | value |
|---|---|
| tokens/step | mean 162,560 (min 119K, max 223K) — 16.3M total |
| per-rank sequence | mean 20,320 tokens |
| step time | mean 24.7 s, median 25.0 s |
| throughput | mean 6,906 tok/s |
| peak memory | 89.5 GB/rank (of 178) |
| **sparse fraction** | **1.000 every step** — never fell back to dense |
| loss | 1.0115 → 0.8280 (step 50) → 0.7630 |

The loss decreases monotonically in trend, which is the gate the design sets for this
phase ("loss decreases", not "matches dense"). Two caveats on reading it: 100 steps of
`grad_accum=1` at lr 1e-5 is a *pipeline* verification, not a quality result, and
per-step loss is noisy because each step is 8 different documents — the first-5/last-5
means are the signal, not any single step.

The step time is dominated by `block_q=1`, which is the accurate-but-expensive end of the
tradeoff. `--block-q 64` would be several times faster per step; the root README's sweep
quantifies it.

## Budget conventions differ between train and serve — reconciled explicitly

This is the one thing that would quietly invalidate a train/serve comparison:

| | formula |
|---|---|
| **vortex_torch** | `selected = topk_val + reserved_bos + reserved_eos` — reservations **additive** |
| **vortex_train** | `Budget.topk` is the **total**, reservations included |

So `--topk 16 --reserve-bos 1 --reserve-local 1` trains on **18 blocks (1152 tokens)**,
and serving must use `topk_val=16, bos=1, eos=1` — *not* `topk_val=18`. `patch.install()`
takes `topk` in vortex_torch's convention and computes the total, so the two cannot drift.

On `eos`: vortex_torch's `reserved_eos` is the *last N blocks of the sequence*, which
during decode is the recent/local window. In training every query position has its own
"most recent" block, so the faithful analogue is `reserve_local`. The checkpoint records
`reserved_eos = max(reserve_local, reserve_eos)` for exactly this reason.

`train.py` writes `vortex_selection.json` beside the weights, and `evaluate.py` reads it
as the source of truth and **refuses to run on a mismatch**.

Two harness details `evaluate.py` drives the engine directly to control:
`examples/math/verify_algo.py` hardcodes `bos=1, eos=2` (19 blocks, not 18) and defaults
`vortex_layers_skip=[0]` (layer 0 dense, where training left every layer sparse).

## Design notes

**Attention is swapped via the registry, not a monkey-patch.** `patch.install()` adds a
`"vortex_sparse"` entry to `ALL_ATTENTION_FUNCTIONS`, and `Qwen3Attention.forward`
already routes through it — passing q/k/v post-RoPE and post-`q_norm` in `[B, H, T, D]`,
exactly the layout the kernels want. Nothing in transformers' modeling code is edited.

**Dense fallbacks are counted, not silent.** The sparse path is skipped when `tq != tkv`,
when the sequence is below the dense threshold, when an attention mask is present, or
when dropout is on. Each is tallied and printed (`sparse%` per step,
`dense_reason` at the end), because a "sparse" run that quietly fell back would look
fast for the wrong reason. The intended state is **100% sparse**, and the run log proves
it: 72 calls per step = 36 layers × (forward + backward recompute).

**FSDP2 rather than DDP.** DDP replicates 4B params, grads, and fp32 Adam state on every
rank; FSDP2 shards all three. Measured: peak 39 GB/rank on 8 GPUs against 99.7 GB
single-GPU. The freed memory goes to sequence length, which is the point.

**Batch size 1 per rank is a correctness choice, not a memory one.** A padded batch needs
the attention mask honoured, and the sparse kernels consume a *pattern* rather than a
mask — a masked batch would fall back to dense. Use `--grad-accum` for a larger effective
batch.

**Each rank streams a disjoint stride** (`i % world_size == rank`), applied before the
length filter so the split does not depend on how many rows a rank keeps. Sharding by
seed instead would overlap, and training one sequence on two ranks in a step silently
doubles its weight. `verify_data`-adjacent check: 48 examples across 8 ranks, 48
distinct.

## Files

| file | role |
|---|---|
| `data.py` | streaming dataset, chat templating, label masking, collation |
| `patch.py` | registers `vortex_sparse` in transformers' attention registry |
| `train.py` | the FSDP2 training loop |
| `verify_data.py` | CPU-only template/masking checks — run before training |

## A bug this application found

The first real run failed to compile at seqlen 12288: the select kernel holds scores as
a `[NTILES, TILE_N]` register tile and Triton requires power-of-two extents, but
`NTILES = ceil(n_kv / 64)` is 3 at that length. **71% of lengths in the 12K–40K range
were affected.** Every unit test had used shapes where `NTILES == 1`, so nothing caught
it — the kernel had never been run at a real training length.

Fixed by rounding `NTILES` up to a power of two; the extra tiles are entirely past
`num_kv_blocks`, so they are masked `-inf` and skipped by the causal guard. Verified at
12288 / 16448 / 20032 / 24576 / 33024 / 40960, with selection counts matching the closed
form exactly.
