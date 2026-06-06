# AGENT.md — Instructions for AI agents writing sparse attention

You are tasked with writing a new **sparse-attention flow** for
`vortex_torch`. The framework JIT-compiles your Python description
into Triton kernels and plugs them into sglang's decode loop.

Your deliverable is **two files placed under `submissions/<tag>/`**,
where `<tag>` is your agent identifier (see "Tag" below):

- `submissions/<tag>/<name>.py`   — a `vFlow` subclass.
- `submissions/<tag>/<name>.json` — an engine config that points at it.

For batched runs (the standard workflow — see §5c), the per-batch
naming convention is:

- `submissions/<tag>/batch_<x>_id<y>.py`
- `submissions/<tag>/batch_<x>_id<y>.json`

where `<x>` is the batch index (0-indexed, incrementing across
batches you launch this session) and `<y>` is the variant slot
within that batch (`0 … 3` — every batch is exactly 4 variants;
see §5c for parallelism rules).

### Tag — your agent identifier

`<tag>` is a sanitized lowercase string that identifies *you* (the
AI agent). Default to your model name with non-alphanumerics replaced
by `_`, e.g. `claude_opus_4_7`, `claude_sonnet_4_6`, `gpt_5`. Pick it
once at session start and reuse it for every submission you write
this session. The point is per-agent isolation: multiple agents
working in the same `submissions/` tree never collide on filenames,
and a `git log -- submissions/<tag>/` cleanly attributes work.

Example existing flows live at the top level (`submissions/example_*`)
and are not under any agent tag — those are framework reference
materials, not work product.

### Environment — activate the `vortex_v1` conda env first

Every python invocation in this contract (`check_engine_config`,
`run_submission_aime24.py`, the pre-flight loops in §5/§5c/§5f,
the iterate driver, etc.) expects the **`vortex_v1`** conda
environment. Activate it once at session start, before running
any bash snippet below:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vortex_v1
python -c "import sys; print(sys.executable)"   # expect .../envs/vortex_v1/...
```

If `conda activate` is unavailable in the current shell, fall
back to `conda run -n vortex_v1 python ...` per call. Either
way, the running interpreter must be the one inside `vortex_v1`
— a system / base / wrong-env python will fail to import
`vortex_torch`'s C extension and the framework's Triton kernels.

---

## Objective

> **Strike the best tradeoff between accuracy (`mean@16`) and
> decoding throughput (tokens/sec) on AIME24.**

There is **no fixed quality floor** and no single number to maximise.
Both `mean@16` and `throughput` are objectives, and the goal is to
push the **Pareto frontier** outward — find variants that *trade*
the two well, not variants that win one axis at any cost on the
other. A flow that buys +30% throughput for −2% `mean@16` is a
useful data point; a flow that buys +3% `mean@16` at −40%
throughput is too. Both belong on the frontier; comparing them is a
judgement call about where on the frontier you want to live, not a
hard pass/fail check.

When designing a batch, vary along axes that *change the tradeoff*
— accuracy-leaning knobs (looser `topk`, fewer `layers_skip`, bf16
KV) on some variants; throughput-leaning knobs (tighter `topk`,
more `layers_skip`, fp8 KV, `approxTopK`) on others — and let the
results map the frontier shape for the current flow family.

Concretely, each benchmark run gives you two headline numbers
(see §5b):

- `mean@16` — accuracy. Higher is better.
- `throughput` — speed (tokens/sec). Higher is better.

Pick winners by where they sit on the `(throughput, mean@16)`
Pareto frontier across the batch and against memory.md §5
(patterns that worked).

---

## 0. Read the tutorials first

Before writing any code, read the files in [`AI/tutorials/`](tutorials/)
**in this order**:

1. [`overview.md`](tutorials/overview.md) — 5-minute map of the whole
   framework and what you're being asked to write.
2. [`program_create_cache.md`](tutorials/program_create_cache.md) —
   how to declare the auxiliary cache fields your flow needs.
3. [`program_forward_cache.md`](tutorials/program_forward_cache.md) —
   how to write the per-block summary updater.
4. [`program_forward_indexer.md`](tutorials/program_forward_indexer.md) —
   how to write the per-decode-step page router.
5. [`cache_op.md`](tutorials/cache_op.md) — math reference for every
   op you can use in `forward_cache`.
6. [`indexer_op.md`](tutorials/indexer_op.md) — math reference for
   every op you can use in `forward_indexer`.

You do **not** need to read `developer_guides` — that's for framework
developers. All six files above stay entirely in the user's mental
model.

If a detail isn't covered in those six files, read the reference
implementations in [`../vortex_torch/flow/algorithms.py`](../vortex_torch/flow/algorithms.py).

---

## 1. The contract

You are writing a subclass of `vFlow` with exactly three methods:

```python
from vortex_torch.flow   import vFlow, register
from vortex_torch.indexer import Mean, GeMM, topK, ...   # whichever ops you need
from vortex_torch.cache   import Mean as CMean, ...

@register("<your_module_name>")
class YourFlow(vFlow):

    def __init__(self):
        super().__init__()
        # declare every op instance you'll use — each CALL SITE needs its own

    def create_cache(self, block_size, head_dim):
        return { "<name>": (D_0, D_1), ... }

    def forward_cache(self, cache, loc, ctx):
        # build summaries from cache["k"] / cache["v"] and write them
        # into the fields you declared above

    def forward_indexer(self, q, o, cache, ctx):
        # build a per-page score [S, 1, 1] and hand it to topK
```

The user's mental model (used throughout the tutorials):

- Pretend the system runs with `batch_size = 1`, one sequence.
- `q` has shape `[1, H_q, D]`.
- Every `cache["<name>"]` is `[S, D_0, D_1]` on the indexer side,
  `[1, D_0, D_1]` on the cache side.
- `cache["k"]` and `cache["v"]` are always there — **don't declare
  them** in `create_cache`.
- The framework replays your one-sequence program across the real
  batch and kv-head axes. You never write batching, paging, or
  Triton.

---

## 2. Hard rules — the framework will reject violations

1. **No native torch ops inside the three methods.** Every tensor
   (`q`, `o`, `cache[...]`, any intermediate) must go through
   `vortex_torch.indexer.*` or `vortex_torch.cache.*` ops. `.view`,
   `.contiguous`, elementwise torch, `.sum(dim=...)`, *etc.* will not
   compile.
2. **Each op instance is for one call site.** If `forward_indexer`
   needs two `Multiply` calls, declare `self.mul_a = Multiply()` and
   `self.mul_b = Multiply()` — don't share.
3. **Don't declare `"k"` or `"v"` in `create_cache`.** The framework
   hard-asserts against that and auto-provides both.
4. **`forward_indexer` must end in either
   `topK(score, o, ctx=ctx)` or
   `approxTopK(tolerate_ratio=…)(score, o, ctx=ctx)`** with
   `score.shape == [S, 1, 1]`. Fold any stray `H_q` / `D` axes with
   `Mean` / `Max` / `Sum` first. `approxTopK` is the throughput-
   oriented variant (adaptive 8-bit radix; `tolerate_ratio ∈
   [0.0, 1.0]`, `0.0` = exact, higher = cheaper-but-looser; output
   indices unsorted within each segment).
   *Trtllm-only multi-stream alternative:* if
   `"vortex_attention_backend": "trtllm"` is set in the JSON, the
   terminal op can instead be `Union()` fed by two
   `TopK(k=…)(score, ctx=ctx) → (block_tables, seqlens)` calls — see
   `AI/tutorials/indexer_op.md §9b`. `TopK` / `Union` will assert at
   profile time if used under flashinfer.
5. **Every declared cache field needs a writer and a reader.** A
   field nobody writes stays silently at stale bytes; a field nobody
   reads is wasted bandwidth.
6. **Reductions on the cache side only support `dim ∈ {1, 2}`.**
   Cross-block summaries (`dim=0`) belong on the indexer side.
7. **If the indexer accumulates into a field via `Load`/`Save`, the
   cache side must `CFill(0.0)` it at block completion** — otherwise
   the first `Load` on a freshly-allocated block reads uninitialised
   memory. See the running-average example in
   `program_forward_indexer.md §9` and `program_forward_cache.md §6b`.
8. **If the indexer uses `Save(...)`, the engine JSON must set
   `"disable_radix_cache": true`.** Without it, sglang's prefix-radix
   cache shares the Save'd per-request state across requests with
   matching prompt prefixes — silently corrupting Save/Load values
   across decode batches. Pre-flight (`check_engine_config`) rejects
   any submission that violates this rule. The default is `false`,
   so submissions that don't use `Save(...)` may omit the field.

---

## 3. Canonical skeleton

Copy-paste this into `submissions/<tag>/<name>.py` (or
`submissions/<tag>/batch_<x>_id<y>.py` for batched runs) and fill
it in. The `@register("<unique_module_name>")` string must be
**globally unique** across all submissions — pick something that
includes your tag and the file stem, e.g.
`@register("claude_opus_4_7_batch_3_id5_cls")`:

```python
import torch
from typing import Dict

from vortex_torch.flow    import vFlow, register
from vortex_torch.abs     import ContextBase
from vortex_torch.indexer import (
    topK, approxTopK, Mean, Max, Min, Sum, L2Norm,
    GeMM, Multiply, Add, Maximum, Minimum, Kron,
    Softmax, Normalize,
    Relu, Sigmoid, Silu, Add_Mul, Abs, Log, Exp,
    Save, Load, MaskSlice, Reshape,
    WhereEqual, WhereNotEqual, WhereGreater,
    WhereGreaterEqual, WhereLess, WhereLessEqual,
    TopK, Union,        # trtllm-only: TopK(k)+Union() multi-stream selection
)
from vortex_torch.cache import (
    Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm,
    MeanInterleave as CMeanInterleave, MaxInterleave as CMaxInterleave,
    MinInterleave as CMinInterleave, L2NormInterleave as CL2NormInterleave,
    GeMM as CGeMM,
    Multiply as CMultiply, Add as CAdd, Maximum as CMaximum, Minimum as CMinimum,
    Relu as CRelu, Sigmoid as CSigmoid, Silu as CSilu,
    Add_Mul as CAdd_Mul, Abs as CAbs, Log as CLog, Exp as CExp,
    Fill as CFill, MaskSlice as CMaskSlice, Reshape as CReshape,
)


@register("<your_module_name>_cls")
class YourFlowCls(vFlow):
    def __init__(self):
        super().__init__()
        # Indexer-side ops (one instance per call site)
        ...

        # Cache-side ops (one instance per call site)
        ...

    def create_cache(self, block_size: int, head_dim: int):
        return {
            # Example: "centroids": (1, head_dim),
            # Do NOT include "k" or "v".
        }

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        # For every field declared above that depends on cache["k"] or
        # cache["v"], compute and persist it here. If a field is
        # accumulated by forward_indexer via Load/Save, CFill(0.0) it
        # here at block completion.
        ...

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        # Build a per-page score of shape [S, 1, 1] and hand it to topK.
        ...
        self.output_func(score, o, ctx=ctx)
```

And the matching config:

```json
{
  "vortex_module_path":         "submissions/<tag>/<name>.py",
  "vortex_module_name":         "<unique_module_name>_cls",
  "vortex_block_size":          16,
  "vortex_workload_chunk_size": 32,
  "vortex_topk_val":            29,
  "vortex_topk_ratio":          0.0625,
  "vortex_block_reserved_bos":  1,
  "vortex_block_reserved_eos":  2,
  "vortex_layers_skip":         [0],
  "vortex_dtype":               "bfloat16",
  "kv_cache_dtype":             "auto",
  "mem_fraction_static":        0.8
}
```

`vortex_module_name` **must exactly match** the string passed to
`@register(...)` in your Python file.

**Every field in this JSON is a tuning knob you may change.** Pick
values that suit your flow. The ones that need special care:

| field | allowed / recommended values |
|---|---|
| `vortex_block_size` | positive power of 2 (e.g. 8, 16, 32). Smaller = finer sparsity granularity, larger = less cache-summary overhead. |
| `vortex_workload_chunk_size` | positive power of 2 (e.g. 16, 32, 64). |
| `vortex_topk_val` / `vortex_topk_ratio` | per-sequence sparse-page budget (see "Budget semantics" below). |
| `vortex_block_reserved_bos` / `_eos` | ints ≥ 1. Reserved first / last blocks that are always selected (see "BOS/EOS semantics" below). |
| `vortex_layers_skip` | list of layer ids that bypass sparse attention (run dense). See "Layer-skip patterns" below. |
| `vortex_dtype` | dtype for **intermediate** tensors. **Recommended: `"bfloat16"`.** Other accepted values: `"float16"`, `"float32"`, `"fp8_e5m2"`, `"fp8_e4m3"` — use only if you have a specific reason; bf16 is the tested default. |
| `kv_cache_dtype` | dtype for the **K/V cache itself**. Choose from: `"auto"` (resolves to bfloat16), `"fp8_e4m3"`, or `"fp8_e5m2"`. Using fp8 halves cache memory at the cost of numerical precision; bf16 via `"auto"` is the safe default. |
| `mem_fraction_static` | fraction of GPU memory sglang reserves for KV cache + model weights. Float. **Default 0.8** (lives in `get_engine`). Higher values usually raise throughput by enabling larger decode batches, but raise the risk of CUDA OOM mid-run. **Sweet spot 0.8-0.9.** Try `0.85` first; if it runs cleanly, push toward `0.9` / `0.95`. If you OOM, drop back by 0.05. |
| `disable_radix_cache` | bool. **Default `false`.** **REQUIRED `true` if your `forward_indexer` uses `Save(...)`** (i.e. persistent per-request state via `Save`/`Load`). sglang's prefix-radix cache otherwise reuses KV across requests sharing a prompt prefix — for normal flows that's free throughput, but for `Save`/`Load` flows it shares per-request state across requests and corrupts your Save'd values. Pre-flight rejects the violation. Don't set it for non-`Save` flows: leaving it `false` lets the prefix cache help you. |

### Budget semantics (`vortex_topk_val`, `vortex_topk_ratio`)

At runtime, the number of pages the attention kernel actually attends to
for each sequence is:

```
selected = min(
    num_blocks_in_seq,
    max(vortex_topk_val + vortex_block_reserved_bos + vortex_block_reserved_eos,
        vortex_topk_ratio * num_blocks_in_seq)
)
```

In plain English:

- The **static floor** is `topk_val + bos + eos`: a fixed minimum
  regardless of sequence length. You always get at least this many
  blocks.
- The **dynamic floor** is `topk_ratio * num_blocks_in_seq`: scales
  with the sequence. Use this to keep the sparsity *fraction* stable
  as sequences grow.
- The engine picks the larger of the two — you get both "at least
  `topk_val` blocks" *and* "at least this fraction of the sequence",
  whichever is more at any given moment.
- The final `min` clamps to the number of blocks actually in the
  sequence — there's no point selecting more blocks than exist.

So `topk_val` dominates on short sequences and `topk_ratio` dominates
on long ones. Typical starting values: `topk_val = 29-253`,
`topk_ratio = 0.0625` (1/16) to `0.25`.

### BOS/EOS semantics (`vortex_block_reserved_bos`, `vortex_block_reserved_eos`)

These are "always-on" blocks that the indexer is *not* allowed to
drop:

- `vortex_block_reserved_bos = N` → the **first N blocks** of every
  sequence are always in the selected set, no matter what score your
  `forward_indexer` produces for them. These blocks hold the system
  prompt / BOS tokens and are usually globally relevant.
- `vortex_block_reserved_eos = N` → the **last N blocks** of every
  sequence are always selected. These hold the most recent tokens,
  which typical generation kernels rely on.

The framework handles these two automatically — your
`forward_indexer` does **not** need to emit large scores for the
first / last blocks; the engine layers the BOS/EOS blocks on top of
your top-k selection afterwards. That's why the static floor in the
budget formula adds `bos + eos` to `topk_val`: the count reflects
"learned top-k" plus "fixed reservations".

Sensible values: `bos = 1-2`, `eos = 1-4`.

### Layer-skip patterns (`vortex_layers_skip`)

`vortex_layers_skip` is a list of layer ids that run **dense**
attention (no cache-side update, no page routing). Common patterns:

- **`[0]`** — skip only the first layer. Maximum sparsity, maximum
  speed; works for many models since layer 0 is often most sensitive
  to full context.
- **`[0, 4, 8, 12, ...]`** or **`[0, 8, 16, ...]`** — **interleaved**
  skip. Every k-th layer runs dense, the rest run sparse. This trades
  some throughput for better quality: the dense layers act as
  "periodic refreshers" of full-context information that downstream
  sparse layers can still benefit from. Use when a pure `[0]` skip
  loses too much accuracy on your task.
- **`[0, 1]`** — skip the first two layers. Moderate-sparsity baseline.

The stride and count are entirely your choice. More skipped layers =
more throughput cost but typically better quality. Start with `[0]`
and widen only if quality is unacceptable.

---

## 4. Reference flows to learn from

Six worked examples live in
[`../vortex_torch/flow/algorithms.py`](../vortex_torch/flow/algorithms.py).
Read them in increasing complexity:

| flow | what it demonstrates |
|---|---|
| `block_sparse_attention` | simplest; centroid per block, score by `q_mean · centroid` |
| `gqa_block_sparse_attention` | adds `Softmax` over pages, multi-head aggregation |
| `gqa_quest_sparse_attention` | QUEST envelope bound (max/min of keys per block) |
| `masked_quest_sparse_attention` | QUEST + feature-axis `MaskSlice` |
| `centered_block_sparse_attention` | demonstrates the `Reduce(dim=0)` mechanism — note: a constant per-sequence shift before `topK` is order-preserving, so this flow's picked set equals `block_sparse_attention`'s; useful pedagogically, not as a recipe |
| `running_avg_block_sparse` | persistent across-step state via `Save` / `Load` + `CFill(0.0)` |

### Submission examples — copy from any of these as a starting template

`submissions/` ships **complete working `(py, json)` pairs** at the
top level (the `<tag>/` subdirectories are agent-tagged work product
and should not be treated as references). The committed examples
cover different points in the design space:

| example | flavour |
|---|---|
| [`submissions/example_block_sparse_attention.{py,json}`](../submissions/example_block_sparse_attention.py) | the canonical minimal flow — single centroid per block, `q · centroid` score |
| [`submissions/gqa_quest_approx.{py,json}`](../submissions/gqa_quest_approx.py) | QUEST envelope (`CMin` + `CMax`), GQA-aware reduction, paired with `approxTopK` for cheaper selection |
| [`submissions/kimi_v0.{py,json}`](../submissions/kimi_v0.py) | tight throughput config (`topk_val=29`, `topk_ratio=0.0625`, `block_size=32`) over a Kimi-style flow |
| [`submissions/oai_v0.{py,json}`](../submissions/oai_v0.py) | wider-budget config (`topk_val=253`, `topk_ratio=0.25`) with aggressive `vortex_layers_skip=[0,4,8,12,16,20,24,28,32]` (every 4th layer dense, rest sparse) |

Pick the example that's *closest* to the flow you intend to write,
copy it into `submissions/<tag>/<name>.{py,json}`, then customise.
All four are tracked in git via the `!submissions/*.json` whitelist
in `.gitignore`; agent-tagged batches under `submissions/<tag>/` are
not.

---

## 5. Before you submit — verify compilation

After writing your flow, **run the pre-flight check**. It validates
the JSON, loads your module, downloads (or reads locally) the
HuggingFace `config.json` for the target model, and runs a compile
pass at the model's actual `(G, num_kv_heads, head_dim)`. It catches
most shape/dispatch mistakes before sglang ever touches a GPU:

```python
from vortex_torch.engine.sgl import check_engine_config
check_engine_config("submissions/<tag>/<name>.json")
```

If this returns without raising, your flow is ready for the engine.
If it raises, fix whatever error it reports and re-run. Common
failures and what they mean:

| error substring | what's wrong |
|---|---|
| `must not declare 'k' key` | you put `"k"` in `create_cache` |
| `must be a positive power of 2` | `vortex_block_size` or `vortex_workload_chunk_size` isn't a power of 2 |
| `does not contain @register` | `vortex_module_name` doesn't match the string in `@register(...)` |
| `failed to build vFlow` | typically a shape mismatch (or, for custom kernels like `Save` / `Load` / `topK` / `Softmax`, a missing `_impl_map` key) — read the traceback |
| `vortex_module_path ... not found` | file path in the JSON is wrong |
| `failed to fetch config.json from HuggingFace repo` | the model named by `model_path` isn't reachable; check the repo ID or pre-populate `~/.cache/huggingface/hub` |

---

## 5b. Run the benchmark and read the summary

Once `check_engine_config` passes, boot the engine and run the fixed
AIME24 protocol:

```bash
python algorithm_scientist/run_submission_aime24.py --config submissions/<tag>/<name>.json
```

Everything else is hard-coded inside the task runner
`algorithm_scientist/run_submission.py` (16 trials, 4096-token input cap,
**32768** max new tokens; task + dataset via `--task`/`--data`; model via the
JSON's `model_path`). **Tensor parallelism is configurable** — set
`"tp_size": K` in the JSON (or pass `--tp K`) and expose exactly `K` GPUs via a
comma-separated `CUDA_VISIBLE_DEVICES`; small models use `tp_size=1`, large ones
(e.g. MiniMax-M2.7 229B) need `K>1`. The launch loop allocates `tp_size` GPUs
per variant and caps concurrency to the agent's `--max-gpus` budget (see
`.claude/CLAUDE.md` "GPU usage"). The legacy `run_submission_aime24.py` /
`run_submission_amc23.py` are back-compat shims. The only thing you change
between runs is your flow's JSON (and, for a non-default model, the
tokenizer-bound task jsonl built with `examples/misc/make_task.py`).

The script prints a summary to stdout and writes it into a
**per-submission subfolder** under `summary_submissions/`:

```
summary_submissions/
└── <config_stem>/
    ├── 2026-04-24_13-22-05__0ca8893beb0b.json   # full summary + embedded .py/.json
    ├── 2026-04-24_14-07-51__17b9a4f2d310.json
    ├── latest.json                              # symlink → newest run
    └── INDEX.jsonl                              # one row per run, grep-able
```

The per-run filename encodes two things:

- `<timestamp>` — sortable, lexicographically newest = most recent.
- `<content_hash>` — first 12 hex chars of `sha256(config.json || module.py)`.
  Two runs of the **exact same code** share a hash (visible re-run);
  any edit produces a new hash. This is the easiest way to tell
  "is this the same submission I benchmarked yesterday?" without
  opening the file.

The JSON itself also embeds the full `.py` and `.json` contents at
run time under `submission_py` / `submission_json`, so any summary
is fully self-contained — you can reproduce a result from the
summary alone.

`INDEX.jsonl` is append-only and contains one compact row per run
(headline metrics + content hash). Use it to compare runs at a
glance:

```bash
cat summary_submissions/<name>/INDEX.jsonl | jq -c '{finished_at, content_hash, "mean@16", throughput}'
```

Example contents of one run JSON:

```json
{
    "mean@16": 0.09375,
    "pass@16": 0.3333333333333333,
    "total_example": 480,
    "e2e_time": 505.16787934303284,
    "total_tokens": 6042801,
    "throughput": 11961.966005951565,
    "args": { ... },
    "content_hash": "0ca8893beb0b",
    "finished_at": "2026-04-24_13-22-05",
    "submission_json": "{ ... full config.json text ... }",
    "submission_py":   "{ ... full module.py text ... }"
}
```

What each field means:

| field | meaning |
|---|---|
| `mean@16` | accuracy averaged across every single (question, trial) pair. The headline quality number — higher is better. |
| `pass@16` | per-question best-of-16. For each AIME24 question, take the max score across the 16 trials, then average across questions. Higher than `mean@16` when the model is good some fraction of the time but not always. |
| `total_example` | `trials × num_questions` = `16 × 30` = **480** for the fixed protocol. If this isn't 480, something's wrong with the dataset or trials count. |
| `e2e_time` | longest end-to-end latency across all 480 generations (seconds). Driven by the slowest / longest generation. |
| `total_tokens` | total completion tokens generated (summed across all 480). |
| `throughput` | `total_tokens / e2e_time` (tokens/sec). The headline *speed* number — higher is better. |
| `args` | echo of all runtime settings so a summary is self-describing. |
| `cuda_visible_devices` | value of `$CUDA_VISIBLE_DEVICES` at run time — useful when several submissions share a host, one per GPU. |
| `content_hash` | first 12 hex chars of `sha256(config.json \|\| module.py)`. Same code → same hash; any edit → new hash. |
| `finished_at` | `YYYY-MM-DD_HH-MM-SS` — local wall-clock time the run finished. |
| `submission_json` | full text of the config JSON at run time. |
| `submission_py` | full text of the module .py at run time. |

### What to iterate on

Both `mean@16` and `throughput` are objectives. Diagnose each run by
asking *where on the (throughput, mean@16) plane* it lands relative
to the rest of the batch and to the running best in memory.md §5:

- **High `throughput`, low `mean@16`** → too aggressive on sparsity
  for this flow family. Try the next variant with looser settings
  to map the accuracy recovery curve: *widen* `vortex_layers_skip`
  away from the most-aggressive end (e.g. `[0, 1]` instead of
  `[0, 4, 8, …]`), *raise* `vortex_topk_val` / `vortex_topk_ratio`,
  back off `kv_cache_dtype` from fp8 to auto. The result tells you
  the slope of the tradeoff, not whether to "fix" anything.
- **Low `throughput`, high `mean@16`** → the flow has accuracy
  headroom you can spend. Try the next variant with tighter
  sparsity to see how much accuracy you give up per unit of
  throughput: *shrink* `vortex_topk_val`, try fp8
  `kv_cache_dtype`, narrow `vortex_layers_skip`, swap `topK` for
  `approxTopK(tolerate_ratio=0.05–0.15)`.
- **High on both** → likely Pareto-optimal for this flow family;
  record in §5 (patterns that worked) and use as the baseline when
  designing the next batch.
- **Low on both** → the flow itself has a structural problem
  (likely the scoring function or a misused op). The decode-time
  cost has bought no signal. Tighten the indexer-side score logic
  before sweeping knobs.
- **`mean@16` nontrivial but `pass@16` much higher** → the flow is
  inconsistent but not broken. Noise, not a design bug; treat the
  current `(throughput, mean@16)` point as roughly fair and move on.
- **`total_example != 480`** → something is wrong with the run
  itself; investigate before trusting any other number. Usually a
  path error (`--config` pointed at the wrong file) or the engine
  crashed mid-run.

A submission is considered "good" when its `(throughput, mean@16)`
point sits on or pushes outward the running Pareto frontier in
memory.md §5 against the `example_block_sparse_attention` baseline.
When two variants are both Pareto-non-dominated relative to the
current best, both belong on the frontier — keep them as separate
data points rather than collapsing to one winner.

---

## 5c. Batched benchmarking — every batch is exactly 4 variants

The workflow **requires every batch to contain exactly 4
variants**. That fixed width is what guarantees analytical width
(orthogonal knob sweeps, Pareto-frontier mapping). What changes
with GPU availability is **parallelism**, not batch size:

- `N >= 4` free GPUs → all 4 variants run in parallel, one per
  GPU.
- `0 < N < 4` free GPUs → the 4 variants run in **waves of `N`**
  on the available GPUs (sequential fallback). With `N = 1` this
  is fully serial; `N = 2` runs `2 + 2`; `N = 3` runs `3 + 1`.
- `N == 0` (`free_gpus.sh` returns empty / exits 1) → hard wait;
  do not launch.

The host may share GPUs with other users or jobs. Detect free
GPUs at the start of every batch with the helper:

```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
    echo "no free GPUs — wait, do not launch" >&2
    exit 1
}
N=${#FREE_GPUS[@]}
BATCH_SIZE=4
PARALLEL=$N
[ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
echo "free GPUs: ${FREE_GPUS[*]}   (N=$N, parallel=$PARALLEL, batch=$BATCH_SIZE)"
```

`free_gpus.sh` excludes any GPU that has a running compute process
*or* has memory.used ≥ 1024 MiB (override threshold via
`free_gpus.sh <mib>`). It exits non-zero when nothing is free —
treat that as a hard "wait" signal, identical to the
"concurrency cap" rule. Any `N >= 1` is launchable; with
`N < 4` you simply pay extra wall-clock time for the additional
waves.

Single-variant runs are still not part of the protocol — every
batch must have 4 variants designed up front. If you only have
one core idea, fill the other 3 slots with orthogonal knob
sweeps (different `vortex_topk_val`, different `kv_cache_dtype`,
different `vortex_layers_skip`, different `vortex_block_size`,
etc.) — and reserve at least one slot for a genuinely novel
variant (see "Novelty budget" below).

### RULER pre-filter — quick quality gate (≥ 0.85)

Before spending 20–60 minutes on AIME24, run the fast RULER filter on
each variant using `algorithm_scientist/run_ruler.py`. Any variant
that scores below **0.85 accuracy** on `examples/ruler/validation.jsonl`
has structurally broken attention — the scoring function is dropping
so many critical tokens that AIME24 would yield no useful signal.
Fix or replace it before launching AIME24.

Run sequentially on one free GPU (RULER is fast — short outputs,
small dataset, finishes in minutes):

```bash
for y in 0 1 2 3; do
    cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
    CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
        python algorithm_scientist/run_ruler.py --config "$cfg"
done
```

Read the `accuracy` line from each run's stdout. A variant below
0.85 needs its `vortex_topk_val` / `vortex_topk_ratio` widened or
its indexer scoring function revised. Re-pre-flight and re-run RULER
until all 4 variants clear the bar, then proceed to the AIME24
launch. Results are also written to
`summary_ruler_submissions/<tag>/<stem>/latest.json` so you can
`jq .accuracy` them after the loop.

Launch the 4 variants in waves of `PARALLEL = min(N, 4)`, with
`wait` between waves so a wave's GPUs are free before the next
wave reuses them:

```bash
TAG=claude_opus_4_7                      # ← your agent identifier
BATCH=$(ls -d submissions/$TAG/batch_*_id0.json 2>/dev/null | wc -l)   # next batch index
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPUs"; exit 1; }
N=${#FREE_GPUS[@]}
BATCH_SIZE=4
PARALLEL=$N
[ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# Variants for this batch live at submissions/<tag>/batch_<x>_id<y>.{py,json}
# for y ∈ {0, 1, 2, 3}. The y index is the variant slot, NOT the GPU index;
# CUDA_VISIBLE_DEVICES is FREE_GPUS[(y - start)] within each wave.
for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
    end=$((start + PARALLEL))
    [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
    for y in $(seq $start $((end - 1))); do
        cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
        gpu="${FREE_GPUS[$((y - start))]}"
        stem=$(basename "$cfg" .json)
        CUDA_VISIBLE_DEVICES=$gpu \
            python algorithm_scientist/run_submission_aime24.py --config "$cfg" \
            > "$LOGDIR/gpu${gpu}_${stem}.out" \
            2> "$LOGDIR/gpu${gpu}_${stem}.err" &
    done
    wait
done
```

What this does:

1. Forks up to `PARALLEL = min(N, 4)` child processes per wave,
   each pinned to one of the free GPUs.
2. Each child runs `python algorithm_scientist/run_submission_aime24.py
   --config <cfg>` and writes its summary into
   `summary_submissions/<tag>/<stem>/<timestamp>__<hash>.json` (and
   updates `latest.json`) on its own. The runner mirrors the
   config's path under `submissions/` into `summary_submissions/`,
   so `submissions/<tag>/batch_<x>_id<y>.json` becomes
   `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
   isolation, no collisions across agents that pick the same
   `batch_x_idy` stem.
3. Per-child stdout/stderr land in
   `logs/submission/<tag>_batch_<x>_<TS>/gpu<i>_<stem>.{out,err}`.
4. The outer loop blocks on `wait` between waves; when `N >= 4`
   that's just one wave. When `N < 4`, additional waves run on
   the same GPU(s); the analysis step starts only after every
   wave finishes.

The runner (`run_submission_aime24.py`) automatically records
`cuda_visible_devices` into each summary JSON, so a quick
`cat summary_submissions/<tag>/*/INDEX.jsonl | jq -c` tells you
which variant landed on which GPU.

### Concurrency cap — one batch at a time on the free GPUs

A batch occupies the GPUs `free_gpus.sh` returned for as long as
its waves are running. **Do not launch a second batch while the
first is still running** (any wave still alive), and **do not
try to "fill the gaps" by launching extra variants on GPUs
another user freed mid-batch** — both contend for GPU memory and
either OOM or thrash. The natural cadence is: detect free GPUs,
launch the 4-variant batch, `wait` for all waves, read the 4
summary JSONs, then re-detect (the free set may have changed)
and launch the next batch.

If `free_gpus.sh` returns an empty set (exit code 1), do not
launch — wait, and use the time on the §5d activities.

### Novelty budget — at least one *genuinely novel* variant per batch

Algorithmic innovation is the primary objective. Of the 4 variants
in every batch, **reserve at least one slot for a genuinely novel
flow** — an idea that:

- does not trace to any single paper in `papers/`,
- is not just a combination of two papers (those are
  catalog-adjacent, see [papers/guide.md §16.1](../papers/guide.md)),
- is not a parameter sweep, a paper replica, or a mechanical
  knob-flip,
- comes from somewhere new: an inversion of a basic assumption
  (§16.3), an untried-knob experiment derived from staring at the
  framework's op set (§16.2), a first-principles answer to a
  question the literature isn't asking (§16.4), or — best of all —
  a hypothesis you simply haven't seen anywhere.

Aim for **two genuinely novel variants per batch** — one is the
floor, not the goal.

**The remaining slots (2–3 per batch) should be filled with
[papers/guide.md §16.5](../papers/guide.md) techniques** — the
catalog-adjacent parameter sweeps and orthogonal-knob variants that
are explicitly *not* novel but are still valuable for mapping the
Pareto frontier: different `vortex_topk_val` / `vortex_topk_ratio`,
`approxTopK` vs `topK`, layer-skip patterns, fp8 vs bf16 KV,
`mem_fraction_static` tuning. These sweeps give the measured curve
that tells you whether the novel idea in the off-catalog slot is
actually buying something. Agents are explicitly encouraged to use
§16.5 techniques for the non-novelty slots.

What "defending it in one sentence" means: the sentence should
name the specific framework op or behaviour the idea exploits, not
"combine X from paper A with Y from paper B". E.g. "use Save/Load
to track an EMA of attention magnitudes per page so the score
threshold drifts with sequence length" is novel; "Prism centroid
+ Keyformer accumulator" is a combination.

Record the novelty hypothesis in one sentence in
`algorithm_scientist/memory.md` §3 the moment the batch is launched,
so the verdict (after `wait`) lands on a pre-registered prediction.

---

## 5d. The "while-you-wait" protocol

A single batch's wall-clock time **depends on the model and task**
(≈ questions × 16 trials × output tokens ÷ throughput; bigger models, MLA,
and harder/longer tasks take much longer — and the 32768 max-new-tokens
budget raises it). **You decide a per-run timeout** (`TIMEOUT_MIN`, ~1.5× your
estimate) and the launch loop **enforces it** by wrapping each child in
`timeout ${TIMEOUT_MIN}m`; a child that exits 124 / leaves no `latest.json`
timed out — log it in §4 and treat that variant as failed. While the children are
running, do one of the following on each polling cycle (e.g. `jobs`
to see how many are still alive, or `ls -lt
summary_submissions/<tag>/<stem>/latest.json` to see which children
have finished):

1. **Deepen understanding.** In priority order:
   `AI/tutorials/` → `AI/developer_guides/` →
   `vortex_torch/flow/algorithms.py` →
   `vortex_torch/{indexer,cache}/*` → `csrc/`.
   After each file, append one bullet to
   [`algorithm_scientist/memory.md`](../algorithm_scientist/memory.md)
   §7 *Reading log* with the single most useful insight.
2. **Design the next batch.** Sketch the next 4 orthogonal
   variants (different theme — not 4 more copies of the running
   theme) so they're ready to launch the moment the current batch
   finishes `wait`. Don't actually launch — concurrent batches
   will OOM the shared GPUs.
3. **Invent.** Open [papers/guide.md §16](../papers/guide.md) and
   pick a §16.2 (untried knob), §16.3 (inversion), or §16.4
   (first-principles) prompt — *not* a §16.1 combination, those
   don't count for the novelty slot (see §5c "Novelty budget").
   Better still: come up with a hypothesis that doesn't fit any
   §16 sub-bucket — something derived from the framework's op set
   itself. Sketch a one-sentence hypothesis and the cache +
   indexer ops it would need. Inventing while a batch runs is
   free time; inventing while staring at a launched batch's row in
   §1 is wasted time. Aim for two novel sketches per wait cycle
   when you can. (For pure ideation without any benchmark, see
   §5f *Innovation-draft mode*.)
4. **Analyse completed children early.** Each child writes its
   `summary_submissions/<tag>/<stem>/latest.json` as soon as it finishes,
   without waiting for the others. As children land, read those
   summaries and start filling a §2 sub-section in memory.md
   (results table + 1-3 sentence takeaway). When all 4 are in,
   close the §1 row and update §3 (hypotheses) / §4 (anti-patterns)
   / §5 (winners).

---

## 5e. memory.md — your persistent notebook

[`algorithm_scientist/memory.md`](../algorithm_scientist/memory.md)
is the single source of truth for what's running, what's done, and
what you've learned. It survives across sessions; conversation
context does not. Sections:

| § | What goes here |
|---|---|
| §1 In-flight batches | one row per active batch (at most 1 — every local GPU is fully consumed) |
| §2 Completed batches | results table + takeaway per batch, oldest first |
| §3 Design hypotheses | open hypotheses + accumulating evidence + verdict |
| §4 Anti-patterns | one-liners of things that didn't work, so future-you doesn't retry |
| §5 Patterns that worked | confirmed winners worth carrying forward |
| §6 Open questions | the agent's own backlog — pick from the top when a slot opens |
| §7 Reading log | timestamped notes from tutorials / dev guides / source |
| §8 Session notes | per-session freeform append-only summary |

**Read at start, update before stop.** Every batch submission and
every batch completion mutates §1 and §2.

---

## 5f. Innovation-draft mode (no benchmark, no iterate loop)

The default workflow described in §5b-§5e is *iterative*:
batch → benchmark → analyse → next batch, with `memory.md` as the
persistent notebook. There is also a complementary one-shot mode
for **algorithmic exploration only**, exposed as `/innovate <N>
[theme-hint]`.

`/innovate` produces **N novel submission pairs** in a single
shot and then returns control to the user. The mode is defined by
two non-negotiable contracts:

1. **Genuinely novel algorithm.** Every variant must draw from
   [papers/guide.md](../papers/guide.md) §16.2 (untried knobs),
   §16.3 (inversions), §16.4 (first-principles), or — best — a
   hypothesis derived from the framework's op set itself that
   does not fit any §16 sub-bucket. Paper replicas, §16.1
   combinations of two papers, and pure parameter sweeps over an
   existing flow are **disqualified** here (stricter than the
   iterate-mode novelty budget, which only requires *one* novel
   slot per batch).
2. **Compiles.** Every variant must pass `check_engine_config`
   locally. A novel idea that does not compile is not an output
   of this mode — fix it or surface the residual error and stop.

The output layout is intentionally separated from the iterate
tree so the two modes never collide on filenames within a tag:

```
submissions/<tag>/innovate_<x>_id<y>.py
submissions/<tag>/innovate_<x>_id<y>.json
```

for `y ∈ {0 … N-1}`, where `<x>` = number of existing
`submissions/<tag>/innovate_*_id0.json` files. The `<y>` slot
range is `0 … N-1` (set by the user's `N`), **not** `0 … 3` —
batch sizing rules from §5c do not apply to this mode.

What the mode is **not**:

- **Not a benchmark.** No GPU is touched; no
  `run_submission_aime24.py` invocation; no `latest.json` written.
- **Not iterative.** One shot, then return. No "wait + analyse +
  next batch" loop.
- **Not bound to batch=4.** `N` is whatever the user passes
  (typically 2-12 depending on the theme's surface area).
- **Not memory.md-aware.** §5e's notebook is left untouched;
  there are no §1/§2 ledger mutations and no §3/§4/§5
  hypothesis/anti-pattern/winner updates.

When the user later wants to evaluate one of these drafts, they
copy or rename the relevant `innovate_<x>_id<y>.{py,json}` pair
into a `batch_<x>_id<y>.{py,json}` slot themselves, then run
`/batch-benchmark` (groups of 4) or `/iterate`. `/innovate` does
**not** automate that hand-off.

Use `/innovate` when the goal is pure ideation (e.g. "give me 6
fundamentally different ways to use Save/Load across decode
steps") and `/iterate` (with §5c) when the goal is to actually
move the Pareto frontier with measured numbers.

---

## 6. Workflow checklist

- [ ] Read the six tutorials in `AI/tutorials/` in order.
- [ ] Decide what per-block summaries your flow needs (this becomes
      `create_cache`).
- [ ] Write `forward_cache` — compute each summary from
      `cache["k"]` / `cache["v"]`; `CFill(0.0)` any field that
      `forward_indexer` will accumulate into.
- [ ] Write `forward_indexer` — build a `[S, 1, 1]` score and pass it
      to `topK`.
- [ ] Save the flow to `submissions/<tag>/<name>.py` with a unique
      `@register(...)` name.
- [ ] Save the config to `submissions/<tag>/<name>.json`.
- [ ] Run `check_engine_config("submissions/<tag>/<name>.json")`. Fix
      errors and re-run until it passes.
- [ ] Run the RULER pre-filter:
      `python algorithm_scientist/run_ruler.py --config submissions/<tag>/<name>.json`.
      Require `accuracy >= 0.85`. If below, widen `vortex_topk_val`/`vortex_topk_ratio`
      or fix the indexer, re-pre-flight, and re-run until it passes.
- [ ] Run the benchmark:
      `python algorithm_scientist/run_submission_aime24.py --config submissions/<tag>/<name>.json`.
      Read the `mean@16` / `pass@16` / `throughput` fields of the saved
      summary JSON.
- [ ] Plot the run's `(throughput, mean@16)` point against the
      running Pareto frontier in `memory.md §5`. If it dominates an
      existing entry, replace it; if it's dominated, log it as a
      data point in `§2` and use the diagnostics in `§5b` to pick
      which axis to push on the next variant.
- [ ] Done — sglang can now boot your flow by pointing at the JSON.

---

## 7. Absolute-minimum rules (if you remember nothing else)

1. Two files: `submissions/<tag>/<name>.py` and `submissions/<tag>/<name>.json`.
2. `@register("<name>")` in the Python file, same `<name>` in the JSON's
   `vortex_module_name`.
3. `create_cache` returns `{name: (D_0, D_1)}`. Never include `"k"` or
   `"v"`.
4. Every op — cache side or indexer side — is from `vortex_torch.cache`
   or `vortex_torch.indexer`. No native torch ops.
5. `forward_indexer` ends in `topK(score, o, ctx=ctx)` *or*
   `approxTopK(tolerate_ratio=…)(score, o, ctx=ctx)` with
   `score.shape == [S, 1, 1]`.
6. If any cache field is read+written across decode steps via
   `Load`/`Save`, zero-initialise it in `forward_cache` with
   `CFill(0.0)`, AND set `"disable_radix_cache": true` in the JSON.
7. Run `check_engine_config(...)` before declaring done.
8. Run `algorithm_scientist/run_ruler.py --config <your JSON>` and confirm
   `accuracy >= 0.85`. Below 0.85 means attention is structurally broken —
   widen knobs or fix the scorer before running AIME24.
9. Run `algorithm_scientist/run_submission_aime24.py --config <your JSON>` to get
   the `mean@16` / `pass@16` / `throughput` numbers.
10. **Objective: strike the best tradeoff between `mean@16` and
    `throughput`.** Both are objectives — push the
    `(throughput, mean@16)` Pareto frontier outward; there is no
    fixed quality floor.
