# vortex_torch — project primer for Claude Code

`vortex_torch` is a JIT-compiled sparse-attention framework that plugs
into sglang's decode loop. Users / AI agents write a **sparse-attention
submission** — two files in `submissions/` — and the framework compiles
them into Triton kernels at runtime.

## Objective

> **Strike the best tradeoff between accuracy (`mean@16`) and
> decoding throughput on AIME24.**

There is **no fixed quality floor** and no single number to maximise.
Both `mean@16` and `throughput` are objectives — the goal is to push
the **Pareto frontier** outward. A variant that buys throughput at
a small `mean@16` cost is useful; a variant that buys accuracy at a
small throughput cost is useful too. Map the frontier across a
batch by varying *along* the tradeoff: accuracy-leaning knobs
(looser `vortex_topk_val`, fewer `vortex_layers_skip`, bf16 KV) on
some variants, throughput-leaning knobs (tighter `topk`, more layer
skips, fp8 `kv_cache_dtype`, `approxTopK(tolerate_ratio=…)`,
`mem_fraction_static → 0.9`) on others. Pick winners by where
they sit on the `(throughput, mean@16)` plane against the running
best in `memory.md §5`, not by clearing a fixed bar.

## Inventing beyond the literature

The `papers/` folder and [papers/guide.md](papers/guide.md) cover
what's already published — sinks, heavy hitters, channel sparsity,
low-rank K, LSH sampling, dual-band centroids. Treat them as
**seeds, not a menu.** A winning flow does not need a citation.

**Novelty bar.** Algorithmic innovation is the primary objective.
Every batch must reserve **at least one slot** (aim for two) for a
*genuinely novel* variant — an idea from §16.2 (untried knobs),
§16.3 (inversions), §16.4 (first-principles), or the framework's own
op set. Paper replicas and §16.1 combinations are catalog-adjacent
and do not fill the novelty slot.

**The remaining 2–3 slots should use `papers/guide.md` §16.5
techniques** — catalog-adjacent parameter sweeps (`vortex_topk_val`,
`approxTopK`, layer-skip patterns, fp8/bf16 KV, etc.) that are
explicitly encouraged for non-novelty slots. They map the Pareto
curve around the novel variant and give the measured context needed
to judge whether the novel idea is actually buying something.

## Where the instructions live

All authoritative content lives under [AI/](AI/). Read in order:

1. [AI/AGENTS.md](AI/AGENTS.md) — the full submission contract, rules,
   budget / BOS / layer-skip semantics, benchmark protocol.
2. [AI/tutorials/overview.md](AI/tutorials/overview.md) — 5-minute map.
3. [AI/tutorials/program_create_cache.md](AI/tutorials/program_create_cache.md)
4. [AI/tutorials/program_forward_cache.md](AI/tutorials/program_forward_cache.md)
5. [AI/tutorials/program_forward_indexer.md](AI/tutorials/program_forward_indexer.md)
6. [AI/tutorials/cache_op.md](AI/tutorials/cache_op.md) — indexer-side
   op math reference.
7. [AI/tutorials/indexer_op.md](AI/tutorials/indexer_op.md) — cache-side
   op math reference.
8. [papers/guide.md](papers/guide.md) — synthesis of the ten
   sparse-attention papers in `papers/`. §14 = catalog of
   known-good submission ideas; **§16 = prompts for inventing
   flows that no paper here explores.**

Framework-internal deep dives live in
[AI/developer_guides/](AI/developer_guides/) — needed only if you are
modifying the compiler itself, not when writing a submission.

## Hard constraints

- **No native torch ops** inside `create_cache` / `forward_cache` /
  `forward_indexer`. Every tensor goes through
  `vortex_torch.indexer.*` / `vortex_torch.cache.*` ops. `.view`,
  `.sum(dim=...)`, elementwise torch, etc. will not compile.
- **Each op instance is one call site.** `self.mul_a = Multiply()`
  and `self.mul_b = Multiply()` — do not share.
- **Do not declare `"k"` or `"v"`** in `create_cache`; they are
  auto-provided.
- **`forward_indexer` must end in `topK(score, o, ctx=ctx)` or
  `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`** — the
  score must be RAGGED `[S, 1, 1]`. `approxTopK` is a faster
  adaptive 8-bit radix variant; `tolerate_ratio ∈ [0.0, 1.0]`
  where `0.0` = exact, higher = cheaper but looser.
  *Trtllm-only alternative:* with
  `"vortex_attention_backend": "trtllm"` you can instead end in
  `Union()((bt_a, sl_a), (bt_b, sl_b), o, ctx=ctx)` fed by two
  `TopK(k=...)(score, ctx=ctx) → (block_tables, seqlens)` calls.
  `TopK` / `Union` assert at profile time if used under flashinfer
  (see `AI/tutorials/indexer_op.md §9b`).
- **Cache-side reductions support `dim ∈ {1, 2}` only.** Cross-block
  reductions (`dim=0`) belong on the indexer side.
- **If a field is read+written across steps via `Load`/`Save`, zero
  it with `CFill(0.0)` in `forward_cache`.**
- **If `forward_indexer` uses `Save(...)`, the engine JSON MUST set
  `"disable_radix_cache": true`** (default `false`). sglang's
  prefix-radix cache otherwise shares per-request Save'd state
  across requests with matching prompt prefixes, corrupting
  Save/Load values. `check_engine_config` rejects the violation.

## Environment — resolve a working env FIRST (don't assume `vortex_v1`)

Every python call in this project needs an interpreter where `import
vortex_torch` works (with sglang for serving). **Do not assume a specific env
exists** — the host may have a different conda env, a uv/venv, or docker, and
**GLM models need a newer transformers** (a separate env). The very first action
of any session is to **establish a working env yourself**:

```bash
python algorithm_scientist/detect_env.py    # probes conda/uv/venv/docker, recommends a run prefix
```

Adopt the recommended **run prefix** and use it on every python call this session
(robust in non-interactive subshells), e.g. `RUN="conda run -n vortex_v1 python"`
— or `".venv/bin/python"`, `"uv run python"`, a docker invocation, etc.:

```bash
RUN="<recommended prefix>"
$RUN -c "import sys, vortex_torch; print(sys.executable, vortex_torch.__version__)"
```

If no working env is found, **build one** — that's the **`/setup-env`** skill
(creates/repairs a conda/uv/venv/docker env from the repo specs, and a separate
`transformers>=5` env for GLM). The `conda activate vortex_v1` snippets in the
command docs below are the **common default** — substitute your detected prefix
when it differs. Confirm `import vortex_torch` succeeds before any GPU work.

## GPU usage — shared cluster, your own budget, TP-aware

Three facts govern every GPU launch:

1. **The cluster is shared.** Other people — *and other processes under your own
   username* — may be using GPUs at any time. Never assume GPU 0, never assume a
   fixed count, never assume a GPU you used last wave is still free.
2. **`free_gpus.sh` tells you what's free for anyone right now.**
   `algorithm_scientist/free_gpus.sh` prints the indices of GPUs with **no
   running compute process and `memory.used < 1024 MiB`** (override the MiB via
   `free_gpus.sh <mib>`), exit 1 = none = hard wait. This already excludes
   everyone else's jobs and your own other jobs.
3. **`--max-gpus` is *your* budget.** The user tells you the **maximum number of
   GPUs this agent may use at the same time** — across all your concurrent
   children, *not* counting anyone else's work (including other jobs under your
   username). Never exceed it. If unset, default to "all currently free".

**TP-awareness.** A single run uses **`tp_size` GPUs** (tensor parallelism):
`1` for small models, but large ones don't fit on one GPU (e.g. MiniMax-M2.7
229B ⇒ `tp_size=4`). Put `"tp_size": K` in the submission JSON; the runners
honour it, and pass `--tp K` to `run_submission.py`. Pin each run to **exactly
`K`** free GPUs via a comma-separated `CUDA_VISIBLE_DEVICES`.

The allocation rule, applied **fresh at the start of every wave** (the free set
shifts during preflight/RULER/runs):

```bash
TP=<tp_size>                 # GPUs per run
MAX_GPUS=<your budget>       # max GPUs this agent uses concurrently (user-provided)
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; exit 1; }
USABLE=( "${FREE_GPUS[@]:0:$MAX_GPUS}" )          # cap free GPUs to your budget
PARALLEL=$(( ${#USABLE[@]} / TP ))                 # runs in flight at once
# need < TP usable GPUs ⇒ wait; otherwise launch PARALLEL runs, TP GPUs each:
#   gpus=$(IFS=,; echo "${USABLE[*]:$((slot*TP)):$TP}"); CUDA_VISIBLE_DEVICES=$gpus ...
```

One free GPU is enough to make progress when `TP=1`; for `TP=K` you need at
least `K` free (and within budget) or you wait.

## Running the benchmark — policy

**Every batch is exactly 4 variants.** That fixed width is what
makes orthogonal-knob sweeps and Pareto-frontier mapping work.
Each variant uses **`TP` GPUs** (tensor parallelism), and you may
run at most **`floor(min(free, MAX_GPUS) / TP)`** variants at once;
the rest go in later waves:

- `TP=1`, `min(free, MAX_GPUS) >= 4` → all 4 variants in parallel,
  one GPU each.
- Otherwise → **waves** of `floor(min(free, MAX_GPUS) / TP)`. E.g.
  `TP=4`, budget 8, 8 free → 2 variants/wave → `2 + 2`. `TP=1`,
  budget 2 → `2 + 2`. The batch still produces 4 results; it just
  takes more waves.
- Fewer than `TP` usable GPUs (free ∩ budget) → **hard wait**.

Detect at the start of **every wave** (free set shifts; budget is yours):

```bash
TP=<tp_size>; MAX_GPUS=<your budget>; BATCH_SIZE=4
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
    echo "no free GPUs — wait, do not launch" >&2; exit 1
}
USABLE=( "${FREE_GPUS[@]:0:$MAX_GPUS}" )            # never exceed your budget
PARALLEL=$(( ${#USABLE[@]} / TP ))                  # variants in flight; 0 ⇒ wait
echo "free: ${FREE_GPUS[*]} | usable(≤$MAX_GPUS): ${USABLE[*]} | TP=$TP parallel=$PARALLEL"
```

`free_gpus.sh` excludes GPUs that have a compute process running
on them or memory.used ≥ 1024 MiB (override via
`free_gpus.sh <mib>`). Empty result (exit 1) ⇒ hard wait. You are
launchable whenever `free ∩ budget >= TP`; smaller values just
serialise into multiple
waves on the available GPUs.

**File layout.** All submissions you write live under
`submissions/<tag>/`, where `<tag>` is your agent identifier
(default: a sanitized lowercase form of your model name, e.g.
`claude_opus_4_7`). Within that dir, batched runs use the
convention `batch_<x>_id<y>.{py,json}` (`<x>` = batch index,
`<y>` = per-variant index in `0…3`). Single-variant runs are
debug-only. Each batch:

1. **4 orthogonal variants** —
   `submissions/<tag>/batch_<x>_id0.{py,json}` …
   `submissions/<tag>/batch_<x>_id3.{py,json}` — varying
   different knobs. The slot count is fixed at 4 regardless of
   how many GPUs are free; only the parallelism changes.
2. **Cheap local pre-flight first** for all 4 (CPU, no GPU):
   ```bash
   TAG=<your_agent_tag>; BATCH=<batch_index>
   for y in 0 1 2 3; do
     python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')"
   done
   ```
   Refuse to launch any variant whose pre-flight fails.
3. **RULER pre-filter — quick quality gate (≥ 0.85).** Before
   spending 20–60 minutes on AIME24, run `algorithm_scientist/run_ruler.py`
   on each variant. Any variant scoring below **0.85 accuracy** on
   `examples/ruler/validation.jsonl` has structurally broken attention —
   fix it (widen `vortex_topk_val`/`vortex_topk_ratio` or revise the
   indexer scoring), re-pre-flight, and re-run RULER until all 4 pass.
   Allocate `TP` GPUs per variant (RULER reads `tp_size` from the JSON), at
   most `floor(min(free, MAX_GPUS) / TP)` at once, re-detecting each wave:
   ```bash
   TP=<tp_size>; MAX_GPUS=<your budget>; y=0
   while [ "$y" -lt 4 ]; do
     FREE=($(algorithm_scientist/free_gpus.sh)) || { echo wait; sleep 60; continue; }
     USABLE=( "${FREE[@]:0:$MAX_GPUS}" ); PAR=$(( ${#USABLE[@]} / TP ))
     [ "$PAR" -lt 1 ] && { echo "need $TP GPUs — wait"; sleep 60; continue; }
     s=0; while [ "$s" -lt "$PAR" ] && [ "$y" -lt 4 ]; do
       g=$(IFS=,; echo "${USABLE[*]:$((s*TP)):$TP}")
       CUDA_VISIBLE_DEVICES=$g python algorithm_scientist/run_ruler.py \
         --config "submissions/${TAG}/batch_${BATCH}_id${y}.json" &
       s=$((s+1)); y=$((y+1)); done
     wait
   done
   ```
   Results land in `summary_ruler_submissions/<tag>/<stem>/latest.json`.
4. **Launch the 4 variants, `TP` GPUs each, ≤ `floor(min(free, MAX_GPUS)/TP)`
   at once**, re-detecting free GPUs each wave and `wait`-ing between waves.
   You decide a per-run **timeout** (`TIMEOUT_MIN`, ~1.5× your model+task
   estimate); the `timeout` wrapper enforces it. Pass `--tp $TP` and
   `--task <task>` (or `--data <jsonl>` for a non-default model):
   ```bash
   LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"
   TP=<tp_size>; MAX_GPUS=<your budget>; TIMEOUT_MIN=<your estimate>; y=0
   while [ "$y" -lt 4 ]; do
       FREE=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; sleep 60; continue; }
       USABLE=( "${FREE[@]:0:$MAX_GPUS}" ); PAR=$(( ${#USABLE[@]} / TP ))
       [ "$PAR" -lt 1 ] && { echo "need $TP GPUs — wait"; sleep 60; continue; }
       s=0
       while [ "$s" -lt "$PAR" ] && [ "$y" -lt 4 ]; do
           cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"; stem=$(basename "$cfg" .json)
           g=$(IFS=,; echo "${USABLE[*]:$((s*TP)):$TP}")
           CUDA_VISIBLE_DEVICES=$g timeout ${TIMEOUT_MIN}m \
               python algorithm_scientist/run_submission.py --tp $TP --task aime24 --config "$cfg" \
               > "$LOGDIR/gpu${g//,/_}_${stem}.out" 2> "$LOGDIR/gpu${g//,/_}_${stem}.err" &
           s=$((s+1)); y=$((y+1))
       done
       wait
   done
   ```
   Each variant gets a contiguous slice of `TP` usable GPUs; `MAX_GPUS`
   caps how many of the free GPUs you occupy at once (the rest belong to
   others / your other jobs). With `TP=1` and budget ≥ 4 this is one wave
   of 4; tighter `TP`/budget just means more waves. Each
   child writes its result into
   `summary_submissions/<tag>/<stem>/<timestamp>__<hash>.json`
   and updates `latest.json` on its own. The runner mirrors the
   config's path under `submissions/` into `summary_submissions/`,
   so `submissions/<tag>/batch_<x>_id<y>.json` becomes
   `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
   isolation, no collisions between agents that happen to use
   the same `batch_x_idy` stem.

5. **One batch at a time on the free GPUs.** Do not launch a
   second batch while the first (any wave) is still running, and
   do not try to "fill the gaps" by launching extra variants on
   GPUs another user freed mid-batch — both contend for memory
   and either OOM or thrash. Use `jobs` (or `ls -lt
   summary_submissions/<tag>/*/latest.json`) to see how many
   children are still alive while you wait.

## While you wait (runtime varies by model+task; the `TIMEOUT_MIN` you set enforces the cap)

Idle is not an option. Each polling cycle, do one of:

- **Read.** Priority: `AI/tutorials/` → `AI/developer_guides/` →
  `papers/` → `vortex_torch/flow/algorithms.py` →
  `vortex_torch/{indexer,cache}/*` → `csrc/`. After each file,
  append one insight to `algorithm_scientist/memory.md` §7.
- **Invent.** Open `papers/guide.md` §16 and pick a §16.2
  (untried knob), §16.3 (inversion), or §16.4 (first-principles)
  prompt — *not* §16.1 combinations, those are catalog-adjacent
  and don't fill the novelty slot. Better still: come up with
  a hypothesis derived from the framework's op set itself that
  doesn't fit any §16 sub-bucket. Sketch a one-sentence
  hypothesis + cache/indexer ops, naming the specific op or
  behaviour exploited. Aim for two such sketches per wait cycle.
- **Design (don't launch) the rest of the next batch.**
  Pre-flight the 4 candidates so they're ready to fire the
  moment `wait` returns. Concurrent batches would OOM the
  shared GPUs.
- **Analyse children early.** As individual `latest.json` files
  appear (children finish at slightly different times), pull
  their `mean@16` / `throughput` and start filling a §2
  sub-section in memory.md. Close the §1 row when all 4 are
  in, then update §3 (hypotheses) / §4 (anti-patterns) / §5
  (winners).

## Persistent state — `algorithm_scientist/memory.md`

The conversation evaporates; `memory.md` does not. Read it at the
start of every session and write to it before stopping. Any batch
submission and any batch completion must mutate it.
   When the job finishes, each run is written into a
   per-submission subfolder so iterations never collide:

   ```
   summary_submissions/<name>/
       <timestamp>__<content_hash>.json   # full summary + embedded .py/.json
       latest.json                        # symlink → newest run
       INDEX.jsonl                        # one-row-per-run index
   ```

   The content hash is `sha256(config.json || module.py)` truncated
   to 12 chars — same code → same hash → you can see re-runs
   at a glance. Read `summary_submissions/<name>/latest.json`
   after a run, and on failure read the per-child log under
   `logs/submission/batch_<TS>/gpu<i>_<stem>.{out,err}` (or the
   `logs/submission/single_<TS>/<stem>.{out,err}` produced by
   `/benchmark`).

## Kickoff prompt for new sessions

To boot a fresh Claude Code session straight into the long-horizon
iterate loop, paste the prompt block from
[algorithm_scientist/iterate_kickoff.md](algorithm_scientist/iterate_kickoff.md)
into the new session's first message. The agent will identify
its tag, bootstrap from this primer + AGENTS.md + tutorials +
papers/guide.md + memory.md, and start the first batch
autonomously. State lives in `algorithm_scientist/memory.md` and
the `submissions/<tag>/` / `summary_submissions/<tag>/` trees,
so any later session resumes cleanly from the same prompt.

## Slash commands available in this session

**Seven top-level commands** (the user-facing interface):

- `/innovate <N> [theme] [--model <hf-id>]` — draft N novel submissions in one
  shot for a chosen model; all must compile; may call `/add-ops` for a missing op.
- `/iterate [--model <hf-id>] [--task aime24|aime25|aime26|amc23|<file.jsonl>] [--max-iterations N]`
  — the autonomous design→preflight→RULER→run→analyse loop; model + task are inputs.
- `/add-ops [new|accelerate|fuse|investigate] <desc>` — full-stack op work
  (Python op **and** kernel). The shared sub-skill other commands call when they
  need an op vortex lacks or one that's too slow. Absorbs the old
  `iterate_topk` / `iterate_centroids_score` kernel loops.
- `/support-model <hf-id>` — check geometry→backend→boot→RULER; wire support if
  unsupported, then re-verify.
- `/debug <symptom>` — catch-all diagnostic for everything else.
- `/write-paper [--scope onepager|full] [--tag <tag>]` — compile a LaTeX paper
  from your results (tectonic `tex` env + `vortex_paper/` style).
- `/reproduce-paper <arxiv-id|path.pdf|papers/<name>>` — compose a paper's
  algorithm as a vortex submission; calls `/add-ops` when an op is missing.

**Research toolkit** (discover algorithms like a human — see
[AI/workflows/research_toolkit.md](../AI/workflows/research_toolkit.md)):

- `/research <question> [--model] [--task]` — the full loop: survey → analyze
  real attention → hypothesize → **offline recall screen** → RULER → iterate →
  journal. Ties the tools below together.
- `/research-ideas <topic> [--deep]` — web/literature survey → a brief that seeds
  `/innovate` (uses `WebSearch`/`WebFetch` + the `deep-research` skill).
- `/ablate <base.json> <knob> <v1,v2,…>` — controlled single-knob sweep →
  quality/efficiency/throughput curve (the isolation `/iterate` doesn't do).
- `/leaderboard [--task] [--tag]` — running Pareto frontier on
  (throughput, mean@16) across all runs.
- Offline tools under `algorithm_scientist/research/` (framework — write your own
  methods; included ones are reference baselines): `capture_trace.py` (real
  q/K/V via HF, no cudagraph, long-context on GPU), `analyze_attention.py`
  (sink/local/retrieval-head profile), `eval_recall.py` + `methods/`
  (**per-head top-k token recall@budget** vs centroid/quest/quest_hw/h2o/
  streaming/random), `efficiency.py` (roofline: **end-to-end** decode
  ceiling = (W + B·KV_full)/(W + B·(KV_sparse + KV_index)); separate weight vs KV
  dtype, MoE-aware active params, per-algorithm index cost — `exact` top-k
  collapses it; raise `--batch` to see KV dominate), `pareto.py` (frontier). Record in
  `research/journal.md`.

**Low-level helpers** (used standalone or called by the seven above):

- `/setup-env [--model]` — **run first**: detect/build a working env
  (conda/uv/venv/docker) where `import vortex_torch` works; returns the run
  prefix; handles the GLM transformers-5 split. Helper: `detect_env.py`.
- `/new-submission <name>` — scaffold a submission pair.
- `/preflight <name>` — cheap local `check_engine_config`.
- `/batch-benchmark <n1> <n2> <n3> <n4>` — the sanctioned 4-variant batch
  (parallel when `N >= 4`, else waves of `N`); task via `TASK_ARG`.
- `/review <name>` — audit a submission against AGENTS.md.
- `/benchmark <name>` — *debug only*: single-variant run.

Workflow guides live in [AI/workflows/](../AI/workflows/) (run_tasks, add_op,
support_model, paper). The task `prompt` is **tokenizer-bound** — a non-default
model needs its jsonl rebuilt with `examples/misc/make_task.py` before running.

## Subagents available

- `vortex-submission-writer` — drafts and iterates on a submission.
- `vortex-submission-reviewer` — audits a submission pair (read-only).
- `vortex-op-author` — full-stack op work (add/accelerate/fuse/investigate);
  the mechanism behind `/add-ops`, also called from `/innovate`, `/iterate`,
  `/reproduce-paper`.
- `vortex-paper-writer` — writes/compiles the LaTeX paper, or reads a paper to
  reproduce it.
- `vortex-math-researcher` — derives/verifies the linear-algebra & bounds behind
  a scoring rule or op; returns an implementable recipe.
- `vortex-kernel-expert` — high-performance Triton/CUDA kernels, using the
  **`third_party/flashinfer/`** and **`third_party/flash-attention/`** submodule
  source + KernelWiki/ncu skills.
- `vortex-research-critic` — adversarial rigor check (confounds, samples,
  baselines, calibration, overclaiming) before you trust a number or write it up.

Invoke via `Task(subagent_type="<name>", ...)` or the matching slash command.

> Note: `.claude/` is now **hand-maintained** (edit the files directly).
> `AI/generate_claude_folder.py` is the deprecated former generator — do not
> re-run it; it would clobber the current commands.
