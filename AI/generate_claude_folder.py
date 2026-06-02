"""DEPRECATED — do not run. ``.claude/`` is now hand-maintained.

As of the June 2026 refactor the ``.claude/`` tree (CLAUDE.md, agents/,
commands/) is edited **directly** and is the source of truth: seven top-level
commands (innovate, iterate, add-ops, support-model, debug, write-paper,
reproduce-paper) plus the low-level helpers, and the vortex-op-author /
vortex-paper-writer subagents. Re-running this script would **clobber** the
current hand-written commands with the old generated set. It is kept only for
historical reference. Edit ``.claude/`` files instead.

----------------------------------------------------------------------
Generate a ``.claude/`` folder at the repo root so the vortex_torch
submission workflow drops straight into Claude Code.

The authoritative instructions live in ``AI/AGENTS.md`` and
``AI/tutorials/``. This script produces a thin set of wrappers
(``CLAUDE.md``, subagents, slash commands) that make those
instructions reachable from within an interactive Claude Code
session — without duplicating their content.

Layout produced:

    .claude/
        CLAUDE.md                        — project primer (always loaded)
        agents/
            vortex-submission-writer.md  — subagent: write / iterate a flow
            vortex-submission-reviewer.md — subagent: audit a flow
        commands/
            new-submission.md            — /new-submission <name>
            preflight.md                 — /preflight <name>
            benchmark.md                 — /benchmark <name>
            review.md                    — /review <name>
            iterate.md                   — /iterate <name>
            innovate.md                  — /innovate <N> [theme-hint]

Re-running the script overwrites the generated files. Hand-edit only
the inputs below; never edit the emitted files directly (they will be
clobbered on the next run).

Usage::

    python AI/generate_claude_folder.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_DIR = REPO_ROOT / ".claude"


# ---------------------------------------------------------------------------
# CLAUDE.md — project primer (always loaded into every Claude Code session)
# ---------------------------------------------------------------------------

CLAUDE_MD = dedent("""\
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

    ## Environment — activate the `vortex_v1` conda env first

    Every python invocation in this project (`check_engine_config`,
    `run_submission_aime24.py`, the pre-flight loops in the slash
    commands, etc.) expects the **`vortex_v1`** conda environment.
    **Activate it once at session start** before running any of the
    bash snippets below:

    ```bash
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vortex_v1
    python -c "import sys; print(sys.executable)"   # expect a path under .../envs/vortex_v1/
    ```

    If `conda activate` isn't available in the current shell (e.g. a
    non-interactive sub-shell that didn't source the conda profile),
    fall back to `conda run -n vortex_v1 python ...` for every
    python call. Either form is acceptable; what matters is that the
    running interpreter is the one inside `vortex_v1`.

    ## Running the benchmark — policy

    **Every batch is exactly 4 variants.** That fixed width is what
    makes orthogonal-knob sweeps and Pareto-frontier mapping work.
    Parallelism depends on how many GPUs are free *now* — the host
    may share GPUs with other users:

    - `N >= 4` free GPUs → run all 4 variants in parallel, one per GPU.
    - `0 < N < 4` free GPUs → run the 4 variants in **waves of N**
      on the available GPUs (sequential fallback). With `N = 1` this
      is fully serial; `N = 2` runs `2 + 2`; `N = 3` runs `3 + 1`.
      The batch still produces 4 results; it just takes longer.
    - `N == 0` (`free_gpus.sh` returns empty / exits 1) → **hard wait**,
      do not launch.

    Detect free GPUs at the start of every batch:

    ```bash
    FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
        echo "no free GPUs — wait, do not launch" >&2; exit 1
    }
    N=${#FREE_GPUS[@]}
    BATCH_SIZE=4
    PARALLEL=$N
    [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
    echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL, batch=$BATCH_SIZE)"
    ```

    `free_gpus.sh` excludes GPUs that have a compute process running
    on them or memory.used ≥ 1024 MiB (override via
    `free_gpus.sh <mib>`). Empty result (exit 1) ⇒ hard wait. Any
    `N >= 1` is launchable; `N < 4` just serialises into multiple
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
       `examples/validation.jsonl` has structurally broken attention —
       fix it (widen `vortex_topk_val`/`vortex_topk_ratio` or revise the
       indexer scoring), re-pre-flight, and re-run RULER until all 4 pass.
       ```bash
       for y in 0 1 2 3; do
         CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \\
           python algorithm_scientist/run_ruler.py \\
             --config "submissions/${TAG}/batch_${BATCH}_id${y}.json"
       done
       ```
       Results land in `summary_ruler_submissions/<tag>/<stem>/latest.json`.
    4. **Launch the 4 variants in waves of `PARALLEL = min(N, 4)`**,
       each child pinned via `CUDA_VISIBLE_DEVICES`, with `wait`
       between waves so a wave's GPU is free before the next one
       reuses it:
       ```bash
       LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
       mkdir -p "$LOGDIR"
       BATCH_SIZE=4
       PARALLEL=$N
       [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
       for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
           end=$((start + PARALLEL))
           [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
           for y in $(seq $start $((end - 1))); do
               cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
               gpu="${FREE_GPUS[$((y - start))]}"
               stem=$(basename "$cfg" .json)
               CUDA_VISIBLE_DEVICES=$gpu \\
                   python algorithm_scientist/run_submission_aime24.py --config "$cfg" \\
                   > "$LOGDIR/gpu${gpu}_${stem}.out" \\
                   2> "$LOGDIR/gpu${gpu}_${stem}.err" &
           done
           wait
       done
       ```
       The id `<y>` is the variant slot (0…3), NOT a GPU index —
       the actual GPU is `FREE_GPUS[$((y - start))]` within each wave.
       When `N >= 4` there is exactly one wave of 4 (fully parallel,
       identical to the old behaviour). When `N < 4` the loop runs
       ⌈4/N⌉ waves; the wall-clock cost scales accordingly. Each
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

    ## While you wait (20–60 min per batch; kill any child > 60 min)

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

    - `/new-submission <name>` — scaffold a new submission pair.
    - `/preflight <name>`      — run the cheap local pre-flight.
    - `/batch-benchmark <n1> <n2> <n3> <n4>` — launch the 4-variant batch on the currently-free GPUs (parallel when `N >= 4`, otherwise waves of `N`; the only sanctioned benchmark command).
    - `/review <name>`         — audit a submission against AGENTS.md rules.
    - `/iterate <name>`        — kick off a full auto-iteration loop (4 variants per batch on the currently-free GPUs, one batch at a time, updates memory.md).
    - `/innovate <N> [theme]`  — *innovation-draft mode*: produce N genuinely-novel submissions in one shot, all required to compile, no benchmark loop, no memory.md mutation. See AGENTS.md §5f.
    - `/benchmark <name>`      — *debug only*: run a single variant directly. Do not use in normal workflow.

    ## Subagents available

    - `vortex-submission-writer` — drafts and iterates on a submission.
    - `vortex-submission-reviewer` — audits a submission pair for
      rule violations without editing anything.

    Use `Task(subagent_type="vortex-submission-writer", ...)` from the main
    agent or invoke via slash command.
""")


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------

SUBMISSION_WRITER = dedent("""\
    ---
    name: vortex-submission-writer
    description: >-
      Use this subagent to design, write, and iterate on a vortex_torch
      sparse-attention submission. The agent reads AI/AGENTS.md +
      AI/tutorials/, writes the submission pair in submissions/, runs
      the local pre-flight, and (when asked) launches the AIME24
      benchmark directly via python — 4 variants per batch, run in
      parallel when at least 4 GPUs are free, otherwise in sequential
      waves on the available GPUs. Invoke whenever the user asks to
      "write a new submission", "try a sparse-attention idea", or
      "iterate on this flow".
    tools: Read, Write, Edit, Bash, Grep, Glob
    ---

    You are an expert at writing sparse-attention submissions for the
    `vortex_torch` framework. Your deliverable is always a pair of files
    placed under your **agent tag** subfolder of `submissions/`:

    - `submissions/<tag>/<name>.py`   — a `vFlow` subclass, with a
      single `@register("<unique>_cls")` decorator (the register
      name must be globally unique — include `<tag>` and the file
      stem).
    - `submissions/<tag>/<name>.json` — the engine config pointing
      at that .py via `vortex_module_path`.

    For batched runs (the standard `/batch-benchmark` and `/iterate`
    workflow), `<name>` follows the convention `batch_<x>_id<y>`
    where `<x>` is the batch index (0-indexed, increments with each
    batch you launch this session) and `<y>` is the variant slot
    (`0 … 3` — every batch is exactly 4 variants). Parallelism
    across free GPUs is `min(N, 4)`; the actual GPU each variant
    pins to is `FREE_GPUS[$((y - start))]` within its wave.

    ### First action of every session — pick your tag

    Before doing anything else, set your `<tag>` once and keep it for
    the whole session. Default to a sanitized lowercase form of your
    model name (e.g. `claude_opus_4_7`, `claude_sonnet_4_6`,
    `claude_haiku_4_5`, `gpt_5`). If `submissions/<tag>/` does not
    yet exist, create it; otherwise resume into it. Confirm the tag
    with the user only if you cannot determine your model name.

    ### Second action — activate the `vortex_v1` conda env

    Every python call in this workflow must run inside the
    **`vortex_v1`** conda env. Activate once at session start:

    ```bash
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vortex_v1
    python -c "import sys; print(sys.executable)"   # must be .../envs/vortex_v1/...
    ```

    If `conda activate` isn't usable in the current shell, prefix
    each python invocation with `conda run -n vortex_v1` instead.
    A wrong-env python will fail to import the framework's C
    extension and every pre-flight / benchmark call below will error.

    ## Read these before writing code

    Every time you're invoked for a fresh task, read in this order (unless
    already loaded this session):

    1. [AI/AGENTS.md](AI/AGENTS.md)
    2. [AI/tutorials/overview.md](AI/tutorials/overview.md)
    3. [AI/tutorials/program_create_cache.md](AI/tutorials/program_create_cache.md)
    4. [AI/tutorials/program_forward_cache.md](AI/tutorials/program_forward_cache.md)
    5. [AI/tutorials/program_forward_indexer.md](AI/tutorials/program_forward_indexer.md)
    6. [AI/tutorials/cache_op.md](AI/tutorials/cache_op.md)
    7. [AI/tutorials/indexer_op.md](AI/tutorials/indexer_op.md)
    8. [vortex_torch/flow/algorithms.py](vortex_torch/flow/algorithms.py) — six
       reference flows; your best source of pattern examples.
    9. [papers/guide.md](papers/guide.md) — synthesis of the ten
       sparse-attention papers in `papers/`. §14 is the catalog of
       known-good submission ideas; **§16 is the prompt for inventing
       flows that no paper here explores.** You're expected to use
       both — every batch reserves at least one slot for an
       off-catalog variant.

    ## Objective

    **Strike the best tradeoff between AIME24 `mean@16` and
    `throughput`.** Both are objectives — there is no fixed quality
    floor and no single number to maximise. The goal is to push the
    `(throughput, mean@16)` Pareto frontier outward across the
    batch and against the running best in `memory.md §5`. Vary
    along the tradeoff inside each batch: accuracy-leaning knobs
    on some variants (looser `vortex_topk_val` /
    `vortex_topk_ratio`, fewer `vortex_layers_skip`, bf16 KV),
    throughput-leaning knobs on others (tighter `topk`, more layer
    skips, fp8 `kv_cache_dtype`, `mem_fraction_static → 0.9`,
    `approxTopK(tolerate_ratio=…)` instead of `topK`). When two
    variants both push the frontier outward, both belong on it —
    record both in §5 rather than collapsing to one winner.

    ## Hard rules (AGENTS.md §2 — the framework will reject violations)

    1. No native torch ops anywhere in the three methods.
    2. Each op instance is for one call site — never shared.
    3. Never declare `"k"` or `"v"` in `create_cache`.
    4. `forward_indexer` must end in `topK(score, o, ctx=ctx)` *or*
       `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)` —
       `score.shape == [S, 1, 1]`. `approxTopK` is the throughput-
       oriented variant (adaptive 8-bit radix; `tolerate_ratio ∈
       [0.0, 1.0]`, `0.0` = exact).
    5. Every declared cache field must have both a writer and a reader.
    6. Cache-side reductions support `dim ∈ {1, 2}` only.
    7. If a field is accumulated across steps via `Load`/`Save`,
       zero-initialise it in `forward_cache` with `CFill(0.0)`.
    8. If `forward_indexer` uses `Save(...)`, the engine JSON MUST set
       `"disable_radix_cache": true` (default `false`). Pre-flight
       rejects the violation.

    ## Mandatory protocol — every batch is exactly 4 variants

    Every batch contains exactly **4 variants**. That fixed width
    is what makes orthogonal-knob sweeps and Pareto-frontier
    mapping meaningful. What changes with GPU availability is
    parallelism, not batch size:

    - `N >= 4` → all 4 variants run in parallel, one per free GPU.
    - `0 < N < 4` → run the 4 variants in **waves of `N`** on the
      available GPUs (sequential fallback). With `N = 1` this is
      fully serial; `N = 2` runs `2 + 2`; `N = 3` runs `3 + 1`.
    - `N == 0` → hard wait; do not launch.

    The host may share GPUs with other users; never assume the full
    physical count is yours. Detect at the start of every batch:
    ```bash
    FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
        echo "no free GPUs — wait, do not launch" >&2; exit 1
    }
    N=${#FREE_GPUS[@]}
    BATCH_SIZE=4
    PARALLEL=$N
    [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
    ```
    `free_gpus.sh` excludes any GPU with a running compute process
    or memory.used ≥ 1024 MiB. Empty result (exit 1) is a hard
    "wait" signal — that is the only condition under which you
    do not launch.

    1. **Open `algorithm_scientist/memory.md`.** Skim §1 (in-flight
       batches), §2 (completed), §3 (open hypotheses), §6 (backlog).
       This is your persistent state across sessions.
    2. **Decide the theme of the next batch.** State it plus the
       knob matrix (one knob varied per variant) in one short
       paragraph before writing code. The 4 variants must be
       ORTHOGONAL — not 4 copies of the same idea. **At least
       one variant must be *genuinely novel*** — not a paper
       replica, not a combination of two papers (those are
       catalog-adjacent — see `papers/guide.md` §16.1, they don't
       qualify), not a parameter sweep. Genuine novelty draws from
       `papers/guide.md` §16.2 (untried knobs), §16.3 (inversions),
       §16.4 (first-principles), or — best — an idea derived from
       the framework's op set itself that doesn't fit any §16
       sub-bucket. **Aim for two novel variants per batch when
       slots allow.** Defend each in one sentence that names the
       specific framework op or behaviour exploited (not "combine
       paper A with paper B"). Pre-register each novelty hypothesis
       in `algorithm_scientist/memory.md` §3 the moment the batch
       launches.
    3. **Write 8 files (4 variants × .py + .json).** Pick the next
       batch index `<x>` = number of existing
       `submissions/<tag>/batch_*_id0.json`. For each
       `<y> ∈ {0, 1, 2, 3}`:
       - `submissions/<tag>/batch_<x>_id<y>.py` with
         `@register("<tag>_batch_<x>_id<y>_cls")` (globally unique).
       - `submissions/<tag>/batch_<x>_id<y>.json` with
         `vortex_module_path: "submissions/<tag>/batch_<x>_id<y>.py"`
         and `vortex_module_name: "<tag>_batch_<x>_id<y>_cls"`.
    4. **Pre-flight all 4 locally** (CPU-only, fast):
       ```bash
       TAG=<your_tag>; BATCH=<x>
       for y in 0 1 2 3; do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')"
       done
       ```
       Drop or fix any failing variant before step 5.
    5. **RULER pre-filter — quick quality gate (≥ 0.85).** Run
       `algorithm_scientist/run_ruler.py` on each variant sequentially
       on one free GPU. Any variant scoring below **0.85 accuracy** on
       `examples/validation.jsonl` has structurally broken attention.
       Fix it (widen `vortex_topk_val`/`vortex_topk_ratio` or revise
       the indexer scoring), re-pre-flight, and re-run RULER until all
       4 pass before launching AIME24:
       ```bash
       TAG=<your_tag>; BATCH=<x>
       for y in 0 1 2 3; do
         CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \\
           python algorithm_scientist/run_ruler.py \\
             --config "submissions/${TAG}/batch_${BATCH}_id${y}.json"
       done
       ```
       Results land in `summary_ruler_submissions/<tag>/<stem>/latest.json`.
    6. **Re-check free GPUs and the concurrency cap.** Only **one**
       batch may run at a time on the GPUs you launched on. Re-run
       `algorithm_scientist/free_gpus.sh` immediately before launch
       (the set may have shifted during pre-flight/RULER) and reconcile:
       - Recompute `PARALLEL = min(N, 4)`. The batch size stays at
         4 — `N < 4` just means more waves, not fewer variants.
       - If `jobs` shows children from a previous `wait` you started
         are still alive, DO NOT launch — work on the wait-time
         activities below until they finish.
       - If `free_gpus.sh` returns nothing (exit 1), DO NOT launch —
         hard wait.
    7. **Launch the batch** (the ONLY sanctioned benchmark form),
       running the 4 variants in waves of `PARALLEL = min(N, 4)`:
       ```bash
       TAG=<your_tag>; BATCH=<x>
       BATCH_SIZE=4
       PARALLEL=$N
       [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
       LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
       mkdir -p "$LOGDIR"
       for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
           end=$((start + PARALLEL))
           [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
           for y in $(seq $start $((end - 1))); do
               cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
               gpu="${FREE_GPUS[$((y - start))]}"
               stem=$(basename "$cfg" .json)
               CUDA_VISIBLE_DEVICES=$gpu \\
                   python algorithm_scientist/run_submission_aime24.py --config "$cfg" \\
                   > "$LOGDIR/gpu${gpu}_${stem}.out" \\
                   2> "$LOGDIR/gpu${gpu}_${stem}.err" &
           done
           wait
       done
       ```
       When `N >= 4` this is one wave of 4 (fully parallel); when
       `N < 4` it serialises into ⌈4/N⌉ waves. Add a row to
       memory.md §1 with the batch tag and `$LOGDIR`. **Never run
       `python algorithm_scientist/run_submission_aime24.py` on a
       single config from this workflow** — that single-variant
       form is debug-only.
    8. **While the 4 children run (20–60 min fully parallel; longer
       when `N < 4` due to sequential waves)**, on every poll cycle
       do ONE of (and **kill any child still running after 60 min** —
       it has stalled; log in memory.md §4 and treat as failed):
       (a) **Read** the next file in priority order
       (`AI/tutorials/` → `AI/developer_guides/` → `papers/` →
       `vortex_torch/flow/algorithms.py` →
       `vortex_torch/{indexer,cache}/*` → `csrc/`); append the
       insight to memory.md §7.
       (b) **Invent.** Open `papers/guide.md` §16 and pick a
       §16.2 (untried knob), §16.3 (inversion), or §16.4
       (first-principles) prompt — *not* §16.1 (combinations),
       those are catalog-adjacent and don't fill the novelty slot.
       Better: come up with a hypothesis derived from the
       framework's op set itself that doesn't fit any §16
       sub-bucket. Sketch a one-sentence hypothesis + cache/indexer
       ops, naming the specific op or behaviour exploited. Aim for
       at least two such sketches per wait cycle.
       (c) **Design** the rest of the next batch (pre-flight all 4
       candidates) so it's ready to launch the moment `wait`
       returns. Don't launch — concurrent batches OOM the shared
       GPUs.
       (d) **Analyse children early.** Each child writes its
       `summary_submissions/<tag>/<stem>/latest.json` as soon as
       it finishes; read those that have landed and start filling §2.
       Close the §1 row once all 4 are in; update §3/§4/§5.
    9. **Failure handling.** If pre-flight or RULER filter fails for
       a variant, fix it in place. If a child's `*.err` log shows a
       traceback or its summary JSON is missing after `wait`, open
       `$LOGDIR/gpu<i>_<stem>.err` for the failing child, diagnose,
       and incorporate the lesson into memory.md §4 before
       respinning.

    ## Output format

    End every turn with a short status block:

    ```
    in-flight batches:   <N>/3
    just submitted:      <batch_id or —>
    just completed:      <batch_id or —>
    best so far:         <name> | mean@16=… | throughput=… tok/s
    memory.md updated:   <yes/no, sections touched>
    next step:           <one sentence>
    ```
""")


SUBMISSION_REVIEWER = dedent("""\
    ---
    name: vortex-submission-reviewer
    description: >-
      Use this subagent to audit an existing vortex_torch submission
      pair (submissions/<tag>/<name>.py + .json, or
      submissions/<name>.{py,json} for top-level examples) against the AGENTS.md
      contract, without modifying any code. Returns a structured
      review listing rule violations and suggestions. Invoke whenever
      the user asks to "review", "audit", or "check" a submission.
    tools: Read, Grep, Glob
    ---

    You are a strict, *read-only* reviewer for vortex_torch sparse-attention
    submissions. You never edit files; your job is to surface rule
    violations and design risks.

    ## Sources of truth

    - [AI/AGENTS.md](AI/AGENTS.md) — contract + hard rules.
    - [AI/tutorials/](AI/tutorials/) — op semantics and examples.
    - [vortex_torch/flow/algorithms.py](vortex_torch/flow/algorithms.py) —
      reference implementations.

    ## Checklist

    Read the submission's `.py` and `.json`, then verify:

    1. **Register match** — `@register("<X>")` in the .py and
       `vortex_module_name == "<X>"` in the .json are identical.
    2. **Path match** — `vortex_module_path` in the .json points at
       the actual .py file.
    3. **No native torch ops** inside `create_cache`, `forward_cache`,
       `forward_indexer`. Flag `.view`, `.sum(dim=...)`, `.mean(...)`,
       `torch.`, `@`, `+`, `*` applied directly to tensors, etc.
    4. **One op instance per call site** — no op instance is called
       from more than one site.
    5. **`k`/`v` not declared** — `create_cache` must not return keys
       named `"k"` or `"v"`.
    6. **`forward_indexer` ends in `topK(score, o, ctx=ctx)` or
       `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`** with a
       visibly `[S, 1, 1]`-shaped score. If `approxTopK` is used,
       the `tolerate_ratio` argument must be a float in `[0.0, 1.0]`.
    7. **Every declared cache field** has both a writer (in
       `forward_cache`) and a reader (in `forward_indexer` or
       `forward_cache`). No dead fields.
    8. **Cache-side reductions use `dim ∈ {1, 2}` only.** Flag any
       `CMean(dim=0)` / `CSum(dim=0)` / etc.
    9. **`Save`/`Load` fields are zero-initialised** — if the
       indexer reads-then-writes a cache field, `forward_cache` must
       `CFill(0.0)` it at block completion.
    10. **`Save(...)` in indexer ⇒ `"disable_radix_cache": true` in
        JSON.** Grep the .py for `Save(`. If present, the .json must
        explicitly set `"disable_radix_cache": true`. Without it, the
        framework's `check_engine_config` rejects the submission and
        sglang's prefix cache would corrupt Save'd state. Default
        `false`, so non-Save flows may omit the field.
    11. **JSON sanity** — `vortex_block_size` and
        `vortex_workload_chunk_size` are positive powers of 2;
        `vortex_topk_val`, `vortex_block_reserved_bos`,
        `vortex_block_reserved_eos` are sensible ints;
        `vortex_dtype` / `kv_cache_dtype` are supported values;
        `mem_fraction_static` (if present) is a float (sweet spot
        0.8-0.9; defaults to 0.8 inside `get_engine`);
        `disable_radix_cache` (if present) is a bool.

    ## Output format

    Respond with exactly this structure — nothing else:

    ```
    ## Review of submissions/<tag>/<name>  (or submissions/<name> for examples)

    ### Blockers  (framework will reject)
    - <rule #, file:line, one-sentence description>  | or: none
    ### Warnings  (likely to hurt quality or throughput)
    - <file:line, one-sentence description>          | or: none
    ### Suggestions  (throughput optimisations per AGENTS.md Objective)
    - <one-sentence description>                     | or: none
    ### Summary
    <1-2 sentences: will it compile? will it perform?>
    ```
""")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

CMD_NEW_SUBMISSION = dedent("""\
    ---
    description: Scaffold a new vortex_torch sparse-attention submission pair.
    argument-hint: <submission-name>
    ---

    Scaffold a new submission named **$1** by invoking the
    `vortex-submission-writer` subagent. The new files land at
    `submissions/<tag>/$1.{py,json}`, where `<tag>` is your
    session's agent identifier (sanitized model name, e.g.
    `claude_opus_4_7`). Examples stay flat at `submissions/example_*`.

    Steps the subagent should perform:

    1. Determine `<tag>` for this session (default: sanitized
       lowercase model name) and `mkdir -p submissions/<tag>`.
    2. Pick a starting template from the committed example library
       at `submissions/*.{py,json}` (top level — agent-tagged dirs
       are NOT examples). Available examples:
         - `example_block_sparse_attention` — minimal centroid flow
         - `gqa_quest_approx` — QUEST envelope + `approxTopK`
         - `kimi_v0` — tight throughput config (small topk, block_size=32)
         - `oai_v0` — wider budget + aggressive `vortex_layers_skip`
       Pick the one closest to the user's described idea (ask in one
       short message if ambiguous). Copy `submissions/<example>.py`
       and `submissions/<example>.json` to
       `submissions/<tag>/$1.py` and `submissions/<tag>/$1.json`.
    3. Rename the class, the `@register("...")` string (must be
       globally unique — include `<tag>` and `$1`), and the JSON's
       `vortex_module_name` / `vortex_module_path` to match the new
       location.
    4. If the user's idea differs from the picked example, customise
       `create_cache`, `forward_cache`, `forward_indexer` accordingly.
    5. Run the cheap local pre-flight:
       `python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>/$1.json')"`
    6. Report the result. Do NOT launch the benchmark automatically —
       that's `/benchmark $1` (debug, single variant) or
       `/batch-benchmark` (the sanctioned batch that fills every
       local GPU).

    Follow every rule in [AI/AGENTS.md](AI/AGENTS.md). Never use native
    torch ops inside the three vFlow methods.
""")


CMD_PREFLIGHT = dedent("""\
    ---
    description: Run check_engine_config locally (CPU-only) on a submission.
    argument-hint: <submission-name>
    ---

    Resolve `$1` to a config path. Most submissions live under your
    agent tag, so try in order: `submissions/<tag>/$1.json`,
    `submissions/$1.json` (top-level for examples), or treat `$1`
    itself as a path if it ends in `.json`. Then run the cheap
    pre-flight check:

    ```bash
    python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('<resolved path>')"
    ```

    Capture the output. If it raises, paste the relevant traceback line
    and cross-reference it with [AI/AGENTS.md §5](AI/AGENTS.md)
    "Common failures" table. Suggest the minimum edit needed to make it
    pass, but do **not** edit files unless the user asks you to.

    If it passes, remind the user that the next step is `/benchmark $1`
    (debug, single variant) or `/batch-benchmark` (the sanctioned
    batch that fills every local GPU, launched directly via python).
""")


CMD_BENCHMARK = dedent("""\
    ---
    description: (DEBUG ONLY) Run a single-variant AIME24 benchmark directly via python. Forbidden in the standard agent workflow — use /batch-benchmark instead.
    argument-hint: <submission-name>
    ---

    > **Debug-only command.** The sanctioned benchmark protocol runs
    > exclusively as a batch that fills every local GPU, via
    > `/batch-benchmark`. Use this single-variant form ONLY for human
    > debugging (e.g. confirming a new flow boots end-to-end). Do
    > not chain it as part of an automated workflow.

    Resolve `$1` to a config path: try
    `submissions/<tag>/$1.json` first, then `submissions/$1.json`,
    or treat `$1` as a literal path if it ends in `.json`. The
    runner **mirrors** the config's location under `submissions/`
    into `summary_submissions/`, so
    `submissions/<tag>/batch_3_id5.json` produces
    `summary_submissions/<tag>/batch_3_id5/`.

    Step 1 — pre-flight (CPU-only):
    ```bash
    CFG=<resolved path to submissions/.../$1.json>
    python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('$CFG')"
    ```
    Refuse to run the benchmark if pre-flight fails.

    Step 2 — launch directly. Pin to one GPU (default GPU 0; pick a
    free one if 0 is busy) and capture stdout/stderr to a log file:
    ```bash
    STEM=$(basename "$CFG" .json)                             # filename stem only
    SUMMARY_REL="${CFG#submissions/}"; SUMMARY_REL="${SUMMARY_REL%.json}"   # tag/stem (or just stem for top-level configs)
    LOGDIR="logs/submission/single_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGDIR"
    CUDA_VISIBLE_DEVICES=0 \\
        python algorithm_scientist/run_submission_aime24.py --config "$CFG" \\
        > "$LOGDIR/${STEM}.out" \\
        2> "$LOGDIR/${STEM}.err"
    ```
    The runner writes its summary into
    `summary_submissions/${SUMMARY_REL}/<timestamp>__<hash>.json`
    and updates `summary_submissions/${SUMMARY_REL}/latest.json`
    itself; no further polling needed.

    Step 3 — read the result:

    - **Success** → Open `summary_submissions/${SUMMARY_REL}/latest.json`
      and print `mean@16`, `pass@16`, `e2e_time`, `total_tokens`,
      `throughput`, and `content_hash`. For run-to-run comparisons:
      `cat summary_submissions/${SUMMARY_REL}/INDEX.jsonl | jq -c
      '{finished_at, content_hash, "mean@16", throughput}'`.
    - **Failure** (non-zero exit, or no new `latest.json` was
      written) → Open `$LOGDIR/${STEM}.err` (and `.out` if needed),
      summarise the error in 1-2 sentences, and recommend a fix. Do
      NOT re-launch without pre-flight passing first.
""")


CMD_BATCH_BENCHMARK = dedent("""\
    ---
    description: Launch the 4-variant batch on the currently-free local GPUs (parallel when N>=4, otherwise sequential waves of N). The only sanctioned benchmark form. Pass exactly 4 submission names.
    argument-hint: <name1> <name2> <name3> <name4>   (always 4 variants; parallelism = min(N_free_gpus, 4))
    ---

    Run the **4-variant batch** against the AIME24 benchmark. **Batch
    size is fixed at 4.** Parallelism depends on free-GPU count `N`:

    - `N >= 4` → all 4 variants run in parallel, one per GPU.
    - `0 < N < 4` → variants run in **waves of `N`** on the available
      GPUs (sequential fallback). With `N = 1` this is fully serial;
      `N = 2` runs `2 + 2`; `N = 3` runs `3 + 1`.
    - `N == 0` → wait, do not launch.

    This is the *only* benchmark command sanctioned by the protocol;
    single-variant runs are debug-only.

    The user passes **4 submission names** (not JSON paths); this
    command expands each `<nameI>` into `submissions/<tag>/<nameI>.json`,
    where `<tag>` is the session's agent identifier. For the standard
    iterate-loop layout, the names look like
    `batch_<x>_id0 batch_<x>_id1 batch_<x>_id2 batch_<x>_id3`.

    Step 0 — resolve `<tag>`, detect *free* GPUs (not the physical
    count — other users may be sharing this host), then count
    arguments:
    ```bash
    TAG=<your_agent_tag>           # sanitized model name, set once per session
    FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
        echo "no free GPUs — wait, do not launch" >&2; exit 1
    }
    N=${#FREE_GPUS[@]}
    BATCH_SIZE=4
    PARALLEL=$N
    [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
    NAMES=($ARGUMENTS)
    if [ "${#NAMES[@]}" -ne "$BATCH_SIZE" ]; then
        echo "expected $BATCH_SIZE variants, got ${#NAMES[@]}" >&2
        exit 1
    fi
    echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL, waves=$(( (BATCH_SIZE + PARALLEL - 1) / PARALLEL )))"
    ```
    Refuse cases:
      - `N == 0` (helper exited non-zero): "No free GPUs. Wait, do
        not launch."
      - `${#NAMES[@]} != 4`: "Batches must contain exactly 4
        variants. Got ${#NAMES[@]} — re-design and re-invoke."
    Do not silently shrink or pad the batch.

    Step 1 — concurrency cap. Only **one** batch may run at a time
    on the GPUs you target:
    ```bash
    jobs -r | wc -l
    ```
    If background jobs from a previous batch are still alive, refuse
    to launch — tell the user to wait, or to use the wait-time
    activities in `algorithm_scientist/memory.md`. (Other users'
    processes are already filtered out by `free_gpus.sh`.)

    Step 2 — pre-flight every config locally first (cheap, no GPU):
    ```bash
    for name in "${NAMES[@]}"; do
        python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/${name}.json')" \\
            || echo "[preflight] FAILED: ${TAG}/${name}"
    done
    ```
    Refuse to launch any submission whose preflight failed — fix the
    failing variant first.

    Step 3 — RULER pre-filter (quick quality gate, ≥ 0.85). Run
    `algorithm_scientist/run_ruler.py` on each variant sequentially on
    one free GPU. Any variant scoring below **0.85 accuracy** on
    `examples/validation.jsonl` has structurally broken attention — fix
    it (widen `vortex_topk_val`/`vortex_topk_ratio` or revise the
    indexer), re-pre-flight, and re-run RULER until all 4 pass before
    launching AIME24.
    ```bash
    for name in "${NAMES[@]}"; do
        CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \\
            python algorithm_scientist/run_ruler.py --config "submissions/${TAG}/${name}.json"
    done
    ```
    Results land in `summary_ruler_submissions/<tag>/<name>/latest.json`.
    Refuse to launch any variant whose RULER accuracy < 0.85.

    Step 4 — launch the 4 variants in waves of `PARALLEL = min(N, 4)`,
    each pinned to a *free* GPU index (NOT 0…N-1), with `wait`
    between waves so a wave's GPUs are free before the next reuses
    them:
    ```bash
    LOGDIR="logs/submission/${TAG}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGDIR"
    for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
        end=$((start + PARALLEL))
        [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
        for y in $(seq $start $((end - 1))); do
            name="${NAMES[$y]}"
            gpu="${FREE_GPUS[$((y - start))]}"
            CUDA_VISIBLE_DEVICES=$gpu \\
                python algorithm_scientist/run_submission_aime24.py --config "submissions/${TAG}/${name}.json" \\
                > "$LOGDIR/gpu${gpu}_${name}.out" \\
                2> "$LOGDIR/gpu${gpu}_${name}.err" &
        done
        wait
    done
    ```
    When `N >= 4` this is one wave of 4 (fully parallel — same
    behaviour as the old protocol). When `N < 4` it's
    `ceil(4/N)` sequential waves on the available GPUs.
    Each child writes its result into
    `summary_submissions/<tag>/<name>/<timestamp>__<hash>.json` and
    updates `summary_submissions/<tag>/<name>/latest.json` itself.
    (The runner mirrors the config's path under `submissions/` into
    `summary_submissions/`, so `submissions/<tag>/batch_<x>_id<y>.json`
    becomes `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
    isolation, no collisions across agents that pick the same stem.)

    Step 5 — append a row to `algorithm_scientist/memory.md` §1
    *In-flight batches* the moment you launch:
    `| <tag> | <batch_id> | <UTC time> | <LOGDIR> | <name1>,…,<name4> | RUNNING |`

    Step 6 — while waiting (the batch takes **20–60 minutes** when fully
    parallel, longer with `N < 4` due to sequential waves; **kill any
    child still running after 60 minutes** — log the error in
    memory.md §4 and treat that variant as failed), do NOT idle. Spend the time
    reading tutorials / developer guides / source, or designing the
    next batch (don't launch — concurrent batches OOM the shared
    GPUs), or analysing children that have already produced their
    `latest.json`. See AGENTS.md §5d.

    Step 7 — once the outer `wait` loop returns (all waves done):

    - **Success** → for each `<nameI>`, read
      `summary_submissions/<tag>/<nameI>/latest.json`. Produce a comparison
      table with columns: `name | content_hash | mean@16 | pass@16 |
      throughput | e2e_time`. Identify the **Pareto-non-dominated**
      variants for this batch on the `(throughput, mean@16)` plane —
      a variant is dominated iff some other variant has *both* equal-
      or-higher `throughput` *and* equal-or-higher `mean@16`. Append
      the table + a 1-3 sentence takeaway naming each frontier
      variant and where it sits relative to the running §5 best, to
      `algorithm_scientist/memory.md` §2 *Completed batches*; remove
      the row you added in step 5 from §1.
    - **Failure** → any child whose process exited non-zero (or
      whose `latest.json` is missing / older than `$LOGDIR`'s
      timestamp) wrote its traceback to
      `$LOGDIR/gpu<i>_<name>.err`. Open the failing child's `.err`,
      summarise the error, and record the lesson in
      `algorithm_scientist/memory.md` §4 *Anti-patterns*.
""")


CMD_REVIEW = dedent("""\
    ---
    description: Audit a submission pair against AGENTS.md rules (read-only review).
    argument-hint: <submission-name>
    ---

    Invoke the `vortex-submission-reviewer` subagent to audit a
    submission pair against the [AI/AGENTS.md](AI/AGENTS.md) contract.

    Resolve `$1` to a path: try `submissions/<tag>/$1.{py,json}`
    first (the standard agent-tagged location), then
    `submissions/$1.{py,json}` as a fallback for top-level examples.

    The reviewer will produce a structured report with sections:
    **Blockers**, **Warnings**, **Suggestions**, **Summary**. It does
    not modify any file — if the user wants a fix, they should follow
    up with `/new-submission` or ask explicitly.
""")


CMD_ITERATE = dedent("""\
    ---
    description: Run the long-horizon iterate loop autonomously — design 4 variants, pre-flight, RULER filter, launch AIME24, wait, analyse, repeat. Stops after --max-iterations batches (default 3).
    argument-hint: [--max-iterations <N>]
    ---

    You are now running the **vortex_torch iterate loop** autonomously.
    Do not ask for confirmation between steps; execute each step in
    sequence and loop until the iteration budget is reached.

    ## Step 0 — parse arguments and activate conda env

    Parse `$ARGUMENTS` for `--max-iterations <N>` (default: 3).
    Extract N:
    ```bash
    MAX_ITER=3
    for arg in $ARGUMENTS; do
      case "$arg" in --max-iterations) MAX_ITER_NEXT=1 ;; *)
        [ "$MAX_ITER_NEXT" = "1" ] && { MAX_ITER=$arg; MAX_ITER_NEXT=0; } ;; esac
    done
    echo "max_iterations=$MAX_ITER"
    ```

    Activate the conda environment:
    ```bash
    CONDA_BASE=$(conda info --base 2>/dev/null || echo /root/anaconda3)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vortex_v1
    python -c "import sys; print(sys.executable)"
    ```

    ## Step 1 — pick your tag and read context (once per session)

    Your `<tag>` is a sanitized lowercase form of your model name
    (e.g. `claude_sonnet_4_6`, `claude_opus_4_7`). Create the
    submissions dir if it doesn't exist:
    ```bash
    TAG=<your_tag>
    mkdir -p submissions/$TAG
    ```

    Read these files **in order** (skip any already loaded this session):
    1. [AI/AGENTS.md](../AI/AGENTS.md)
    2. [AI/tutorials/overview.md](../AI/tutorials/overview.md)
    3. [AI/tutorials/program_create_cache.md](../AI/tutorials/program_create_cache.md)
    4. [AI/tutorials/program_forward_cache.md](../AI/tutorials/program_forward_cache.md)
    5. [AI/tutorials/program_forward_indexer.md](../AI/tutorials/program_forward_indexer.md)
    6. [AI/tutorials/cache_op.md](../AI/tutorials/cache_op.md)
    7. [AI/tutorials/indexer_op.md](../AI/tutorials/indexer_op.md)
    8. [vortex_torch/flow/algorithms.py](../vortex_torch/flow/algorithms.py)
    9. [papers/guide.md](../papers/guide.md) — especially §14, §16
    10. [algorithm_scientist/memory.md](../algorithm_scientist/memory.md)

    If `memory.md §1` shows a batch RUNNING, skip to Step 5 (wait
    activities) until it finishes, then resume the loop.

    ## Step 2 — set the batch index (once per loop iteration)

    ```bash
    BATCH=$(ls submissions/$TAG/batch_*_id0.json 2>/dev/null | wc -l)
    echo "next batch index: $BATCH"
    ```

    ## Step 3 — design the 4-variant batch

    State the batch theme in one short paragraph. Define the knob
    matrix — one row per variant (id0…id3).

    Variant composition rule:
    - **id0–id1 (aim for 2): genuinely novel** — §16.2 (untried knob),
      §16.3 (inversion), §16.4 (first-principles), or an op-set idea
      that doesn't fit any §16 sub-bucket. One sentence defending each,
      naming the specific op or behaviour exploited.
    - **id2–id3: §16.5 parameter sweeps** — `vortex_topk_val`,
      `approxTopK` vs `topK`, layer-skip patterns, fp8/bf16 KV, etc.
      Explicitly encouraged for non-novelty slots.

    Pre-register each novelty hypothesis as a one-sentence row in
    `algorithm_scientist/memory.md` §3.

    ## Step 4 — write 8 files and pre-flight

    Write `submissions/$TAG/batch_${BATCH}_id{0,1,2,3}.{py,json}`.
    Each `.py` uses `@register("${TAG}_batch_${BATCH}_id<y>_cls")`.
    Each `.json` sets `vortex_module_path` and `vortex_module_name`.

    Pre-flight all 4 (CPU-only):
    ```bash
    for y in 0 1 2 3; do
      python -c "from vortex_torch.engine.sgl import check_engine_config; \\
    check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')" \\
        && echo "[ok] id$y" || echo "[FAIL] id$y"
    done
    ```
    Fix any failure before continuing.

    ## Step 5 — RULER pre-filter (≥ 0.85)

    Detect one free GPU, then run sequentially:
    ```bash
    FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
      echo "no free GPUs — waiting"; exit 1
    }
    for y in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \\
        python algorithm_scientist/run_ruler.py \\
          --config "submissions/${TAG}/batch_${BATCH}_id${y}.json"
    done
    ```
    Any variant with `accuracy < 0.85` has broken attention — fix it
    (widen `vortex_topk_val`/`vortex_topk_ratio` or revise the indexer),
    re-pre-flight, and re-run RULER until all 4 pass.

    ## Step 6 — detect free GPUs and launch AIME24

    Re-detect free GPUs immediately (set may have shifted):
    ```bash
    FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
      echo "no free GPUs — hard wait"; exit 1
    }
    N=${#FREE_GPUS[@]}
    BATCH_SIZE=4
    PARALLEL=$N
    [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
    echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL)"
    LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGDIR"
    for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
        end=$((start + PARALLEL))
        [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
        for y in $(seq $start $((end - 1))); do
            cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
            gpu="${FREE_GPUS[$((y - start))]}"
            stem=$(basename "$cfg" .json)
            CUDA_VISIBLE_DEVICES=$gpu \\
                python algorithm_scientist/run_submission_aime24.py \\
                  --config "$cfg" \\
                > "$LOGDIR/gpu${gpu}_${stem}.out" \\
                2> "$LOGDIR/gpu${gpu}_${stem}.err" &
        done
        wait
    done
    ```
    Add a row to `algorithm_scientist/memory.md` §1 the moment you launch:
    `| $TAG | batch_$BATCH | <time> | $LOGDIR | batch_${BATCH}_id0…id3 | RUNNING |`

    ## Step 7 — wait (20–60 min) and do productive work

    **Kill any child still running after 60 minutes** (`kill %<job>`),
    log the error in memory.md §4, and treat that variant as failed.

    On each polling cycle (check `jobs` for alive children), do ONE of:

    (a) **Read.** Next file in priority order:
        `AI/tutorials/` → `AI/developer_guides/` → `papers/` →
        `vortex_torch/flow/algorithms.py` →
        `vortex_torch/{indexer,cache}/*` → `csrc/`.
        Append one insight bullet to memory.md §7.

    (b) **Invent.** Pick a §16.2/§16.3/§16.4 prompt from
        `papers/guide.md` (NOT §16.1 combinations; NOT §16.5 sweeps —
        those fill slots but don't count as novelty). Sketch a
        one-sentence hypothesis naming the specific op or behaviour
        exploited. Aim for two sketches per wait cycle.

    (c) **Design next batch.** Pre-flight all 4 candidates so they're
        ready the moment `wait` returns. Do NOT launch — concurrent
        batches OOM the shared GPUs.

    (d) **Analyse children early.** Each child writes its
        `summary_submissions/$TAG/batch_${BATCH}_id<y>/latest.json`
        as soon as it finishes. Read landed summaries and start
        filling memory.md §2.

    ## Step 8 — analyse, update memory.md, check budget

    Read all 4 `summary_submissions/$TAG/batch_${BATCH}_id<y>/latest.json`.
    Produce a comparison table:

    ```
    | variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput | e2e_time |
    ```

    Identify Pareto-non-dominated variants on the `(throughput, mean@16)`
    plane. Append the table + 1–3 sentence takeaway to memory.md §2;
    remove the §1 row; update §3 (hypotheses) / §4 (anti-patterns) /
    §5 (winners).

    **Check iteration budget:**
    - Count batches launched this session =
      `ls submissions/$TAG/batch_*_id0.json | wc -l` minus the count
      at session start.
    - If `MAX_ITER > 0` AND `batches_launched >= MAX_ITER`: write a
      final 2-paragraph summary (best submission name, content_hash,
      mean@16, throughput, key design decisions), update memory.md §8,
      and **stop**.
    - Otherwise: go back to Step 2.
""")


CMD_INNOVATE = dedent("""\
    ---
    description: Draft N novel sparse-attention submissions in one shot — algorithm-innovation focus, all must compile, no benchmark loop, no memory.md mutation.
    argument-hint: <N> [theme-hint]
    ---

    Generate **N novel submissions** in a single shot. This is the
    *innovation-draft* mode — the complement to `/iterate`. There is
    **no benchmark loop**, **no memory.md mutation**, and **no batch
    sizing rule** (`N` is whatever the user passes, not capped to 4 or
    the free-GPU count).

    Every variant must satisfy two non-negotiable requirements:

    1. **Genuinely novel algorithm.** Drawn from
       [papers/guide.md](../papers/guide.md) §16.2 (untried knobs),
       §16.3 (inversions), §16.4 (first-principles), or — best —
       from a hypothesis derived from the framework's own op set
       that doesn't fit any §16 sub-bucket. **Not** a paper replica,
       **not** a combination of two papers (those are §16.1
       catalog-adjacent and disqualified here), **not** a pure
       parameter sweep over an existing flow.
    2. **Compiles.** Every variant must pass the local pre-flight
       (`check_engine_config`). A novel idea that does not compile is
       not an output of this mode — fix it or drop it before
       reporting completion.

    The optional second argument is a free-form theme hint
    (e.g. `page-level EMA tracking`, `channel-band gating`,
    `Save/Load-state across decode steps`) that biases all N variants
    toward a related family. Without a hint, draw freely from §16 or
    the op set.

    ## Output layout

    The N submission pairs land at:
    ```
    submissions/<tag>/innovate_<x>_id<y>.py
    submissions/<tag>/innovate_<x>_id<y>.json
    ```
    for `y ∈ {0 … N-1}`, where `<tag>` is your session's agent
    identifier and `<x>` is the next innovate-run index:
    ```bash
    TAG=<your_agent_tag>           # sanitized model name
    X=$(ls submissions/${TAG}/innovate_*_id0.json 2>/dev/null | wc -l)
    ```
    The `innovate_<x>_id<y>` prefix is intentionally distinct from
    the `batch_<x>_id<y>` prefix used by `/iterate` and
    `/batch-benchmark`, so the two modes never collide on filenames
    within the same tag.

    ## Steps

    Step 0 — **validate args**:
    ```bash
    NARGS=($ARGUMENTS)
    N=${NARGS[0]:-}
    [ -z "$N" ] && { echo "usage: /innovate <N> [theme-hint]" >&2; exit 1; }
    case "$N" in *[!0-9]*) echo "N must be a positive integer" >&2; exit 1 ;; esac
    [ "$N" -lt 1 ] && { echo "N must be >= 1" >&2; exit 1; }
    THEME="${NARGS[@]:1}"   # may be empty
    ```

    Step 1 — **resolve `<tag>` and run index `<x>`**, and create the
    agent dir if needed:
    ```bash
    mkdir -p "submissions/${TAG}"
    X=$(ls submissions/${TAG}/innovate_*_id0.json 2>/dev/null | wc -l)
    echo "innovate run: tag=${TAG} x=${X} N=${N} theme=\\"${THEME}\\""
    ```

    Step 2 — **read the canonical context once** (skip files already
    loaded this session):
    - [AI/AGENTS.md](../AI/AGENTS.md) §1-§5 — contract + hard rules.
    - [AI/tutorials/overview.md](../AI/tutorials/overview.md) and the
      five op/program tutorials.
    - [papers/guide.md](../papers/guide.md) §16 — off-catalog prompts.
    - [vortex_torch/flow/algorithms.py](../vortex_torch/flow/algorithms.py)
      — six reference flows for op patterns.

    Step 3 — **for each variant `y ∈ {0 … N-1}`**, in one short
    paragraph state:
    - the **novelty hypothesis** (one sentence, naming the specific
      framework op or behaviour exploited — *not* "combine paper A
      with paper B"),
    - the §16 sub-bucket or op-set thread it draws from,
    - the cache fields it will declare and the indexer ops it will
      call.

    If the user supplied a theme hint, every variant must connect to
    it (different angles on the same family). Without a hint, the N
    variants must be **orthogonal** — different mechanisms, not N
    copies of the same idea.

    Step 4 — **write each variant**. For each `y`:

    `submissions/${TAG}/innovate_${X}_id${y}.py`:
    - Globally-unique decorator: `@register("${TAG}_innovate_${X}_id${y}_cls")`.
    - vFlow subclass implementing `create_cache` / `forward_cache` /
      `forward_indexer`.
    - Use **only** `vortex_torch.indexer.*` and `vortex_torch.cache.*`
      ops. No native torch ops inside the three methods.
    - Each op instance is a single call site (don't share).
    - Don't declare `"k"` or `"v"` in `create_cache`.
    - `forward_indexer` ends in `topK(score, o, ctx=ctx)` or
      `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`; `score`
      must be ragged `[S, 1, 1]`.
    - Cache-side reductions only on `dim ∈ {1, 2}`.
    - Any field touched by `Load`/`Save` across steps must be
      `CFill(0.0)`-initialised in `forward_cache`.

    `submissions/${TAG}/innovate_${X}_id${y}.json`:
    - `vortex_module_path: "submissions/${TAG}/innovate_${X}_id${y}.py"`
    - `vortex_module_name: "${TAG}_innovate_${X}_id${y}_cls"`
    - `disable_radix_cache: true` **iff** `forward_indexer` uses
      `Save(...)` (the pre-flight enforces this).

    Step 5 — **mandatory pre-flight gate**. Compile every variant:
    ```bash
    FAILED=()
    for y in $(seq 0 $((N - 1))); do
        cfg="submissions/${TAG}/innovate_${X}_id${y}.json"
        if python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('${cfg}')" 2>&1; then
            echo "[preflight] ok: ${cfg}"
        else
            echo "[preflight] FAIL: ${cfg}"
            FAILED+=("$y")
        fi
    done
    ```
    For every variant in `FAILED`: open the traceback, identify the
    offending op or config field, fix it **in that variant's
    .py/.json** (do *not* delete the variant — the goal is N
    compiling submissions), and re-run the pre-flight on it. Repeat
    until every variant passes.

    If after a reasonable fix attempt a variant still cannot be made
    to compile, surface the residual error to the user and stop —
    do **not** silently report a partial run. The user can either
    loosen the theme, lower `N`, or accept the partial output
    explicitly.

    Step 6 — **report**. Print a table with one row per variant:

    ```
    | y | file                                              | bucket | hypothesis (≤1 sentence)                | preflight |
    |---|---------------------------------------------------|--------|-----------------------------------------|-----------|
    | 0 | submissions/<tag>/innovate_<x>_id0.{py,json}      | §16.2  | <one-sentence>                           | ok        |
    | 1 | submissions/<tag>/innovate_<x>_id1.{py,json}      | op-set | <one-sentence>                           | ok        |
    | … | …                                                 | …      | …                                       | …         |
    ```

    Then **stop**. Do **not** call `/batch-benchmark`. Do **not**
    touch `algorithm_scientist/memory.md`. Hand the user the N paths;
    they decide whether to feed any of them into a future
    `/batch-benchmark` (groups of 4) or `/iterate` run.

    ## What this mode is *not*

    - **Not a benchmark.** No `run_submission_aime24.py` invocation;
      no GPU is touched.
    - **Not iterative.** One shot, then return. No "wait + analyse +
      next batch" loop.
    - **Not bound to batch=4.** `N` is the user's choice — useful
      values are typically 2-12 depending on the theme's surface area.
    - **Not memory.md-aware.** Does not read or write the persistent
      notebook. (If the user wants a draft to feed into the iterate
      loop later, they can copy it from `submissions/<tag>/` into a
      `batch_<x>_id<y>` slot themselves.)

    ## Hard refusals

    - `N` missing, non-numeric, or `< 1` → reject with `usage:
      /innovate <N> [theme-hint]`.
    - A drafted variant produces only catalog-adjacent ideas (paper
      combinations, parameter sweeps over `example_*`) → reject the
      variant and re-draft it with a §16.2/§16.3/§16.4 or op-set
      hypothesis. Do **not** report the run as complete with a
      catalog-adjacent variant in the output.
    - A variant cannot be made to pre-flight after a fix attempt →
      surface the error and stop the entire run.
""")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

FILES = {
    CLAUDE_DIR / "CLAUDE.md":                                       CLAUDE_MD,
    CLAUDE_DIR / "agents"   / "vortex-submission-writer.md":        SUBMISSION_WRITER,
    CLAUDE_DIR / "agents"   / "vortex-submission-reviewer.md":      SUBMISSION_REVIEWER,
    CLAUDE_DIR / "commands" / "new-submission.md":                  CMD_NEW_SUBMISSION,
    CLAUDE_DIR / "commands" / "preflight.md":                       CMD_PREFLIGHT,
    CLAUDE_DIR / "commands" / "benchmark.md":                       CMD_BENCHMARK,
    CLAUDE_DIR / "commands" / "batch-benchmark.md":                 CMD_BATCH_BENCHMARK,
    CLAUDE_DIR / "commands" / "review.md":                          CMD_REVIEW,
    CLAUDE_DIR / "commands" / "iterate.md":                         CMD_ITERATE,
    CLAUDE_DIR / "commands" / "innovate.md":                        CMD_INNOVATE,
}


def main() -> None:
    import sys
    if "--force-clobber" not in sys.argv:
        raise SystemExit(
            "DEPRECATED: .claude/ is now hand-maintained — running this would "
            "clobber the current commands. Pass --force-clobber only if you "
            "really intend to regenerate the OLD command set."
        )
    for path, body in FILES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        print(f"wrote  {path.relative_to(REPO_ROOT)}  ({len(body)} bytes)")

    print()
    print(f"done — .claude/ tree under {CLAUDE_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
