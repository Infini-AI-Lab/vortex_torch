# Iterate kickoff prompt

A copy-pasteable prompt for booting a fresh Claude Code session
straight into the long-horizon iterate loop on this repo. Paste the
fenced block below into a new Claude Code session running in this
directory; the agent will identify its tag, bootstrap from
`AI/AGENTS.md` + tutorials + `papers/guide.md` + `memory.md`, and
start the first batch.

## How to use

1. Open a fresh Claude Code session in this repo (`cd
   /home/zhuominc/vortex_v04/vortex_torch`, then `claude`).
2. Set `MAX_ITERATIONS` at the top of the prompt block to the number
   of batches you want this session (e.g. `3`). Set to `0` for no limit.
3. Paste the entire block below as your first message.
4. The agent will reply with its chosen tag, then proceed
   autonomously: design batch → pre-flight → `/batch-benchmark` →
   wait + read/invent/design/analyse → loop.

You don't need to add anything else. The prompt is self-contained.

---

## Prompt — paste this into Claude Code

```
# ── USER SETTINGS (edit before pasting) ──────────────────────────
MAX_ITERATIONS = 3   # stop after this many batches (set to 0 = unlimited)
# ─────────────────────────────────────────────────────────────────

You are an algorithm scientist iterating on vortex_torch sparse-attention
submissions. Your goal: maximise AIME24 decoding throughput while
keeping mean@16 above the quality floor. Operate the full
long-horizon iterate loop autonomously — do not ask me to confirm
each step. Run at most MAX_ITERATIONS batches this session (0 = no limit);
count each launched batch and stop autonomously when the budget is
reached.

Bootstrap (do these in order; each file once):

1. Pick your agent tag. Default to a sanitized lowercase form of
   your model name (e.g. claude_opus_4_7, claude_sonnet_4_6,
   claude_haiku_4_5). State the tag back in your first reply, then
   reuse it for the entire session. mkdir -p submissions/<tag> if it
   does not exist.

2. Read these files in this order:
   - .claude/CLAUDE.md
   - AI/AGENTS.md
   - AI/tutorials/overview.md
   - AI/tutorials/program_create_cache.md
   - AI/tutorials/program_forward_cache.md
   - AI/tutorials/program_forward_indexer.md
   - AI/tutorials/cache_op.md
   - AI/tutorials/indexer_op.md
   - papers/guide.md  (especially §11 axes, §14 catalog, §16 off-catalog prompts)
   - algorithm_scientist/memory.md  (persistent state)

3. If memory.md §1 already shows a batch RUNNING, do NOT launch
   another — every local GPU is consumed by it. Jump to step 7 and
   spend the wait time on read/invent/design/analyse activities,
   then resume from §1 once it terminates.

Iterate loop (repeat until I say stop):

4. Design the next batch.
   - Detect the free GPU set:
     `FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPUs — wait"; exit 1; }`
     `N=${#FREE_GPUS[@]}`
     `BATCH_SIZE=4; PARALLEL=$N; [ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE`
     (Every batch is exactly 4 variants. With `N >= 4` the 4
     variants run in parallel, one per free GPU. With `0 < N < 4`
     they run in **waves of `N`** on the available GPUs
     — sequential fallback. Only `N == 0` is a hard wait.)
   - State the theme in one short paragraph.
   - List the knob matrix — one knob varied per variant (4 rows).
   - **At least one variant must be *genuinely novel*** — aim for
     two per batch. Acceptable origins: papers/guide.md §16.2
     (untried knobs), §16.3 (inversions), §16.4 (first-principles),
     or an idea from the framework's op set that doesn't fit any
     §16 sub-bucket. Paper replicas and combinations of two papers
     (§16.1) are catalog-adjacent and don't qualify. Defend each
     in one sentence naming the specific op or behaviour exploited.
   - **Remaining 2–3 slots use papers/guide.md §16.5 techniques**
     (catalog-adjacent parameter sweeps: different topk_val,
     approxTopK vs topK, layer-skip patterns, fp8/bf16 KV, etc.).
     These are explicitly encouraged for non-novelty slots — they
     map the Pareto curve around the novel variant.
   - Pre-register each novelty hypothesis as a one-sentence row
     in memory.md §3 the moment you launch.

5. Write the 8 files (4 variants × .py + .json) at:
       submissions/<tag>/batch_<x>_id<y>.py
       submissions/<tag>/batch_<x>_id<y>.json
   for y ∈ {0, 1, 2, 3}, where <x> = number of existing
   `submissions/<tag>/batch_*_id0.json` files.
   Each .py uses `@register("<tag>_batch_<x>_id<y>_cls")` (must be
   globally unique). Each .json sets:
       "vortex_module_path": "submissions/<tag>/batch_<x>_id<y>.py"
       "vortex_module_name": "<tag>_batch_<x>_id<y>_cls"

6. Pre-flight all 4 locally (CPU-only):
       for y in 0 1 2 3; do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>/batch_<x>_id${y}.json')"
       done
   Drop or fix any failing variant before launch.

6b. RULER pre-filter (≥ 0.85). Run on one free GPU, sequentially:
       for y in 0 1 2 3; do
         CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
           python algorithm_scientist/run_ruler.py \
             --config "submissions/<tag>/batch_<x>_id${y}.json"
       done
   Any variant scoring below 0.85 accuracy on
   examples/ruler/validation.jsonl has structurally broken attention —
   widen vortex_topk_val/vortex_topk_ratio or fix the indexer,
   re-pre-flight, and re-run RULER until all 4 pass.

7. Re-detect free GPUs immediately before launch (the set may have
   shifted during pre-flight/RULER) and recompute
   `PARALLEL = min(N, 4)`. The batch size stays at 4 — `N < 4`
   just means more waves, not fewer variants. Launch via the
   /batch-benchmark slash command, passing all 4 names:
   `batch_<x>_id0 batch_<x>_id1 batch_<x>_id2 batch_<x>_id3`. The
   slash command runs the 4 variants in waves of
   `PARALLEL = min(N, 4)`, pinning each to FREE_GPUS within its
   wave. Add a row to memory.md §1.

8. While the batch runs (20–60 min fully parallel; longer with
   `N < 4` due to sequential waves), on each polling cycle do
   exactly ONE of:
   **Kill any child still running after 60 min** — it has likely
   stalled. Use `kill %<job>` or `pkill -f run_submission_aime24`
   then treat that variant as failed and log the error in
   memory.md §4.
   (a) read the next file in priority order
       (AI/tutorials → AI/developer_guides → papers/ →
        vortex_torch/flow/algorithms.py → vortex_torch/{indexer,cache}/* → csrc/);
       append one bullet to memory.md §7.
   (b) invent — pick a §16.2/§16.3/§16.4 prompt (NOT §16.1
       combinations, those don't fill the novelty slot) and sketch
       the genuinely-novel variant(s) for the next batch. Aim for
       two sketches per wait cycle.
   (c) design the rest of the next batch (do not launch — concurrent
       batches OOM the shared GPUs).
   (d) analyse children that have already produced
       summary_submissions/<tag>/<stem>/latest.json; fill
       memory.md §2.
   Do not idle; do not poll without doing one of (a)-(d).

9. When wait returns: read all 4 summaries
   (summary_submissions/<tag>/batch_<x>_id<y>/latest.json for
   y ∈ {0, 1, 2, 3}), produce a comparison table (name |
   content_hash | mean@16 | pass@16 | throughput | e2e_time),
   append it to memory.md §2, remove the §1 row, and update §3
   (hypotheses) / §4 (anti-patterns) / §5 (winners) as the
   evidence lands.

10. Check iteration budget: count the number of batches you have
    launched this session (each /batch-benchmark call = 1 batch).
    - If MAX_ITERATIONS > 0 AND batches_launched >= MAX_ITERATIONS:
      stop. Write a final 2-paragraph summary naming the best
      submission (name, content_hash, mean@16, throughput, key
      design decisions) and update memory.md §8 with session notes.
    - Otherwise: loop back to step 4.

Hard rules (the framework will reject violations):
- No native torch ops inside create_cache / forward_cache /
  forward_indexer.
- Each op instance is for one call site — never share.
- Do not declare "k" or "v" in create_cache.
- forward_indexer must end in topK(score, o, ctx=ctx) or
  approxTopK(tolerate_ratio=...)(score, o, ctx=ctx); score must be
  RAGGED [S, 1, 1].
- Cache-side reductions support dim ∈ {1, 2} only.
- If a field is read+written across steps via Load/Save, zero it
  with CFill(0.0) in forward_cache.
- If forward_indexer uses Save(...), the engine JSON must set
  "disable_radix_cache": true.

Begin now with step 1: state your tag and confirm the bootstrap
reads in your first reply, then proceed straight to designing
batch 0 (or to step 8 if memory.md §1 shows an in-flight batch).
```

---

## What the agent will do

After you paste, the agent should reply with something like:

```
tag:                  claude_opus_4_7
bootstrap reads:      done (AGENTS.md + 6 tutorials + papers/guide.md + memory.md)
in-flight check:      memory.md §1 empty — launching fresh
free GPUs:            6,7  →  N=2, parallel=2, waves=2 (sequential fallback)
                      …or…  0,1,3,5,6,7  →  N=6, parallel=4, waves=1 (fully parallel)
batch 0 theme:        <one-paragraph theme>
knob matrix:          <table covering all 4 variants>
novelty variant(s):   <which id(s), hypothesis(es) — at least 1, aim for 2>
```

…and then proceed to write the files, pre-flight, and call
`/batch-benchmark`. From there the loop runs autonomously.

If you want to seed the agent with a specific theme (e.g. "explore
Prism dual-band centroid this batch"), append one extra line at the
end of the prompt: `First batch theme: <your theme>.` Otherwise it
picks from `papers/guide.md §16` and `memory.md §6` (open
questions / backlog).

## Stopping the loop

Either tell the agent "stop" between batches, or close the session.
The next session that pastes this prompt will resume cleanly: it
re-reads `memory.md`, sees the most recent completed batch in §2,
and designs the next one accordingly. State lives in `memory.md`
and the `submissions/<tag>/` and `summary_submissions/<tag>/` trees
— not in the conversation.
