# Writing and reproducing papers

## Toolchain (already set up)

- **LaTeX engine:** `tectonic` in the **`tex`** conda env (fetches packages on
  first run; `poppler`'s `pdftoppm` is installed there too for rendering page
  previews to PNG so you can visually verify).
  ```bash
  conda activate tex
  tectonic <file>.tex            # -> <file>.pdf
  pdftoppm -r 130 -png <file>.pdf /tmp/pg   # preview pages
  ```
  Note: `fontawesome5` crashes this tectonic build (`free(): invalid pointer`);
  use the older `fontawesome` (FA4) package instead.
- **Style + example:** `vortex_paper/` holds the NeurIPS sources and
  `reference.bib`; `vortex_paper/vortex_onepager.tex` is a colorful, compact
  standalone one-pager (gradient banner, icon-badged section headings, stat
  strip, drop-shadow boxes, two-up Figure 1). Reuse it as the template.
- **Results:** `algorithm_scientist/collect_results.py --tag <tag> --json out.json`
  aggregates every `summary_*submissions/<tag>/*/latest.json` into a table
  (`mean@16` / `pass@16` / `throughput` per submission per task). Narrative
  context lives in `algorithm_scientist/memory.md` (§2 completed, §5 winners).

## `/write-paper`

Builds a compiled PDF from your `iterate` / `innovate` results.

- `--scope onepager` (default): start from `vortex_paper/vortex_onepager.tex`,
  swap in your headline numbers (best `(throughput, mean@16)` vs full attention),
  your method's one-paragraph description, and a results table/figure from
  `collect_results.py`.
- `--scope full`: scaffold a multi-section paper (intro / method / evaluation /
  related / refs) under `vortex_paper/`, pulling the Pareto plot and tables from
  the collected results and the design rationale from `memory.md`.

Always `tectonic`-compile and render page 1 to confirm it builds and fits before
reporting done. Cite from `vortex_paper/reference.bib` (add entries as needed).

## `/reproduce-paper <arxiv-id | path/to.pdf | papers/<name>>`

Compose a paper's algorithm as a vortex submission.

1. **Acquire + read** the paper: arXiv id → fetch; local PDF → read; or a
   curated `papers/<name>` entry (`papers/guide.md` summarizes the ten bundled
   papers). Extract the *sparsity mechanism* (what each query attends to) and
   the *scoring* (how pages are ranked).
2. **Map onto vortex ops.** Express the score in `forward_indexer` ending in
   `topK`/`approxTopK`; put per-block summaries in `create_cache` /
   `forward_cache`. Check the op exists in `vortex_torch/{indexer,cache}/`.
3. **Missing op? → chain `/add-ops`** (see [add_op.md](add_op.md)) to author the
   op + kernel, then continue.
4. **Scaffold + preflight** the submission(s) (`/new-submission`,
   `check_engine_config`), RULER-gate, then optionally `/iterate` to map the
   accuracy–throughput tradeoff against full attention.
5. Record the reproduction (paper → flow mapping, any new ops) in `memory.md`.
