---
description: Write a compiled LaTeX paper (one-pager or full) about your vortex research, pulling numbers from summary_*submissions/ + memory.md, using the tectonic tex env + vortex_paper/ style.
argument-hint: [--scope onepager|full] [--tag <agent-tag>]
---

Produce a **compiled** LaTeX paper from your `/iterate` / `/innovate` results.
Driven by the `vortex-paper-writer` subagent. Reference:
[AI/workflows/paper.md](../../AI/workflows/paper.md).

## Step 0 — parse args, gather results

- `--scope onepager` (default) | `full`.
- `--tag <agent-tag>` (default: your sanitized model name).

```bash
conda activate tex      # tectonic + poppler live here
python algorithm_scientist/collect_results.py --tag <tag> --json /tmp/results.json
```
`collect_results.py` aggregates every `summary_*submissions/<tag>/*/latest.json`
into a table (`mean@16` / `pass@16` / `throughput` per submission per task).
Narrative + design rationale come from `algorithm_scientist/memory.md` (§2
completed, §5 winners). Use the *Pareto-non-dominated* variants as the headline.

## Step 1 — build the paper

Invoke `Task(subagent_type="vortex-paper-writer", ...)` with the scope, the
results JSON, and the memory.md highlights.
- **onepager:** clone `vortex_paper/vortex_onepager.tex` (colorful, 2-page:
  content + refs); swap in the headline `(throughput, mean@16)` vs full
  attention, the method paragraph, and a results table/figure.
- **full:** scaffold intro / method / evaluation / related / refs under
  `vortex_paper/`, with a Pareto plot + tables from the collected results.

Cite from `vortex_paper/reference.bib` (add entries as needed). Use the
`fontawesome` (FA4) package — `fontawesome5` crashes this tectonic build.

## Step 2 — compile + verify (mandatory)

```bash
cd vortex_paper && tectonic <file>.tex
pdfinfo <file>.pdf | grep ^Pages        # confirm page count
pdftoppm -r 130 -png -f 1 -l 1 <file>.pdf /tmp/pg   # render + visually check page 1
```
Fix any undefined citations / overflow before reporting done. Artifacts live
under git-ignored `vortex_paper/`.

## Output

`scope | .tex/.pdf paths | pages | compiled (exit 0) | undefined-cites: none |
headline result quoted`. Don't claim it builds without actually compiling and
rendering it.
