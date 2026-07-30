# Response to W1: AI-agent-driven exploration

We thank the reviewer and agree that the exploration procedure should be
described more explicitly. We will add the following summary. The complete
executable prompts are released in `.claude/commands/{innovate,iterate}.md`;
the programming contract and operation set are in `AI/AGENTS.md` and
`AI/tutorials/`.

| Aspect | **Innovate** | **Iterate** |
|---|---|---|
| **Prompt** | `/innovate N [theme] [--model ...]` asks the agent to formulate \(N\) novel hypotheses, specify their cache state and routing operations, implement them, and repair all compilation failures. It explicitly prohibits benchmarking and experiment-memory updates. | `/iterate [--model ...] [--task ...] [--max-iterations ...]` runs a closed loop: propose four variants, preregister hypotheses, implement, validate, benchmark, update the experiment ledger, and design the next batch. |
| **Search space** | Programs expressible by Vortex's cache/indexer DSL: per-page key/value summaries, query-dependent scoring functions, and exact or approximate top-\(k\) selection. A verified new operation may be added when the DSL is insufficient. Every candidate must represent a new algorithmic hypothesis rather than a parameter sweep. | The same program space plus system knobs affecting the quality–speed tradeoff: page budget, exact/approximate top-\(k\), layer skipping, KV precision, backend, and memory configuration. Each batch contains four orthogonal variants, including at least one novel algorithm and controlled sweeps around it. |
| **Selection criteria** | Candidates must satisfy the novelty rule and pass `check_engine_config`. Optional offline routing-recall tests can reject weak ideas. No downstream accuracy or throughput is exposed in this stage. | Candidates first pass `check_engine_config`, then a **0.85 RULER** quality gate. Surviving variants are evaluated using AIME24 `mean@16` and decoding throughput. Selection uses the non-dominated Pareto frontier rather than a scalar reward or post-hoc threshold. |
| **Human intervention** | The human only starts the command. Hypothesis generation, implementation, repair, and candidate output are autonomous; there is no human filtering or editing. | The human only starts the command. Candidate design, implementation, repair, advancement, evaluation, failure logging, Pareto analysis, and subsequent iterations are autonomous, without human inspection or steering of intermediate results. |

Innovate therefore separates open-ended generation from benchmark feedback,
while Iterate performs measurement-guided refinement. Every candidate is saved
as Python source plus an engine configuration, and Iterate records hypotheses,
configuration hashes, results, failures, and Pareto decisions in
`algorithm_scientist/memory.md`. This makes the exploration trace reproducible
and auditable rather than leaving it only in the agent's conversation history.
