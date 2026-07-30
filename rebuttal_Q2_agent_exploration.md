# Response to Q2: Reproducibility of AI-agent-assisted exploration

We thank the reviewer and agree that these details should be explicit. We will
add the following description and release the complete executable prompts.

| Aspect | Protocol |
|---|---|
| **Prompt/template** | The verbatim prompts are provided in `.claude/commands/{innovate,iterate}.md`, with the programming contract and available operations in `AI/AGENTS.md` and `AI/tutorials/`. Each iteration asks the agent to propose four orthogonal variants, state their hypotheses, implement them, validate them, benchmark them, and use the recorded results to design the next batch. |
| **Search budget** | The reported long-horizon run used **23 iterations × 4 variants = 92 candidates**. Each batch reserved at least one slot for a new algorithmic hypothesis; the remaining slots performed controlled sweeps around it. The iteration count, model, task, and GPU budget were fixed when the command was launched. |
| **Failed cases** | Invalid programs were rejected by `check_engine_config`. Candidates scoring below **0.85 on RULER** were repaired or rejected before the expensive AIME24 evaluation. Compiling but ineffective candidates—including Pareto-dominated routing rules and parameter settings—were retained in `algorithm_scientist/memory.md` under completed results, anti-patterns, and measured non-levers. Thus, unsuccessful trials are preserved rather than omitted from subsequent agent context. |
| **Selection criteria** | Final selection used the non-dominated frontier of AIME24 `mean@16` and decoding throughput; no scalar reward or post-hoc accuracy threshold was used. A candidate was considered an improvement only if it extended the running Pareto frontier relative to prior candidates and the full-attention baseline. |
| **Human involvement** | The human only started the command. Hypothesis generation, implementation, repair, candidate advancement, evaluation, failure logging, and Pareto selection were performed autonomously, without human filtering of intermediate results. |

These additions make the search trace auditable: every candidate is
materialized as source code and configuration, while hypotheses, failures,
configuration hashes, measurements, and selection decisions are recorded in
the persistent experiment ledger.
