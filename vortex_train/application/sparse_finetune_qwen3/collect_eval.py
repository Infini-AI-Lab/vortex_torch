"""Merge the sharded eval JSONs into one trained-vs-untrained comparison table.

``run_eval.sh`` runs the four (task, model) combinations as separate processes on
separate GPUs, so each writes its own JSON. This joins them and reports the delta that
the whole exercise is for.

    python -m application.sparse_finetune_qwen3.collect_eval /scratch/zhuominc/eval_1k
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/zhuominc/eval_1k")
    rows = []
    selection = None
    for f in sorted(out_dir.glob("*.json")):
        d = json.loads(f.read_text())
        selection = selection or d.get("selection")
        rows.extend(d.get("results", []))

    if not rows:
        print(f"no results in {out_dir}")
        sys.exit(1)

    if selection:
        total = (selection["vortex_topk_val"] + selection["vortex_block_reserved_bos"]
                 + selection["vortex_block_reserved_eos"])
        print(f"budget: block_size={selection['vortex_block_size']}, "
              f"topk_val={selection['vortex_topk_val']} + "
              f"bos={selection['vortex_block_reserved_bos']} + "
              f"eos={selection['vortex_block_reserved_eos']} = {total} blocks "
              f"({total * selection['vortex_block_size']} KV tokens)")
        print(f"trained with block_q={selection.get('train_block_q')}\n")

    key = next(k for k in rows[0] if k.startswith("mean@"))
    print(f"{'task':>8} {'model':>10} {key:>10} {'pass@k':>8} {'tokens':>10} {'tok/s':>8}")
    print("-" * 60)
    for r in sorted(rows, key=lambda r: (r["task"], r["model"])):
        print(f"{r['task']:>8} {r['model']:>10} {r[key]:>10.4f} {r['pass@k']:>8.4f} "
              f"{r['completion_tokens']:>10} {r['tok_per_s']:>8.0f}")

    print()
    tasks = sorted({r["task"] for r in rows})
    deltas = []
    for task in tasks:
        got = {r["model"]: r for r in rows if r["task"] == task}
        if "trained" in got and "base" in got:
            b, t = got["base"][key], got["trained"][key]
            deltas.append(t - b)
            print(f"{task}: base {b:.4f} -> trained {t:.4f}  ({t - b:+.4f})")
        else:
            print(f"{task}: incomplete (have {sorted(got)})")
    if len(deltas) == len(tasks) and deltas:
        print(f"\nmean delta across {len(tasks)} tasks: {sum(deltas)/len(deltas):+.4f}")
        # 30 problems x k trials is a small sample; say so rather than over-read it.
        n = rows[0]["problems"]
        print(f"note: {n} problems per task at {rows[0]['trials']} trials — a "
              f"{1/n:.3f} change is one problem, so small deltas are not resolvable.")


if __name__ == "__main__":
    main()
