"""Greedy forward-selection of layers whose indexer can be skipped (index cache).

Usage (from repo root):
    cd examples && python ../benchmarks/greedy_layer_search.py \
        --model-name Qwen/Qwen3-1.7B --topk-val 30 --threshold 0.95 \
        --trials 1 --num-layers 28 --mem 0.7

The script prints progress to stderr and outputs the final selected layer list
(as a Python list literal) on the **last line of stdout** so callers can parse it.
"""

import argparse
import os
import sys

# Add examples/ to path so we can import verify_algos
_examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
sys.path.insert(0, _examples_dir)

from verify_algo import verify_algos  # noqa: E402


def _evaluate(shared_layers, args):
    """Run verify_algos with the given shared layers and return pass@trials accuracy."""
    summary = verify_algos(
        trials=args.trials,
        topk_val=args.topk_val,
        page_size=args.page_size,
        vortex_module_name=args.vortex_module_name,
        model_name=args.model_name,
        sparse_attention=True,
        mem=args.mem,
        kv_cache_dtype=args.kv_cache_dtype,
        topk_type=args.topk_type,
        topk_mapping_mode=0,
        topk_mapping_power=args.topk_mapping_power,
        index_cache_shared_layers=sorted(shared_layers) if shared_layers else None,
        disable_cuda_graph=True,
    )
    acc_key = f"pass@{args.trials}"
    return summary[acc_key]


def greedy_search(args):
    # Ensure we're in examples/ so amc23.jsonl relative path works
    os.chdir(_examples_dir)

    candidates = list(range(1, args.num_layers))

    # Baseline: no shared layers
    print("Evaluating baseline (no shared layers)...", file=sys.stderr)
    baseline_acc = _evaluate([], args)
    print(f"Baseline accuracy: {baseline_acc:.4f}", file=sys.stderr)

    threshold = args.threshold
    shared_set = []

    while candidates:
        best_layer = None
        best_acc = -1.0

        for layer in candidates:
            trial_set = shared_set + [layer]
            print(f"  Trying shared_set={sorted(trial_set)} ...", file=sys.stderr, end=" ")
            acc = _evaluate(trial_set, args)
            print(f"acc={acc:.4f}", file=sys.stderr)

            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        if best_acc >= threshold * baseline_acc:
            shared_set.append(best_layer)
            candidates.remove(best_layer)
            print(
                f"Added layer {best_layer} (acc={best_acc:.4f} >= "
                f"{threshold * baseline_acc:.4f}). Current set: {sorted(shared_set)}",
                file=sys.stderr,
            )
        else:
            print(
                f"Stopping: best candidate layer {best_layer} acc={best_acc:.4f} < "
                f"{threshold * baseline_acc:.4f}",
                file=sys.stderr,
            )
            break

    result = sorted(shared_set)
    print(f"Final shared layers: {result}", file=sys.stderr)
    # Last stdout line: parseable Python list
    print(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy forward-selection of index-cache shared layers."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--topk-val", type=int, default=30)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--vortex-module-name", type=str, default="block_sparse_attention")
    parser.add_argument("--mem", type=float, default=0.8)
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--topk-type", type=str, default="naive")
    parser.add_argument("--topk-mapping-power", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Minimum accuracy ratio vs baseline to keep adding layers (default: 0.95).")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Total number of model layers (default: 28 for Qwen3-1.7B).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    greedy_search(args)
