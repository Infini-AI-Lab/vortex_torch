import sglang as sgl
import vortex_torch
import argparse
import json
import os
from datetime import datetime


def run(
    trials: int = 2,
    topk_val: int = 29,
    topk_ratio: float = 0.0625,
    block_size: int = 16,
    page_size: int = 256,
    workload_chunk_size: int = 32,
    max_input_length: int = 4096,
    generation_max_new_tokens: int = 8192,
    vortex_module_name: str = "gqa_block_sparse_attention",
    model_name: str = "Qwen/Qwen3-1.7B",
    sparse_attention: bool = True,
    mem: float = 0.8,
    data_path: str = "examples/code/lcbv5.jsonl",
    tp_size: int = 1,
    kv_cache_dtype: str = "auto",
):
    llm = sgl.Engine(
        model_path=model_name,
        disable_cuda_graph=False,
        vortex_block_size=block_size,
        page_size=page_size,
        vortex_topk_val=topk_val,
        tp_size=tp_size,
        kv_cache_dtype=kv_cache_dtype,
        disable_overlap_schedule=True,
        attention_backend="flashinfer",
        enable_vortex_sparsity=sparse_attention,
        vortex_block_reserved_bos=1,
        vortex_block_reserved_eos=2,
        vortex_topk_ratio=topk_ratio,
        vortex_layers_skip=list(range(1)),
        vortex_module_name=vortex_module_name,
        vortex_max_seq_lens=max_input_length + generation_max_new_tokens,
        mem_fraction_static=mem,
        vortex_workload_chunk_size=max(page_size // block_size, workload_chunk_size),
        vortex_compilation_cache_dir="~/.vortex_compilation_cache",
        context_length=40960,
    )

    with open(data_path, "r", encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]

    requests = requests * trials
    prompts = [req["prompt"] for req in requests]

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": generation_max_new_tokens,
    }

    o = llm.generate(prompts, sampling_params)

    raw_outputs = []
    total_tokens = 0
    e2e_time = 0.0
    for req, item in zip(requests, o):
        total_tokens += item["meta_info"]["completion_tokens"]
        e2e_time = max(e2e_time, item["meta_info"]["e2e_latency"])
        raw_outputs.append({
            "id": req.get("id"),
            "question_id": req.get("question_id"),
            "difficulty": req.get("difficulty"),
            "prompt": req["prompt"],
            **item,
        })

    summary = {
        "total_example": len(raw_outputs),
        "e2e_time": e2e_time,
        "total_tokens": total_tokens,
        "throughput": total_tokens / e2e_time if e2e_time > 0 else 0.0,
    }
    return summary, raw_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vortex_torch generation on LCB-style prompts and dump raw outputs."
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--topk-val", type=int, default=29)
    parser.add_argument("--topk-ratio", type=float, default=0.0625)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--workload-chunk-size", type=int, default=32)
    parser.add_argument("--generation-max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--vortex-module-name", type=str, default="gqa_block_sparse_attention")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("-f", "--full-attention", action="store_true")
    parser.add_argument("--mem", type=float, default=0.8)
    parser.add_argument("--data-path", type=str, default="examples/code/lcbv5.jsonl")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--summary-dir", type=str, default="summary_lcb")
    return parser.parse_args()


def already_finished(summary_dir, args):
    if not os.path.isdir(summary_dir):
        return None
    for fname in os.listdir(summary_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(summary_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        prev_args = prev.get("args", {})
        if (
            prev_args.get("trials") == args.trials
            and prev_args.get("topk_val") == args.topk_val
            and prev_args.get("topk_ratio") == args.topk_ratio
            and prev_args.get("vortex_module_name") == args.vortex_module_name
            and prev_args.get("model_name") == args.model_name
            and prev_args.get("data_path") == args.data_path
            and prev_args.get("tp_size") == args.tp_size
        ):
            return path
    return None


if __name__ == "__main__":
    args = parse_args()

    existing = already_finished(args.summary_dir, args)
    if existing is not None:
        print(
            f"Skip: already finished "
            f"(trials={args.trials}, topk_val={args.topk_val}, topk_ratio={args.topk_ratio}, "
            f"module={args.vortex_module_name}, model={args.model_name}, "
            f"data={args.data_path}) -> {existing}"
        )
        raise SystemExit(0)

    summary, raw_outputs = run(
        trials=args.trials,
        topk_val=args.topk_val,
        topk_ratio=args.topk_ratio,
        block_size=args.block_size,
        page_size=args.page_size,
        workload_chunk_size=args.workload_chunk_size,
        generation_max_new_tokens=args.generation_max_new_tokens,
        max_input_length=args.max_input_length,
        vortex_module_name=args.vortex_module_name,
        model_name=args.model_name,
        sparse_attention=(args.vortex_module_name != "full_attention"),
        mem=args.mem,
        data_path=args.data_path,
        tp_size=args.tp_size,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.summary_dir, exist_ok=True)
    output_path = os.path.join(
        args.summary_dir,
        f"{args.model_name.replace('/', '_')}_{args.vortex_module_name}_{args.trials}trials_tp{args.tp_size}_{current_time}.json",
    )
    summary["args"] = {
        "trials": args.trials,
        "topk_val": args.topk_val,
        "topk_ratio": args.topk_ratio,
        "block_size": args.block_size,
        "page_size": args.page_size,
        "workload_chunk_size": args.workload_chunk_size,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "max_input_length": args.max_input_length,
        "vortex_module_name": args.vortex_module_name,
        "model_name": args.model_name,
        "sparse_attention": (args.vortex_module_name != "full_attention"),
        "mem": args.mem,
        "data_path": args.data_path,
        "tp_size": args.tp_size,
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    outputs_path = output_path[:-len(".json")] + ".outputs.jsonl"
    with open(outputs_path, "w", encoding="utf-8") as f:
        for res in raw_outputs:
            json.dump(res, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote summary to {output_path}")
    print(f"Wrote {len(raw_outputs)} raw outputs to {outputs_path}")
