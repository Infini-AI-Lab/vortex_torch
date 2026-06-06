import sglang as sgl
import vortex_torch
from transformers import AutoTokenizer, AutoConfig
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.models.model_output import ModelResponse
from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
def verify_algos(
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
data_path: str = "examples/amc23.jsonl",
tp_size: int = 1,
kv_cache_dtype: str = "auto",
attention_backend: str = "flashinfer",
vortex_attention_backend: str = "trtllm",
vortex_impl_backend: str = "cuda",
vortex_use_tensor_core: bool = False,
vortex_layers_skip: list = None,
vortex_module_path: str = None,
disable_cuda_graph: bool = False,
):
    if vortex_layers_skip is None:
        vortex_layers_skip = [0]

    llm = sgl.Engine(model_path=model_name,
                    disable_cuda_graph=disable_cuda_graph,
                    vortex_module_path=vortex_module_path,
                    vortex_block_size=block_size,
                    page_size=page_size,
                    vortex_topk_val=topk_val,
                    vortex_max_topk_val=256,
                    tp_size=tp_size,
                    kv_cache_dtype=kv_cache_dtype,
                    disable_overlap_schedule=False,
                    attention_backend=attention_backend,
                    vortex_attention_backend=vortex_attention_backend,
                    vortex_impl_backend=vortex_impl_backend,
                    vortex_use_tensor_core=vortex_use_tensor_core,
                    enable_vortex_sparsity=sparse_attention,
                    vortex_block_reserved_bos=1,
                    vortex_block_reserved_eos=2,
                    vortex_topk_ratio=topk_ratio,
                    vortex_layers_skip=vortex_layers_skip,
                    vortex_module_name=vortex_module_name,
                    vortex_max_seq_lens=max_input_length + generation_max_new_tokens,
                    mem_fraction_static=mem,
                    vortex_workload_chunk_size=max(page_size // block_size, workload_chunk_size),
                    vortex_compilation_cache_dir="~/.vortex_compilation_cache",
                    context_length=40960,
                    trust_remote_code=True,
                    )
    
    with open(data_path, "r", encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    
    requests = requests * trials
    prompts = [req["prompt"] for req in requests]

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": generation_max_new_tokens}
    
    o = llm.generate(prompts, sampling_params)
    gold_metric =  MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )
    
    results = []
    for data, item in zip(requests, o):
        golds = [data["answer"]]
        target = Doc(query=data["question"],choices=golds, gold_index=0)
        predictions = item["text"]
        try:
            result = gold_metric.compute(model_response=ModelResponse(text=[predictions]), doc=target)
        except:
            result = 0.0
        
        results.append(
            {
                "score": float(result),
                "prediction": [predictions],
                "choices": golds,
                "query": data["question"],
                "e2e_latency": item["meta_info"]["e2e_latency"],
                "num_tokens": item["meta_info"]["completion_tokens"]
            }
        )
    

    total_accuracy = 0.0
    total_tokens = 0
    e2e_time = 0
    count = 0
    unique_result = {}
    per_question_scores = defaultdict(list)

    for item in results:
        total_accuracy += item['score']
        count += 1
        total_tokens += item["num_tokens"]
        e2e_time = max(e2e_time, item["e2e_latency"])
        if item['query'] not in unique_result:
            unique_result[item['query']] = item["score"]
        else:
            unique_result[item['query']] = max(item["score"], unique_result[item['query']])
        per_question_scores[item['query']].append(item['score'])

    def pass_at_k(n: int, c: int, k: int) -> float:
        """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
        if k > n:
            return float('nan')
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    pass_at_k_summary = {}
    for k in (4, 8):
        vals = []
        for scores in per_question_scores.values():
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)
            if k <= n:
                vals.append(pass_at_k(n, c, k))
        if vals:
            pass_at_k_summary[f'pass@{k}'] = sum(vals) / len(vals)

    global_summary = {
        f'mean@{trials}': total_accuracy / count if count > 0 else 0,
        f'pass@{trials}': sum(unique_result.values()) / len(unique_result),
        **pass_at_k_summary,
        'total_example': count,
        "e2e_time": e2e_time,
        "total_tokens": total_tokens,
        "throughput": total_tokens / e2e_time,
    }
    
    return global_summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vortex_torch verify_algos benchmark."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="Number of trials to run (default: 2).",
    )

    parser.add_argument(
        "--topk-val",
        type=int,
        default=29,
        help="Top-k value to use in the algorithm (default: 30).",
    )
    
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=0.0625,
        help="Top-k ratio to use in the algorithm (default: 0.0625).",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block Size for Sglang (default: 16).",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="Page Size for Sglang (default: 256).",
    )

    parser.add_argument(
        "--workload-chunk-size",
        type=int,
        default=32,
        help="Workload Chunk Size for Sglang (default: 32).",
    )

    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=8192,
        help="Max new tokens to generate (default: 8192).",
    )

    parser.add_argument(
        "--max-input-length",
        type=int,
        default=4096,
        help="Max input tokens (default: 4096).",
    )

    parser.add_argument(
        "--vortex-module-name",
        type=str,
        default="gqa_block_sparse_attention",
        help='Name of the vortex module to test (default: "gqa_block_sparse_attention").',
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help='HuggingFace model name to load (default: "Qwen/Qwen3-1.7B").',
    )

    parser.add_argument(
        "-f", "--full-attention",
        action="store_true",
        help="Use full attention instead of vortex sparse attention.",
    )

    parser.add_argument(
        "--mem",
        type=float,
        default=0.8,
        help="memory fraction in sglang",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="examples/amc23.jsonl",
        help="Path to the evaluation data (default: examples/amc23.jsonl).",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for Sglang (default: 1).",
    )

    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        help="Data type for KV cache (default: auto).",
    )

    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flashinfer",
        help="sglang attention backend selecting the decode path. For MLA + vortex "
             "sparsity use 'cuda_mla' (hand-CUDA decode), 'triton' (Triton decode), "
             "or 'trtllm_mla' (DeepSeek-only). 'flashinfer' is the dense MLA path "
             "(use for the full_attention baseline).",
    )

    parser.add_argument(
        "--vortex-attention-backend",
        type=str,
        default="trtllm",
        choices=["flashinfer", "trtllm"],
        help="Vortex sparse-attention block-table planner path (default: trtllm).",
    )

    parser.add_argument(
        "--vortex-impl-backend",
        type=str,
        default="cuda",
        choices=["triton", "cuda"],
        help="Vortex indexer compilation backend (default: cuda).",
    )

    parser.add_argument(
        "--vortex-use-tensor-core",
        action="store_true",
        help="Triton indexer: keep compute blocks in bf16 (fp32 "
             "accumulation) and emit tl.dot for MMA-friendly GeMMs. "
             "Requires --vortex-impl-backend triton.",
    )

    parser.add_argument(
        "--vortex-layers-skip",
        type=int,
        nargs="*",
        default=[0],
        help="Decoder layer indices that run dense (non-sparse) "
             "attention (default: 0). Pass the flag with no values "
             "(--vortex-layers-skip) to skip none (all layers sparse).",
    )

    parser.add_argument(
        "--vortex-module-path",
        type=str,
        default=None,
        help="Path to a .py file declaring @register(<vortex_module_name>) "
             "(for custom flows like the channel-study variants).",
    )

    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph capture (eager decode). Use when a custom "
             "flow trips graph capture.",
    )

    parser.add_argument(
        "--summary-dir",
        type=str,
        default="summary_ratio",
        help="Directory to read prior summaries from and write this run's summary to (default: summary_ratio).",
    )

    parser.add_argument(
        "--skip-already-finished-check",
        action="store_true",
        help="Skip checking for prior matching summaries and always run.",
    )

    return parser.parse_args()

def already_finished(summary_dir, args):
    """Return path of a prior summary matching this run's key, or None."""
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
            and prev_args.get("block_size") == args.block_size
            and prev_args.get("page_size") == args.page_size
            and prev_args.get("topk_ratio") == args.topk_ratio
            and prev_args.get("block_size") == args.block_size
            and prev_args.get("page_size") == args.page_size
            and prev_args.get("vortex_module_name") == args.vortex_module_name
            and prev_args.get("model_name") == args.model_name
            and prev_args.get("data_path") == args.data_path
            and prev_args.get("tp_size") == args.tp_size
        ):
            return path
    return None


if __name__ == "__main__":
    args = parse_args()

    existing = None if args.skip_already_finished_check else already_finished(args.summary_dir, args)
    if existing is not None:
        print(
            f"Skip: already finished "
            f"(trials={args.trials}, topk_ratio={args.topk_ratio}, "
            f"module={args.vortex_module_name}, model={args.model_name}, "
            f"data={args.data_path}) -> {existing}"
        )
        raise SystemExit(0)

    summary = verify_algos(
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
        attention_backend=args.attention_backend,
        vortex_attention_backend=args.vortex_attention_backend,
        vortex_impl_backend=args.vortex_impl_backend,
        vortex_use_tensor_core=args.vortex_use_tensor_core,
        vortex_layers_skip=args.vortex_layers_skip,
        vortex_module_path=args.vortex_module_path,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.summary_dir, exist_ok=True)
    output_path = os.path.join(
        args.summary_dir,
        f"{args.model_name.replace('/', '_')}_{args.vortex_module_name}_{args.trials}trials_tp{args.tp_size}_{current_time}.json",
    )
    ## args to summary
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
        "attention_backend": args.attention_backend,
        "vortex_attention_backend": args.vortex_attention_backend,
        "vortex_impl_backend": args.vortex_impl_backend,
        "vortex_use_tensor_core": args.vortex_use_tensor_core,
        "vortex_layers_skip": args.vortex_layers_skip,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    