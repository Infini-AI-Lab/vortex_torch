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
import os
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
kv_cache_dtype: str = "auto"
):  

    llm = sgl.Engine(model_path=model_name, 
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

    for item in results:
        total_accuracy += item['score']
        count += 1
        total_tokens += item["num_tokens"]
        e2e_time = max(e2e_time, item["e2e_latency"])
        if item['query'] not in unique_result:
            unique_result[item['query']] = item["score"]
        else:
            unique_result[item['query']] = max(item["score"], unique_result[item['query']])


    global_summary = {
        f'mean@{trials}': total_accuracy / count if count > 0 else 0,
        f'pass@{trials}': sum(unique_result.values()) / len(unique_result),
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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

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
        kv_cache_dtype=args.kv_cache_dtype
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"summary_fp8/{args.model_name.replace('/', '_')}_{args.vortex_module_name}_{args.trials}trials_tp{args.tp_size}_{current_time}.json"
    os.makedirs("summary_fp8", exist_ok=True)
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
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    