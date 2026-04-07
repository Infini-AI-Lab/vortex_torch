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
import ast
import json
import os
import subprocess
import sys

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

def generate_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    total = len(dataset)
    
    # Step 2: Partition across ranks
    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)
    subset = dataset.select(list(range(start, end)))

    # Step 3: Format requests
    for data in subset:
        conversations = [
            {"role": "user", "content": data_format.format(Question=data[field_name])}
        ]
        data["conversations"] = conversations
        requests.append(data)

    return requests

BENCHMARK_REGISTRY = {
    "amc23": {
        "type": "jsonl",
        "path": "amc23.jsonl",
        "prompt_key": "prompt",
        "answer_key": "answer",
        "question_key": "question",
    },
    "aime24": {
        "type": "huggingface",
        "path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "field_name": "problem",
        "answer_key": "answer",
    },
}

def _load_benchmark(benchmark_name: str, trials: int, tokenizer=None):
    """Load benchmark data and return (prompts, requests) tuple."""
    cfg = BENCHMARK_REGISTRY[benchmark_name]

    if cfg["type"] == "jsonl":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_path = os.path.join(script_dir, cfg["path"])
        with open(jsonl_path, "r", encoding="utf-8") as f:
            requests = [json.loads(line) for line in f]
        requests = requests * trials
        prompts = [req[cfg["prompt_key"]] for req in requests]
        return prompts, requests

    elif cfg["type"] == "huggingface":
        dataset = load_dataset(cfg["path"], split=cfg["split"])
        hf_requests = generate_requests(dataset, cfg["field_name"], MATH_QUERY_TEMPLATE)
        # Normalize keys: ensure "question" and "answer" exist
        for req in hf_requests:
            if "question" not in req and cfg["field_name"] in req:
                req["question"] = req[cfg["field_name"]]
        # Build chat-template prompts if tokenizer is provided
        if tokenizer is not None:
            texts = [x["conversations"] for x in hf_requests]
            prompts = [
                tokenizer.apply_chat_template(
                    text, tokenize=False, add_generation_prompt=True, enable_thinking=True
                ) for text in texts
            ] * trials
            hf_requests = hf_requests * trials
        else:
            prompts = [
                MATH_QUERY_TEMPLATE.format(Question=x[cfg["field_name"]]) for x in hf_requests
            ] * trials
            hf_requests = hf_requests * trials
        return prompts, hf_requests

    else:
        raise ValueError(f"Unknown benchmark type: {cfg['type']}")


def verify_algos(
trials: int = 2,
topk_val: int = 30,
page_size: int = 16,
vortex_module_name: str = "gqa_block_sparse_attention",
model_name: str = "Qwen/Qwen3-1.7B",
sparse_attention: bool = True,
mem: float = 0.8,
kv_cache_dtype: str = "auto",
topk_type: str = "naive",
topk_mapping_mode: int = 0,
topk_mapping_power: float = 0.5,
topk_mapping_lut_path: str = None,
topk_mapping_quantiles_path: str = None,
index_cache_shared_layers: list = None,
disable_cuda_graph: bool = False,
benchmark: str = "amc23",
):

    llm = sgl.Engine(model_path=model_name,
                    disable_cuda_graph=disable_cuda_graph,
                    page_size=page_size,
                    vortex_topk_val=topk_val,
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=sparse_attention,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=2,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name=vortex_module_name,
                    vortex_max_seq_lens=12288,
                    mem_fraction_static=mem,
                    kv_cache_dtype=kv_cache_dtype,
                    vortex_topk_type=topk_type,
                    vortex_topk_mapping_mode=topk_mapping_mode,
                    vortex_topk_mapping_power=topk_mapping_power,
                    vortex_topk_mapping_lut_path=topk_mapping_lut_path,
                    vortex_topk_mapping_quantiles_path=topk_mapping_quantiles_path,
                    vortex_index_cache_shared_layers=index_cache_shared_layers,
                    )
    tokenizer = AutoTokenizer.from_pretrained(model_name) if benchmark != "amc23" else None
    prompts, requests = _load_benchmark(benchmark, trials, tokenizer=tokenizer)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 8192}

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
        # --- Per-question debug output  ---
        # print(f"[Q{len(results):03d}] score={float(result):.1f} "
        #       f"tokens={item['meta_info']['completion_tokens']} "
        #       f"latency={item['meta_info']['e2e_latency']:.2f}s "
        #       f"gold={golds[0]}")
        # print(f"  question: {data['question'][:120]}...")
        # print(f"  prediction: {predictions[:200]}...")
        # print()


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

    if sparse_attention:
        llm_cfg = AutoConfig.from_pretrained(model_name)
        flow = vortex_torch.flow.build_vflow(vortex_module_name)
        try:
            memory_access_runtime = flow.run_indexer_virtual(
                group_size=llm_cfg.num_attention_heads // llm_cfg.num_key_value_heads,
                page_size=page_size,
                head_dim=llm_cfg.head_dim,
            )
        except Exception:
            # External algorithms (nsa, fsa, flash_moba) override run_indexer_virtual
            # to return 0 since their vendored kernels don't participate in vortex profiling
            memory_access_runtime = 0.0
    else:
        memory_access_runtime = 0.0
    
    global_summary = {
        f'mean@{trials}': total_accuracy / count if count > 0 else 0,
        f'pass@{trials}': sum(unique_result.values()) / len(unique_result),
        'total_example': count,
        "e2e_time": e2e_time,
        "total_tokens": total_tokens, 
        "throughput": total_tokens / e2e_time,
        "auxilary memory_access_runtime (bytes per page)": memory_access_runtime
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
        default=30,
        help="Top-k value to use in the algorithm (default: 30).",
    )
    
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Page Size for Sglang (default: 16).",
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
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8_e5m2", "fp8_e4m3", "int8"],
        help='KV cache dtype (default: "auto").',
    )

    parser.add_argument(
        "--topk-type",
        type=str,
        default="naive",
        choices=["naive", "sglang", "sglang_ori"],
        help='TopK kernel type: "naive" for topk_output, "sglang" for topk_output_sglang, "sglang_ori" for original sglang baseline (default: "naive").',
    )
    parser.add_argument(
        "--topk-mapping-mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help='TopK mapping mode: 0=none, 1=lut_cdf, 2=quantile, 3=power, 4=log, 5=index_cache, 6=asinh, 7=log1p, 8=trunc8, 9=erf, 10=tanh, 11=subtract, 12=adaptive_tail_window, 13=exp_stretch, 14=topk_window (default: 0).',
    )

    parser.add_argument(
        "--topk-mapping-power",
        type=float,
        default=0.5,
        help='Hyperparameter for parametric modes: power exponent (mode 3), beta (mode 6 asinh), alpha (mode 7 log1p), rho tail expansion (mode 12). Default: 0.5.',
    )

    parser.add_argument(
        "--topk-mapping-lut-path",
        type=str,
        default=None,
        help="Path to .npy file with uint8[256] LUT for topk mapping mode 1.",
    )

    parser.add_argument(
        "--topk-mapping-quantiles-path",
        type=str,
        default=None,
        help="Path to .npy file with float32[256] quantiles for topk mapping mode 2.",
    )

    parser.add_argument(
        "--index-cache-shared-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer IDs that reuse indices from the nearest preceding full layer (skip indexer).",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        default=["amc23"],
        help="Benchmark(s) to run. Available: amc23, aime24. "
             "Use multiple values to run several benchmarks sequentially (default: amc23).",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --- Mode 5: Index Cache (default even-layer pattern) ---
    if args.topk_mapping_mode == 5:
        if args.index_cache_shared_layers is None:
            args.index_cache_shared_layers = list(range(2, 28, 2))  # [2,4,6,...,26]
        args.topk_mapping_mode = 0

    for bench_name in args.benchmark:
        if bench_name not in BENCHMARK_REGISTRY:
            print(f"WARNING: Unknown benchmark '{bench_name}', skipping. Available: {list(BENCHMARK_REGISTRY.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")
        summary = verify_algos(
            trials=args.trials,
            topk_val=args.topk_val,
            page_size=args.page_size,
            vortex_module_name=args.vortex_module_name,
            model_name=args.model_name,
            sparse_attention=not(args.full_attention),
            mem=args.mem,
            kv_cache_dtype=args.kv_cache_dtype,
            topk_type=args.topk_type,
            topk_mapping_mode=args.topk_mapping_mode,
            topk_mapping_power=args.topk_mapping_power,
            topk_mapping_lut_path=args.topk_mapping_lut_path,
            topk_mapping_quantiles_path=args.topk_mapping_quantiles_path,
            index_cache_shared_layers=args.index_cache_shared_layers,
            benchmark=bench_name,
        )
        summary["benchmark"] = bench_name
        print(summary)

    exit(0)