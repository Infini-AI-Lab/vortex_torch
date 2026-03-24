import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import time
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

from datasets import load_dataset, Dataset, concatenate_datasets
def generate_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
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
    for data in dataset:
        conversations = [
                {"role": "user", "content": data_format.format(Question=data[field_name])}
        ]
        data["conversations"] = conversations
        requests.append(data)

    return requests


def main():
      model_name = "Qwen/Qwen3-0.6B"
      llm = sgl.Engine(model_path=model_name,
                              disable_cuda_graph=False,
                              page_size=16,
                              vortex_num_selected_pages=29,
                              disable_overlap_schedule=True,
                              attention_backend="flashinfer",
                              enable_vortex_sparsity=True,
                              vortex_page_reserved_bos=1,
                              vortex_page_reserved_eos=2,
                              vortex_layers_skip=list(range(1)),
                              mem_fraction_static=0.9,
                              vortex_cg=True,
                              vortex_graph=True,
                              vortex_module_name="block_sparse_attention",
                              vortex_max_seq_lens=20480
                              )

      dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

      requests = generate_requests(dataset, "problem", MATH_QUERY_TEMPLATE)



      texts = [
            x["conversations"] for x in requests
      ]

      tokenizer = AutoTokenizer.from_pretrained(model_name)
      prompts = [
            tokenizer.apply_chat_template(
            text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
      ) for text in texts
      ] * 8

      sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 16384}
      total_tokens = 0
      total_time = 0.0
      start = time.perf_counter()
      o = llm.generate(prompts, sampling_params)
      elapsed = time.perf_counter() - start
      total_time += elapsed
      e2e_time = 0
      with open(f"0.6B_VTX_CG_TP1_16K.jsonl", "w", encoding="utf-8") as f:
            for item in o:
                  total_tokens += item["meta_info"]["completion_tokens"]
                  e2e_time = max(e2e_time, item["meta_info"]["e2e_latency"])
                  json.dump(item, f, ensure_ascii=False)
                  f.write("\n")

            meta_data = {"e2e_time": e2e_time, "total_time": total_time, "total_tokens": total_tokens, "throughput": total_tokens / total_time}
            json.dump(meta_data, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
      main()