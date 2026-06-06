import json
import os
import sys

# --- Speed up the vortex/flashinfer JIT (get_decode_planner / prefill kernels) ---
# Must run BEFORE torch is imported (transformers below pulls it in).
# 1) Compile only for THIS GPU's compute capability. With TORCH_CUDA_ARCH_LIST
#    unset, torch's load_inline builds a fat binary for the whole default arch
#    list (+PTX) — the usual cause of 10-15 min nvcc compiles. Pinning the single
#    arch cuts it to one gencode.
# 2) Persist the build caches so only the FIRST run compiles; later runs reuse.
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    try:
        import subprocess
        _cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0].strip()
        if _cap:
            os.environ["TORCH_CUDA_ARCH_LIST"] = _cap   # e.g. "10.0", "9.0", "8.0"
    except Exception:
        pass
os.environ.setdefault("TORCH_EXTENSIONS_DIR",
                      os.path.expanduser("~/.cache/torch_extensions"))

from transformers import AutoTokenizer


def _generate_server(url, prompts, sampling_params):
    """Server mode: POST the batch to a running sglang server's native
    ``/generate`` endpoint (the HTTP analogue of ``Engine.generate``).
    Returns a list of ``{"text": ...}`` dicts, same shape as the offline
    engine, so the accuracy loop below is identical for both modes."""
    import requests

    url = url.rstrip("/")
    resp = requests.post(
        f"{url}/generate",
        json={"text": prompts, "sampling_params": sampling_params},
        timeout=3600,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    # $1: HF model id (positional, optional). Default: Qwen/Qwen3-4B.
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"

    # Server mode: when RULER_SERVER_URL is set, skip building an in-process
    # Engine and drive an already-running sglang server (e.g. the one started
    # by examples/server_launch.sh) over HTTP. The server's own launch flags
    # (topk_val, block_size, module, etc.) define the sparse-attention config;
    # this script only feeds prompts and scores the answers.
    server_url = os.environ.get("RULER_SERVER_URL")

    # Baseline toggle: ENABLE_VORTEX_SPARSITY=0 runs dense sglang (no vortex
    # sparse path) to confirm the reference accuracy; default (1) is sparse.
    # (Offline-engine mode only — server mode inherits the server's config.)
    enable_vortex_sparsity = os.environ.get("ENABLE_VORTEX_SPARSITY", "1") == "1"
    if server_url:
        print(f"[run_ruler] SERVER mode via {server_url}", flush=True)
    else:
        print(f"[run_ruler] OFFLINE engine mode, "
              f"enable_vortex_sparsity={enable_vortex_sparsity}", flush=True)

    default_policy = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""

    llm = None
    if not server_url:
        import sglang as sgl
        import vortex_torch  # noqa: F401  (wires sglang integration)
        llm = sgl.Engine(model_path=model_name,
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_block_size=16,
                    vortex_topk_val=29,
                    disable_overlap_schedule=False,
                    kv_cache_dtype="auto",
                    vortex_dtype="bfloat16",
                    attention_backend="flashinfer",
                    vortex_schedule_policy=default_policy,
                    enable_vortex_sparsity=enable_vortex_sparsity,
                    vortex_block_reserved_bos=1,
                    vortex_block_reserved_eos=2,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name="gqa_block_sparse_attention",
                    vortex_attention_backend="trtllm",
                    trust_remote_code=True,
                    #vortex_module_path="submissions/example_block_sparse_attention.py",
                    vortex_max_seq_lens=8192,
                    mem_fraction_static=0.9,
                    vortex_workload_chunk_size=32,
                    vortex_compilation_cache_dir=os.path.expanduser("~/.vortex_compilation_cache"),
                    tp_size=1,
                    )
    
    with open("examples/validation.jsonl", "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line)["input"] for line in f]

    with open("examples/validation.jsonl", "r", encoding="utf-8") as f:
        ruler_outputs = [json.loads(line)["outputs"][0] for line in f]
    
    texts = [
        [{"role":"user","content": x}] for x in ruler_data
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for text in texts
    ]
    # MiniMax-M2 is a reasoning model: it emits a chain-of-thought preamble
    # before stating the answer, so a 64-token cap truncates the answer
    # (RULER scores by substring match). Give it room to finish.
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 1024}
    accuracy = 0
    with open("examples/ruler_output.jsonl", "w", encoding="utf-8") as f:
            if server_url:
                o = _generate_server(server_url, prompts, sampling_params)
            else:
                o = llm.generate(prompts, sampling_params)
            for res, answer in zip(o, ruler_outputs):
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
                    if answer in res["text"]:
                        accuracy += 1.0
    print(f"Ruler Accuracy: {accuracy / len(ruler_outputs) * 100:.2f}%")

if __name__ == "__main__":
    main()
