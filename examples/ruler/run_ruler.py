import json
import os
import sys
from transformers import AutoTokenizer

# Data files live alongside this script (examples/ruler/), so anchor to it and
# work regardless of the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))

# NOTE: the CUDA-arch JIT speedup is now handled centrally for every vortex
# entrypoint at `import vortex_torch` (see vortex_torch/_jit_setup.py), so it no
# longer needs to be set per-script here.


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
    # by examples/misc/server_launch.sh) over HTTP. The server's own launch flags
    # (topk_val, block_size, module, etc.) define the sparse-attention config;
    # this script only feeds prompts and scores the answers.
    server_url = os.environ.get("RULER_SERVER_URL")

    # Baseline toggle: ENABLE_VORTEX_SPARSITY=0 runs dense sglang (no vortex
    # sparse path) to confirm the reference accuracy; default (1) is sparse.
    # (Offline-engine mode only — server mode inherits the server's config.)
    enable_vortex_sparsity = os.environ.get("ENABLE_VORTEX_SPARSITY", "1") == "1"
    # Which registered flow to run (default gqa_block_sparse_attention), and
    # whether to disable sglang's prefix-radix cache — REQUIRED (=1) for flows
    # whose forward_indexer uses Save(...) (e.g. running_avg_block_sparse).
    vortex_module = os.environ.get("VORTEX_MODULE", "gqa_block_sparse_attention")
    disable_radix_cache = os.environ.get("DISABLE_RADIX_CACHE", "0") == "1"
    # Indexer backend: flashinfer (default) or trtllm. (TopK/Union ops are
    # trtllm-only; topK/approxTopK flows run under either.)
    vortex_attention_backend = os.environ.get("VORTEX_ATTENTION_BACKEND", "flashinfer")
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
                    vortex_module_name=vortex_module,
                    vortex_attention_backend=vortex_attention_backend,
                    trust_remote_code=True,
                    vortex_max_seq_lens=8192,
                    mem_fraction_static=0.9,
                    vortex_workload_chunk_size=32,
                    vortex_compilation_cache_dir=os.path.expanduser("~/.vortex_compilation_cache"),
                    disable_radix_cache=disable_radix_cache,
                    tp_size=1,
                    )
    
    with open(os.path.join(_HERE, "validation.jsonl"), "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line)["input"] for line in f]

    with open(os.path.join(_HERE, "validation.jsonl"), "r", encoding="utf-8") as f:
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
    with open(os.path.join(_HERE, "ruler_output.jsonl"), "w", encoding="utf-8") as f:
            if server_url:
                o = _generate_server(server_url, prompts, sampling_params)
            else:
                o = llm.generate(prompts, sampling_params)
            for res, answer in zip(o, ruler_outputs):
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
                    if answer in res["text"]:
                        accuracy += 1.0
    _tag = "server" if server_url else (vortex_module if enable_vortex_sparsity else "dense")
    print(f"Ruler Accuracy [{_tag}]: {accuracy / len(ruler_outputs) * 100:.2f}%")

if __name__ == "__main__":
    main()
