#!/usr/bin/env python
"""Run the RULER needle-in-a-haystack benchmark on an MHA/GQA model with vortex
sparse attention.

This is a self-contained, single-GPU example of the vortex sparse decode path
for GQA models (default Qwen3-4B, block=page=32, topk=29); every knob is
overridable on the CLI. The CLI — flags AND defaults — is unified with
run_ruler_mla.py; only the model/module/backend defaults differ.

    flashinfer            sglang attention backend (--attn-backend, default)
    flashinfer | trtllm   vortex indexer backend (--indexer-backend);
                          TopK/Union flows need trtllm, topK/approxTopK
                          flows run under either

Runs in the ``vortex_v1`` env (transformers 4.x):

    conda activate vortex_v1
    CUDA_VISIBLE_DEVICES=<free-gpu> python examples/ruler/run_ruler_mha.py
    # or pin the GPU / shrink the slice:
    python examples/ruler/run_ruler_mha.py --gpu 3 --n 20 --dump

Server mode (this script only): pass ``--server-url`` (or set RULER_SERVER_URL)
to drive an already-running sglang server over HTTP instead of building an
in-process Engine. The server's own launch flags (topk, block, module, …)
define the sparse-attention config; this script only feeds prompts and scores
the answers.

The script forces HF offline mode by default (the model is expected to be
cached); pass ``--online`` to allow hub access.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# NOTE: the CUDA-arch JIT speedup is handled centrally for every vortex
# entrypoint at `import vortex_torch` (see vortex_torch/_jit_setup.py), so it
# does not need to be set per-script here.

DEFAULT_POLICY = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="HF model id (default: Qwen3-4B).")
    p.add_argument("--module", default="gqa_block_sparse_attention",
                   help="vortex MHA flow name (default: gqa_block_sparse_attention).")
    p.add_argument("--data", default=os.path.join(_HERE, "validation_4k.jsonl"),
                   help="RULER jsonl with {input, outputs:[str]} rows.")
    p.add_argument("--gpu", default=None,
                   help="GPU index to pin (sets CUDA_VISIBLE_DEVICES). Default: inherit env.")
    p.add_argument("--n", type=int, default=100, help="Number of examples (default: 100).")
    p.add_argument("--block", type=int, default=32, help="vortex block size == page size.")
    p.add_argument("--topk", type=int, default=29, help="vortex_topk_val (selected blocks).")
    p.add_argument("--layers-skip", default="",
                   help="Layers to run dense, comma-separated (e.g. '0,1'). Default: none.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mem-fraction", type=float, default=0.9)
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size.")
    p.add_argument("--thinking", action="store_true",
                   help="Enable chat-template thinking mode (default off; needles answer directly).")
    p.add_argument("--disable-cuda-graph", action="store_true",
                   help="Run eager (no cuda graph capture).")
    p.add_argument("--disable-radix-cache", action="store_true",
                   help="Disable sglang's prefix-radix cache — REQUIRED for flows whose "
                        "forward_indexer uses Save(...) (e.g. running_avg_block_sparse).")
    p.add_argument("--dense", action="store_true",
                   help="Disable vortex sparsity: run dense attention on --attn-backend.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="sglang kv_cache_dtype (auto|fp8_e4m3|bfloat16).")
    p.add_argument("--attn-backend", default="flashinfer",
                   help="sglang attention_backend (default: flashinfer).")
    p.add_argument("--indexer-backend", default="flashinfer",
                   help="vortex indexer backend: flashinfer (default) or trtllm. "
                        "TopK/Union ops are trtllm-only; topK/approxTopK flows run under either.")
    p.add_argument("--server-url", default=os.environ.get("RULER_SERVER_URL"),
                   help="Drive a running sglang server's /generate endpoint instead of "
                        "building an in-process Engine (default: $RULER_SERVER_URL).")
    p.add_argument("--online", action="store_true",
                   help="Allow HF hub access (default: HF_HUB_OFFLINE=1).")
    p.add_argument("--dump", action="store_true",
                   help="Print the first few (expected, generated) pairs.")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "0" if args.online else "1")
    layers_skip = [int(x) for x in args.layers_skip.split(",") if x.strip()]

    from transformers import AutoTokenizer

    llm = None
    if args.server_url:
        print(f"[run_ruler_mha] SERVER mode via {args.server_url}", flush=True)
    else:
        import sglang as sgl
        import vortex_torch  # noqa: F401  -- installs the ServerArgs / VortexConfig adapter

        mode = "dense" if args.dense else "sparse"
        print(f"[run_ruler_mha] model={args.model} mode={mode} module={args.module} "
              f"block={args.block} topk={args.topk} layers_skip={layers_skip} n={args.n} "
              f"indexer_backend={args.indexer_backend} kv_cache_dtype={args.kv_cache_dtype} "
              f"cuda_graph={'off' if args.disable_cuda_graph else 'on'}", flush=True)

        engine_kwargs = dict(
            model_path=args.model,
            trust_remote_code=True,
            tp_size=args.tp,
            page_size=args.block,                   # page == block (one block per page)
            attention_backend=args.attn_backend,
            kv_cache_dtype=args.kv_cache_dtype,
            mem_fraction_static=args.mem_fraction,
            disable_cuda_graph=args.disable_cuda_graph,
            disable_overlap_schedule=False,
            disable_radix_cache=args.disable_radix_cache,
        )
        if not args.dense:
            # Flat vortex_* kwargs; the adapter folds them into one VortexConfig and
            # ships them across the spawn boundary. These reproduce the known-good
            # Qwen3-4B run.
            engine_kwargs.update(
                enable_vortex_sparsity=True,
                vortex_module_name=args.module,
                vortex_attention_backend=args.indexer_backend,
                vortex_block_size=args.block,
                vortex_topk_val=args.topk,
                vortex_schedule_policy=DEFAULT_POLICY,
                vortex_block_reserved_bos=1,
                vortex_block_reserved_eos=2,
                vortex_dtype="bfloat16",
                vortex_layers_skip=layers_skip,
                vortex_max_seq_lens=40960,
                vortex_workload_chunk_size=32,
                vortex_compilation_cache_dir=os.path.expanduser("~/.vortex_compilation_cache"),
            )
        llm = sgl.Engine(**engine_kwargs)

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][: args.n]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def render(text: str) -> str:
        msg = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.thinking)
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    prompts = [render(r["input"]) for r in rows]
    # Reasoning models (e.g. MiniMax-M2) emit a chain-of-thought preamble before
    # stating the answer, so a small token cap truncates the answer (RULER scores
    # by substring match) — raise --max-new-tokens (e.g. 1024) for those.
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "max_new_tokens": args.max_new_tokens}
    if args.server_url:
        outs = _generate_server(args.server_url, prompts, sampling_params)
    else:
        outs = llm.generate(prompts, sampling_params)

    hits = [rows[i]["outputs"][0] in outs[i]["text"] for i in range(len(rows))]
    with open(os.path.join(_HERE, "ruler_output.jsonl"), "w", encoding="utf-8") as f:
        for res in outs:
            json.dump(res, f, ensure_ascii=False)
            f.write("\n")
    if args.dump:
        for i in range(min(3, len(rows))):
            print(f"\n--- ex{i} hit={hits[i]} expect={rows[i]['outputs'][0]!r} ---")
            print(f"GEN: {outs[i]['text'][:400]!r}", flush=True)

    acc = sum(hits)
    if args.server_url:
        tag = "server"
    elif args.dense:
        tag = f"dense {args.attn_backend}"
    else:
        tag = f"sparse {args.module}"
    print(f"\n>>> RULER {args.model.split('/')[-1]} | {tag} | kv={args.kv_cache_dtype}: "
          f"{acc}/{len(rows)} = {acc / len(rows) * 100:.1f}%", flush=True)
    if llm is not None:
        llm.shutdown()


if __name__ == "__main__":
    main()
