#!/usr/bin/env python
"""Run the RULER needle-in-a-haystack benchmark on GLM-4.7-Flash with vortex
sparse MLA attention (rope-aware block-sparse routing on the hand-written CUDA
decode kernel).

This is a self-contained, single-GPU example of the vortex sparse-MLA decode
path (default GLM-4.7-Flash, block=page=32, topk=29); every knob is overridable
on the CLI (the known-good ~100%-RULER GLM config used topk=61). The CLI —
flags AND defaults — is unified with run_ruler_mha.py; only the
model/module/backend defaults differ.

    cuda_mla   sglang attention backend  -> vortex CUDA MLA decode + flashinfer prefill
    trtllm     vortex indexer backend    -> 2D block-table page selection
    rope_aware_block_sparse_mla          -> scores pages by the FULL fused-latent dot
                                            <q_nope,c_kv> + <q_pe,c_pe>, then top-k

GLM-4.7-Flash (model type ``glm4_moe_lite``) requires transformers >= 5.0, so run
this in the ``vortex_glm`` conda env (NOT ``vortex_v1``, which can't load it):

    conda activate vortex_glm
    CUDA_VISIBLE_DEVICES=<free-gpu> python examples/ruler/run_ruler_mla.py
    # or pin the GPU / shrink the slice:
    python examples/ruler/run_ruler_mla.py --gpu 3 --n 20 --dump

The script forces HF offline mode by default (the model is expected to be cached);
pass ``--online`` to allow hub access.
"""
import argparse
import json
import os

# Must be set before sglang/vortex import these. setdefault so the caller's env
# (or the shell) still wins.
os.environ.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zai-org/GLM-4.7-Flash",
                   help="HF model id (default: GLM-4.7-Flash).")
    p.add_argument("--module", default="rope_aware_block_sparse_mla",
                   help="vortex MLA flow name (default: rope_aware_block_sparse_mla).")
    p.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_4k.jsonl"),
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
                   help="Enable GLM thinking mode (default off; needles answer directly).")
    p.add_argument("--disable-cuda-graph", action="store_true",
                   help="Run eager (no cuda graph capture).")
    p.add_argument("--disable-radix-cache", action="store_true",
                   help="Disable sglang's prefix-radix cache — REQUIRED for flows whose "
                        "forward_indexer uses Save(...).")
    p.add_argument("--dense", action="store_true",
                   help="Disable vortex sparsity: run dense MLA on --attn-backend.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="sglang kv_cache_dtype (auto|fp8_e4m3|bfloat16).")
    p.add_argument("--attn-backend", default="cuda_mla",
                   help="sglang attention_backend. Sparse vortex MLA uses 'cuda_mla' "
                        "(default); dense baselines can use 'trtllm_mla' or 'triton'. "
                        "Note: flashinfer MLA does not work for GLM.")
    p.add_argument("--indexer-backend", default="trtllm",
                   help="vortex indexer backend (default: trtllm — the 2D block-table "
                        "page selection the MLA flows are written for).")
    p.add_argument("--online", action="store_true",
                   help="Allow HF hub access (default: HF_HUB_OFFLINE=1).")
    p.add_argument("--dump", action="store_true",
                   help="Print the first few (expected, generated) pairs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "0" if args.online else "1")

    import sglang as sgl
    import vortex_torch  # noqa: F401  -- installs the ServerArgs / VortexConfig adapter
    from transformers import AutoTokenizer

    mode = "dense" if args.dense else "sparse"
    layers_skip = [int(x) for x in args.layers_skip.split(",") if x.strip()]
    print(f"[run_ruler_mla] model={args.model} mode={mode} module={args.module} "
          f"block={args.block} topk={args.topk} layers_skip={layers_skip} n={args.n} "
          f"indexer_backend={args.indexer_backend} kv_cache_dtype={args.kv_cache_dtype} "
          f"cuda_graph={'off' if args.disable_cuda_graph else 'on'}", flush=True)

    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        tp_size=args.tp,
        page_size=args.block,                       # page == block (one block per page)
        attention_backend=args.attn_backend,        # sparse vortex MLA: cuda_mla
        kv_cache_dtype=args.kv_cache_dtype,
        mem_fraction_static=args.mem_fraction,
        disable_cuda_graph=args.disable_cuda_graph,
        disable_radix_cache=args.disable_radix_cache,
    )
    if not args.dense:
        # Flat vortex_* kwargs; the adapter folds them into one VortexConfig and ships
        # them across the spawn boundary. These reproduce the known-good GLM run.
        engine_kwargs.update(
            enable_vortex_sparsity=True,
            vortex_module_name=args.module,
            vortex_attention_backend=args.indexer_backend,  # 2D block-table indexer
            vortex_impl_backend="triton",               # tensor-core indexer GeMM
            vortex_use_tensor_core=True,
            vortex_block_size=args.block,
            vortex_topk_val=args.topk,
            vortex_topk_ratio=0.0,
            vortex_block_reserved_bos=1,
            vortex_block_reserved_eos=2,
            vortex_dtype="bfloat16",
            vortex_layers_skip=layers_skip,
            vortex_max_seq_lens=40960,
            vortex_workload_chunk_size=64,
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
    outs = llm.generate(
        prompts, {"temperature": 0.0, "max_new_tokens": args.max_new_tokens})

    hits = [rows[i]["outputs"][0] in outs[i]["text"] for i in range(len(rows))]
    if args.dump:
        for i in range(min(3, len(rows))):
            print(f"\n--- ex{i} hit={hits[i]} expect={rows[i]['outputs'][0]!r} ---")
            print(f"GEN: {outs[i]['text'][:400]!r}", flush=True)

    acc = sum(hits)
    tag = f"dense {args.attn_backend}" if args.dense else f"sparse {args.module}"
    print(f"\n>>> RULER {args.model.split('/')[-1]} | {tag} | kv={args.kv_cache_dtype}: "
          f"{acc}/{len(rows)} = {acc / len(rows) * 100:.1f}%", flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
