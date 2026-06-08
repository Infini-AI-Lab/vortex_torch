#!/usr/bin/env python
"""Profile vortex sparse-MLA selection quality on RULER, then print a report.

Drives a short decode with ``attention_backend=cuda_mla_profile`` (the profiling
twin of ``cuda_mla`` — identical execution plus, for every decoded token and
every layer/head, **p-coverage** = fraction of the dense softmax mass captured by
the selected KV, and **recall@N** = fraction of the exact top-N tokens that were
selected). The backend writes a JSON report; this script renders it.

    conda activate vortex_glm          # GLM needs transformers >= 5
    export HF_HOME=/raid/catalyst/models/
    CUDA_VISIBLE_DEVICES=0 python examples/ruler/run_profile_mla.py --n 4

NOTE: profiling recomputes the dense attention in PyTorch per layer per token, so
it is much slower than cuda_mla and runs eager (cuda graph disabled). Use a small
--n. The companion math-workload profiler is examples/math/run_profile_mla.py.
"""
import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zai-org/GLM-4.7-Flash", help="HF model id (MLA).")
    p.add_argument("--module", default="rope_aware_block_sparse_mla",
                   help="vortex MLA flow name.")
    p.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_4k.jsonl"),
                   help="jsonl with the prompts.")
    p.add_argument("--field", default="input", help="jsonl field holding the prompt text.")
    p.add_argument("--gpu", default=None, help="GPU index to pin (CUDA_VISIBLE_DEVICES).")
    p.add_argument("--n", type=int, default=4, help="Number of prompts (keep small).")
    p.add_argument("--block", type=int, default=32, help="vortex block size == page size.")
    p.add_argument("--topk", type=int, default=61, help="vortex_topk_val (selected blocks).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--recall-n", default="16,64,128", help="comma list of N for recall@N.")
    p.add_argument("--out", default="mla_profile.json", help="report JSON path.")
    p.add_argument("--mem-fraction", type=float, default=0.85)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--online", action="store_true")
    return p.parse_args()


def run_profile(args) -> str:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "0" if args.online else "1")
    os.environ.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")
    # Profiling backend knobs (read by VortexCudaMLAProfileBackend at init).
    os.environ["VORTEX_MLA_PROFILE_OUT"] = os.path.abspath(args.out)
    os.environ["VORTEX_MLA_PROFILE_RECALL_N"] = args.recall_n

    import sglang as sgl
    import vortex_torch  # noqa: F401  installs the adapter + registers backends
    from transformers import AutoTokenizer

    print(f"[profile] model={args.model} module={args.module} block={args.block} "
          f"topk={args.topk} n={args.n} recall_N={args.recall_n}", flush=True)

    llm = sgl.Engine(
        model_path=args.model,
        trust_remote_code=True,
        tp_size=args.tp,
        page_size=args.block,
        attention_backend="cuda_mla_profile",
        disable_cuda_graph=True,                 # profiling is not cuda-graph safe
        mem_fraction_static=args.mem_fraction,
        enable_vortex_sparsity=True,
        vortex_module_name=args.module,
        vortex_attention_backend="trtllm",
        vortex_impl_backend="triton",
        vortex_use_tensor_core=True,
        vortex_block_size=args.block,
        vortex_topk_val=args.topk,
        vortex_topk_ratio=0.0,
        vortex_block_reserved_bos=1,
        vortex_block_reserved_eos=2,
        vortex_dtype="bfloat16",
        vortex_layers_skip=[],
        vortex_max_seq_lens=8192,
        vortex_workload_chunk_size=64,
    )

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][: args.n]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def render(text: str) -> str:
        msg = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=args.thinking)
        except TypeError:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    prompts = [render(str(r[args.field])) for r in rows]
    llm.generate(prompts, {"temperature": 0.0, "max_new_tokens": args.max_new_tokens})
    llm.shutdown()   # triggers the backend's atexit final dump
    return os.path.abspath(args.out)


def render_report(path: str) -> None:
    if not os.path.exists(path):
        print(f"[profile] no report at {path} — did any token decode?")
        return
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    meta, overall, layers = rep["meta"], rep["overall"], rep["layers"]
    Ns = meta["recall_N"]

    print("\n" + "=" * 72)
    print(f"  vortex sparse-MLA selection profile  ({meta.get('module')})")
    print("=" * 72)
    print(f"  model={meta.get('model')}  backend={meta.get('attention_backend')}")
    print(f"  block_size={meta['block_size']}  topk_val={meta['topk_val']}  "
          f"heads={meta['num_heads']}  tokens_profiled={overall['tokens_profiled']}")
    rec_str = "  ".join(f"recall@{N}={overall['recall_mean'][str(N)]:.3f}"
                        for N in Ns if overall['recall_mean'][str(N)] is not None)
    print(f"  OVERALL  p-coverage={overall['p_coverage_mean']:.3f}   {rec_str}")
    print("-" * 72)

    hdr = f"  {'layer':>5} {'tokens':>7} {'p-cov':>7}" + "".join(f" {'r@'+str(N):>7}" for N in Ns)
    print(hdr)
    for lid in sorted(layers, key=lambda x: int(x)):
        L = layers[lid]
        rcells = "".join(f" {L['recall'][str(N)]['mean']:>7.3f}" for N in Ns)
        print(f"  {lid:>5} {L['count']:>7} {L['p_coverage_mean']:>7.3f}{rcells}")
    print("-" * 72)

    # worst layers by coverage — where the flow is leaking the most mass.
    by_cov = sorted(layers.items(), key=lambda kv: kv[1]["p_coverage_mean"])
    worst = ", ".join(f"L{lid}({L['p_coverage_mean']:.2f})" for lid, L in by_cov[:5])
    print(f"  lowest-coverage layers: {worst}")
    print(f"  full per-head detail in: {path}")


def main() -> None:
    args = parse_args()
    path = run_profile(args)
    render_report(path)


if __name__ == "__main__":
    main()
