"""Capture real per-layer decode tensors from a model for offline recall study.

The human way to prototype a sparse-attention *scoring/selection* idea is to grab
the real query + context keys a model produces on a long prompt, then test —
offline, in seconds, no engine boot — how well a candidate block-scoring function
retrieves the blocks full attention actually attends to.

This captures, via the HuggingFace **unified attention interface** (no CUDA graph,
so tensor saving just works): for each chosen layer, the **last-token query**
(the decode query), the full-context **K** and **V**, and the softmax scaling.
Ground-truth attention is recomputed exactly from (q, K) in ``eval_recall.py`` —
so the trace stores only q/K/V.

Usage
-----
::

    python algorithm_scientist/research/capture_trace.py \\
        --model Qwen/Qwen3-1.7B --data examples/ruler/validation.jsonl \\
        --num-samples 2 --max-ctx 4096 --layers even8 \\
        --out algorithm_scientist/research/traces/qwen3_1.7b.pt

VORTEX/SGLANG MODE (optional, more faithful): to capture the *real page
summaries* the indexer computes, run the flow with ``"disable_cuda_graph": true``
(cudagraph freezes the forward and skips Python-side dumps) and insert a dump in
``forward_indexer``/``forward_cache`` or the attention backend. That mode is
heavier and coupled to a live flow; see AI/workflows/research_toolkit.md.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AttentionInterface


# Filled per-run; the registered attention fn writes into it.
_STORE = {}
_TARGET = set()
_CAPTURE_ON = True   # disabled during --generate warmup so only the final forward is stored
_ACCUM_W = 64        # how many recent query rows to accumulate for the H2O baseline


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _causal_align_mask(Sq, Sk, device):
    """Boolean allow-mask [Sq, Sk] with the query aligned to the END of the keys
    (correct for prefill Sq==Sk and KV-cache decode Sq<Sk)."""
    qpos = torch.arange(Sk - Sq, Sk, device=device)
    kpos = torch.arange(Sk, device=device)
    return kpos[None, :] <= qpos[:, None]            # [Sq, Sk] True = attend


def _capture_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Memory-efficient capture: SDPA for the forward output (O(S) memory, works
    at long context on GPU), and store only the cheap pieces for target layers."""
    li = getattr(module, "layer_idx", None)
    n_rep = query.shape[1] // key.shape[1]
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)

    if _CAPTURE_ON and li in _TARGET:
        B, Hq, S, D = query.shape
        W = min(_ACCUM_W, S)
        qw = query[:, :, -W:, :]                                      # [B, Hq, W, D]
        aw = torch.matmul(qw, k.transpose(-2, -1)) * scaling          # [B, Hq, W, S] (W small)
        if attention_mask is not None:
            aw = aw + attention_mask[..., -W:, : k.shape[-2]]
        else:
            allow = _causal_align_mask(W, S, query.device)            # [W, S]
            aw = aw.masked_fill(~allow[None, None], float("-inf"))
        aw = torch.softmax(aw, dim=-1, dtype=torch.float32)
        accum = aw.mean(dim=(1, 2)).squeeze(0).cpu()                  # [S]
        _STORE[li] = {
            "q": query[:, :, -1, :].detach().float().cpu().squeeze(0),       # [Hq, D]
            "K": key.detach().to(torch.float16).cpu().squeeze(0),            # [Hkv, S, D] fp16
            "V": value.detach().to(torch.float16).cpu().squeeze(0),         # [Hkv, S, D] fp16
            "accum": accum.float(),                                         # [S]
            "scaling": float(scaling),
        }
        del aw

    if attention_mask is not None:
        out = F.scaled_dot_product_attention(query, k, v, attn_mask=attention_mask)
    else:
        allow = _causal_align_mask(query.shape[2], k.shape[2], query.device)
        out = F.scaled_dot_product_attention(query, k, v, attn_mask=allow[None, None])
    out = out.transpose(1, 2).contiguous()                           # [B, S, Hq, D]
    return out, None


def _resolve_layers(spec, n_layers):
    if spec in (None, "all"):
        return list(range(n_layers))
    if spec.startswith("even"):
        k = int(spec[4:] or "8")
        k = min(k, n_layers)
        step = max(1, n_layers // k)
        return list(range(0, n_layers, step))[:k]
    return [int(x) for x in spec.split(",")]


def _load_prompts(data_path, field, n):
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(d)
            if len(rows) >= n:
                break
    out = []
    for d in rows:
        if field and field in d:
            out.append(d[field])
        else:
            out.append(d.get("prompt") or d.get("input") or d.get("question") or "")
    return out


def main():
    ap = argparse.ArgumentParser(description="Capture per-layer decode q/K/V for recall study.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="examples/ruler/validation.jsonl",
                    help="CHOOSE THIS to match the workload you're optimizing for "
                         "(attention is workload-dependent): e.g. examples/math/aime24.jsonl "
                         "for math reasoning, examples/ruler/validation.jsonl (RULER/NIAH) to "
                         "surface long-context retrieval heads. jsonl needs a "
                         "prompt/input/question field.")
    ap.add_argument("--field", default=None, help="explicit field name in the jsonl")
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--generate", type=int, default=0,
                    help="generate N tokens (greedy) before capturing, so the captured "
                         "query is a real mid-generation decode step (match the target "
                         "task's long-output regime, e.g. AIME). 0 = prompt last-token.")
    ap.add_argument("--layers", default="even8", help="all | evenN | comma list")
    ap.add_argument("--block-size", type=int, default=16, help="default block size for analysis")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    AttentionInterface.register("capture", _capture_attention)

    print(f"[capture] loading {args.model} (attn=capture, device={device})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="capture",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    layers = _resolve_layers(args.layers, n_layers)
    _TARGET.clear(); _TARGET.update(layers)
    print(f"[capture] {n_layers} layers; capturing {layers}")

    sw = getattr(model.config, "sliding_window", None)
    if sw and getattr(model.config, "use_sliding_window", True) is not False:
        print(f"[capture] WARNING: model has sliding_window={sw}. The recall harness "
              f"recomputes ground-truth attention as FULL causal — for a "
              f"sliding-window model the last-token query is masked to its window, "
              f"so recall numbers would be wrong. Use a full-attention model, or "
              f"keep --max-ctx <= the window.")

    global _CAPTURE_ON
    prompts = _load_prompts(args.data, args.field, args.num_samples)
    # leave room so prompt + generated tokens stays within --max-ctx
    prompt_cap = max(1, args.max_ctx - args.generate)
    samples = []
    for si, prompt in enumerate(prompts):
        if not prompt:
            continue
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=prompt_cap).input_ids.to(device)
        if args.generate > 0:
            # warm up: extend the context by generating, WITHOUT storing tensors,
            # so the captured query is a genuine mid-generation decode step.
            _CAPTURE_ON = False
            with torch.no_grad():
                ids = model.generate(ids, max_new_tokens=args.generate, do_sample=False,
                                     use_cache=True, pad_token_id=tok.eos_token_id)
            _CAPTURE_ON = True
        _STORE.clear()
        with torch.no_grad():
            model(ids)                        # one full forward -> capture last-token q + full K/V
        per_layer = {li: {k: v for k, v in _STORE[li].items()} for li in layers if li in _STORE}
        S = ids.shape[1]
        samples.append({"seq_len": S, "layers": per_layer})
        print(f"[capture] sample {si}: ctx={S} (gen={args.generate}), "
              f"layers captured={len(per_layer)}")

    captured = sum(len(s["layers"]) for s in samples)
    if captured == 0:
        raise SystemExit(
            "[capture] ERROR: nothing captured — this model did not route attention "
            "through the unified interface (AttentionInterface). The trace would be "
            "empty/misleading. Check the transformers version / model architecture.")

    cfg = model.config
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", Hq)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // Hq)
    trace = {
        "model": args.model,
        "num_attention_heads": Hq,
        "num_key_value_heads": Hkv,
        "head_dim": head_dim,
        "G": Hq // Hkv,
        "block_size": args.block_size,
        "layers": layers,
        "calibration_data": args.data,     # which workload this trace reflects
        "generate": args.generate,
        "samples": samples,
    }
    out = args.out or f"algorithm_scientist/research/traces/{args.model.split('/')[-1].lower()}.pt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, out)
    print(f"[capture] wrote {len(samples)} samples × {len(layers)} layers -> {out}")


if __name__ == "__main__":
    main()
