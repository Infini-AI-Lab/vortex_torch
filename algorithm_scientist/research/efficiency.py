"""Roofline / efficiency analyzer for a sparse-attention config (no GPU needed).

Decode throughput is memory-bound on the TOTAL bytes moved per step, not just KV.
Sparse attention only shrinks the KV-attention term — so its real end-to-end
speedup ceiling is gated by everything else it does NOT change:

    per-step bytes ≈ W (active weights, read once/step, shared across the batch)
                   + B · ( KV_attention  +  KV_index )      (per sequence × batch)

  ceiling_end_to_end = (W + B·KV_full) / (W + B·(KV_sparse + KV_index))

Key modeling points (per user feedback):
  * **weight dtype ≠ KV dtype** — set separately (`--weight-bytes`, `--kv-bytes`/
    config `kv_cache_dtype`). Weights may be fp8/int8 while KV is bf16, or vice versa.
  * **MoE** — only the *active* experts are read per token (router + top-k +
    shared), not all experts; that changes W a lot. Estimated from config; override
    with `--active-params-b`.
  * **per-algorithm index overhead** — scoring cost is algorithm-specific.
    `centroid`=1 summary vec/block, `quest`=2 (min+max); **`exact`** scores by
    reading ALL keys → KV_index ≈ a full K-scan → ceiling ≤ 1 (exact top-k is NOT
    efficient). Use `--algo` or `--index-vecs`.
  * **batch** — weights amortize over the decode batch (`--batch`); at batch 1 W
    often dominates and sparse attention barely helps end-to-end; at large batch
    KV dominates and the attention-only ceiling is approached.

Numbers are first-order and bound the ceiling, not the achieved speedup.

Usage
-----
::

    python algorithm_scientist/research/efficiency.py --config sub.json \\
        --seq-len 16384 --batch 64 --algo centroid --weight-bytes 2
"""

import argparse
import json
from pathlib import Path

GPUS = {"h200": (4800.0, 990.0), "h100": (3350.0, 990.0),
        "b200": (8000.0, 2250.0), "a100": (2039.0, 312.0)}

# algo -> summary vecs read per block during scoring ('exact' is special).
ALGO_VECS = {"centroid": 1, "quest": 2, "dual": 2, "minmax": 2, "none": 0}


def _load_cfg(model_path):
    p = Path(model_path).expanduser()
    if p.is_dir():
        cfg = json.loads((p / "config.json").read_text())
    else:
        from huggingface_hub import hf_hub_download
        cfg = json.loads(Path(hf_hub_download(model_path, "config.json")).read_text())
    for nest in ("text_config", "language_config"):
        if isinstance(cfg.get(nest), dict):
            cfg = {**cfg, **cfg[nest]}
    return cfg


def _geom(cfg):
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // nq
    return nq, nkv, hd, cfg["num_hidden_layers"]


def estimate_params(cfg):
    """Approximate (active_params_per_token, total_params, moe_info). Assumptions
    stated; override with --active-params-b for precision."""
    hidden = cfg["hidden_size"]; L = cfg["num_hidden_layers"]
    nq, nkv, hd, _ = _geom(cfg)
    vocab = cfg.get("vocab_size", 0)
    inter = cfg.get("intermediate_size", 4 * hidden)
    attn = hidden * nq * hd + 2 * hidden * nkv * hd + nq * hd * hidden

    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    topk = (cfg.get("num_experts_per_tok") or cfg.get("moe_topk")
            or cfg.get("num_experts_per_token"))
    moe_inter = cfg.get("moe_intermediate_size") or inter
    n_shared = cfg.get("n_shared_experts") or cfg.get("num_shared_experts") or 0
    first_dense = cfg.get("first_k_dense_replace", 0) or 0
    is_moe = bool(n_exp and topk)

    dense_mlp = 3 * hidden * inter
    moe_active = hidden * n_exp + (topk + n_shared) * 3 * hidden * moe_inter if is_moe else 0
    moe_total = hidden * n_exp + (n_exp + n_shared) * 3 * hidden * moe_inter if is_moe else 0

    active = total = 0
    for li in range(L):
        active += attn; total += attn
        if is_moe and li >= first_dense:
            active += moe_active; total += moe_total
        else:
            active += dense_mlp; total += dense_mlp
    lm = vocab * hidden
    active += lm
    total += lm + (0 if cfg.get("tie_word_embeddings") else lm)
    info = dict(is_moe=is_moe, n_exp=n_exp, topk=topk, n_shared=n_shared,
                first_dense=first_dense)
    return active, total, info


def analyze(cfg_json, model_cfg, seq_len, batch, weight_bytes, kv_bytes, algo,
            index_vecs, active_params_b):
    nq, nkv, hd, L = _geom(model_cfg)
    active, total, moe = estimate_params(model_cfg)
    if active_params_b:
        active = active_params_b * 1e9

    bs = int(cfg_json.get("vortex_block_size", 16))
    topk = int(cfg_json.get("vortex_topk_val", 30))
    resv = int(cfg_json.get("vortex_block_reserved_bos", 1)) + \
           int(cfg_json.get("vortex_block_reserved_eos", 1))
    skip = cfg_json.get("vortex_layers_skip") or []
    n_skip = len(skip) if isinstance(skip, list) else 0
    sparse_layers = max(0, L - n_skip)

    n_blocks = max(1, (seq_len + bs - 1) // bs)
    sel_blocks = min(n_blocks, topk + resv)
    sel_tokens = min(seq_len, sel_blocks * bs)

    kv_layer_full = 2 * nkv * hd * seq_len * kv_bytes
    kv_layer_sel = 2 * nkv * hd * sel_tokens * kv_bytes
    KV_full = kv_layer_full * L
    KV_sparse = kv_layer_full * n_skip + kv_layer_sel * sparse_layers

    if algo == "exact":
        # score by reading ALL keys (a full K-scan) per sparse layer -> not efficient
        KV_index = (nkv * hd * seq_len * kv_bytes) * sparse_layers
    else:
        vecs = index_vecs if index_vecs is not None else ALGO_VECS.get(algo, 1)
        KV_index = (n_blocks * vecs * hd * nkv * kv_bytes) * sparse_layers

    W = active * weight_bytes
    full = W + batch * KV_full
    sparse = W + batch * (KV_sparse + KV_index)
    return {
        "active_params": active, "total_params": total, "moe": moe,
        "weight_bytes": weight_bytes, "kv_bytes": kv_bytes, "algo": algo,
        "seq_len": seq_len, "batch": batch, "block_size": bs,
        "selected_tokens": sel_tokens, "token_ratio": sel_tokens / seq_len,
        "sparse_layers": sparse_layers, "dense_layers": n_skip,
        "W_MB": W / 1e6, "KV_full_MB": KV_full / 1e6, "KV_sparse_MB": KV_sparse / 1e6,
        "KV_index_MB": KV_index / 1e6,
        "ceiling_attn_only": KV_full / (KV_sparse + KV_index) if (KV_sparse + KV_index) else float("inf"),
        "ceiling_end_to_end": full / sparse if sparse else float("inf"),
        "dominant": "weights" if W > batch * KV_full else "KV",
    }


def main():
    ap = argparse.ArgumentParser(description="Roofline/efficiency analyzer (weights+KV+index, MoE-aware).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1, help="decode batch (weights amortize over it)")
    ap.add_argument("--gpu", default="h200", choices=list(GPUS))
    ap.add_argument("--weight-bytes", type=float, default=2.0, help="weight dtype bytes (bf16=2, fp8/int8=1)")
    ap.add_argument("--kv-bytes", type=float, default=None, help="override KV dtype bytes (else from kv_cache_dtype)")
    ap.add_argument("--algo", default="centroid",
                    help="index cost model: centroid|quest|dual|none|exact")
    ap.add_argument("--index-vecs", type=int, default=None, help="override summary vecs/block")
    ap.add_argument("--active-params-b", type=float, default=None, help="override active params (billions)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    model = cfg.get("model_path") or "Qwen/Qwen3-1.7B"
    mcfg = _load_cfg(model)
    kv_bytes = args.kv_bytes if args.kv_bytes is not None else \
        (1.0 if "fp8" in str(cfg.get("kv_cache_dtype", "auto")).lower() else 2.0)

    r = analyze(cfg, mcfg, args.seq_len, args.batch, args.weight_bytes, kv_bytes,
                args.algo, args.index_vecs, args.active_params_b)
    m = r["moe"]
    moestr = (f"MoE: {m['n_exp']} experts, top-{m['topk']}+{m['n_shared']} shared"
              if m["is_moe"] else "dense")
    print(f"# efficiency — {model}  seq={r['seq_len']} batch={r['batch']} "
          f"gpu={args.gpu}  W={r['weight_bytes']:.0f}B KV={r['kv_bytes']:.0f}B  algo={r['algo']}")
    print(f"  {moestr};  active ~{r['active_params']/1e9:.2f}B / total ~{r['total_params']/1e9:.2f}B params")
    print(f"  selected {r['selected_tokens']} tok (ratio {r['token_ratio']:.3f}); "
          f"{r['dense_layers']} dense / {r['sparse_layers']} sparse layers")
    print(f"  per-step bytes:  W {r['W_MB']:.0f} MB  +  batch×( KV_full {r['KV_full_MB']:.0f} "
          f"-> KV_sparse {r['KV_sparse_MB']:.0f} + index {r['KV_index_MB']:.1f} ) MB")
    print(f"  attention-only ceiling (ignores weights): {r['ceiling_attn_only']:.2f}x")
    print(f"  ** END-TO-END ceiling @ batch {r['batch']}: {r['ceiling_end_to_end']:.2f}x **  "
          f"(dominant term: {r['dominant']})")
    if r["algo"] == "exact":
        print("  exact top-k scores by reading ALL keys -> index ≈ full K-scan; "
              "ceiling collapses — exact top-k attention is NOT efficient.")
    print("  raise --batch to see KV dominate (ceiling ↑); measured speedup is below "
          "this ceiling (gather/launch/occupancy) -> /add-ops investigate if far off.")


if __name__ == "__main__":
    main()
