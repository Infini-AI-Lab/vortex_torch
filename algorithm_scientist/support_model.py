"""Static support check for a model under vortex_torch (CPU-only, no GPU).

Reads a model's ``config.json`` (local dir or HF repo id), detects the attention
geometry (MLA vs MHA/GQA), recommends the vortex attention backend and conda
env, and reports the shapes a submission/preflight will need. This is the
*static* half of ``/support-model``; the command does the live boot + RULER.

Usage
-----
::

    python algorithm_scientist/support_model.py Qwen/Qwen3-4B
    python algorithm_scientist/support_model.py /path/to/local/model --json

Exit code 0 = geometry recognized & a backend is recommendable; 2 = unknown
geometry (needs new wiring in vortex_torch/engine/sgl/integration.py).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _load_config(model: str) -> Dict[str, Any]:
    local = Path(model).expanduser()
    if local.is_dir():
        p = local / "config.json"
        if not p.is_file():
            raise SystemExit(f"local model dir {local} has no config.json")
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub to fetch remote configs") from e
    p = Path(hf_hub_download(repo_id=model, filename="config.json"))
    return json.loads(p.read_text(encoding="utf-8"))


def analyze(model: str) -> Dict[str, Any]:
    cfg = _load_config(model)
    # Some models nest the real config under "text_config" / "llm_config".
    for nest in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(nest), dict):
            merged = dict(cfg)
            merged.update(cfg[nest])
            cfg = merged
            break

    model_type = str(cfg.get("model_type", "")).lower()
    is_mla = "kv_lora_rank" in cfg and cfg.get("kv_lora_rank")

    out: Dict[str, Any] = {
        "model": model,
        "model_type": model_type,
        "geometry": "MLA" if is_mla else "MHA/GQA",
    }

    if is_mla:
        out.update({
            "kv_lora_rank": cfg.get("kv_lora_rank"),
            "qk_rope_head_dim": cfg.get("qk_rope_head_dim"),
            "qk_nope_head_dim": cfg.get("qk_nope_head_dim"),
            "v_head_dim": cfg.get("v_head_dim"),
            # MLA decode backends registered in integration.py.
            "recommended_backends": ["trtllm_mla", "triton", "cuda_mla"],
            "vortex_attention_backend_note":
                "MLA models use the MLA shims (trtllm_mla decode / triton / "
                "cuda_mla). vFlow must be a vFlowMLA subclass.",
        })
    else:
        nq = cfg.get("num_attention_heads")
        nkv = cfg.get("num_key_value_heads", nq)
        hidden = cfg.get("hidden_size")
        head_dim = cfg.get("head_dim")
        if head_dim is None and hidden and nq:
            head_dim = hidden // nq
        out.update({
            "num_attention_heads": nq,
            "num_key_value_heads": nkv,
            "head_dim": head_dim,
            "G_group_size": (nq // nkv) if (nq and nkv) else None,
            # Non-MLA vortex path -> flashinfer/trtllm shims.
            "recommended_backends": ["flashinfer", "trtllm"],
            "vortex_attention_backend_note":
                "MHA/GQA models use the flashinfer/trtllm shims (non-MLA "
                "vortex path). vFlow is a standard vFlow subclass.",
        })

    # Conda env: GLM-family loads only in vortex_glm; everything else vortex_v1.
    glm = "glm" in model_type or "glm" in model.lower()
    out["recommended_env"] = "vortex_glm" if glm else "vortex_v1"
    out["env_note"] = (
        "GLM-family (glm4_moe*) only loads in vortex_glm (transformers 5.0)."
        if glm else "Qwen/Llama/DeepSeek-family load in vortex_v1."
    )

    # Known-supported quick verdict.
    known = any(k in model_type for k in ("qwen", "llama", "glm", "deepseek", "mistral", "olmo"))
    out["static_verdict"] = "likely-supported" if known else "unknown-geometry"
    out["next_step"] = (
        "Boot a tiny engine + run RULER (the /support-model command does this)."
        if known else
        "Geometry not in the known set — inspect vortex_torch/engine/sgl/"
        "integration.py shims and wire the backend, then re-verify."
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Static vortex support check for a model.")
    ap.add_argument("model", help="HF repo id or local model dir")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    info = analyze(args.model)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"# vortex support check — {info['model']}")
        for k, v in info.items():
            if k == "model":
                continue
            print(f"  {k}: {v}")
    sys.exit(0 if info["static_verdict"] == "likely-supported" else 2)


if __name__ == "__main__":
    main()
