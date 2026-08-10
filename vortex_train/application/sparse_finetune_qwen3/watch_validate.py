"""Score validation checkpoints as the trainer emits them.

Runs alongside training, polls for ``<val_dir>/step*/READY``, and evaluates each new
checkpoint on AIME24 through vortex_torch's serving engine at the trained budget. Results
are appended to ``<val_dir>/history.jsonl`` and printed as a running curve.

    python -m application.sparse_finetune_qwen3.watch_validate \\
        --val-dir /scratch/zhuominc/ckpt_qwen/val --trials 16

**Why a separate process** (this was tried in-process first, and both failures look like
tuning problems but are structural):

* validation needs a *decode* engine, so a second model in a second process either way;
* the non-validating FSDP ranks sit in a barrier and NCCL's 600 s watchdog SIGABRTs them;
* the trainer's ~90 GB of live parameters stay resident, so the engine's KV allocation
  gets OOM-killed. Lowering the child's memory fraction only postpones it, since the
  trainer's peak tracks sequence length.

Out of process, training never pauses, the engine sees whatever memory is actually free,
and a failed validation cannot take the run down.

By default it waits for the GPU to be free before starting an engine, so it does not
fight the trainer for memory — it lags training rather than destabilising it. Pass
``--mem`` to size the engine explicitly if you want it to run concurrently.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def gpu_free_mb(index: int = 0) -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
        used, total = (int(x) for x in out.split(","))
        return total - used
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--task", default="aime24")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--need-free-mb", type=int, default=120_000,
                   help="wait until the GPU has this much free before evaluating; "
                        "0 disables the wait")
    ap.add_argument("--mem", type=float, default=None,
                   help="pass a fixed mem_fraction_static to the engine instead of "
                        "waiting for a free GPU")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--stop-after", type=int, default=0,
                   help="exit once this step has been scored (0 = run forever)")
    ap.add_argument("--keep-checkpoints", action="store_true",
                   help="keep each validation checkpoint (8.8 GB each) after scoring")
    a = ap.parse_args()

    val_dir = Path(a.val_dir)
    history = val_dir / "history.jsonl"
    done: set[int] = set()
    if history.exists():
        for line in history.read_text().splitlines():
            try:
                done.add(json.loads(line)["step"])
            except Exception:
                pass
    print(f"watching {val_dir} (already scored: {sorted(done)})", flush=True)

    while True:
        ready = sorted(
            (int(p.parent.name[4:]), p.parent)
            for p in val_dir.glob("step*/READY")
            if p.parent.name[4:].isdigit()
        )
        pending = [(s, d) for s, d in ready if s not in done]

        for step, ckpt in pending:
            if a.need_free_mb and not a.mem:
                while gpu_free_mb(a.gpu) < a.need_free_mb:
                    print(f"[step {step}] waiting for GPU{a.gpu} "
                          f"({gpu_free_mb(a.gpu)} MB free < {a.need_free_mb})",
                          flush=True)
                    time.sleep(a.poll)

            out_json = val_dir / f"{a.task}_step{step}.json"
            cmd = [
                sys.executable, "-m", "application.sparse_finetune_qwen3.evaluate",
                "--trained", str(ckpt), "--tasks", a.task, "--which", "trained",
                "--trials", str(a.trials), "--max-new-tokens", str(a.max_new_tokens),
                "--out", str(out_json),
            ]
            if a.mem is not None:
                cmd += ["--mem", str(a.mem)]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
            env.setdefault("HF_HOME", "/scratch/zhuominc/hf")
            print(f"[step {step}] scoring -> {out_json}", flush=True)
            t0 = time.perf_counter()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            dt = time.perf_counter() - t0

            rec = {"step": step, "task": a.task, "trials": a.trials,
                   "wall_s": round(dt, 1), "exit": r.returncode}
            if out_json.exists():
                try:
                    d = json.loads(out_json.read_text())
                    res = d["results"][0]
                    key = next(k for k in res if k.startswith("mean@"))
                    rec["mean_at_k"] = res[key]
                    rec["pass_at_k"] = res["pass@k"]
                    rec["completion_tokens"] = res.get("completion_tokens")
                except Exception as e:
                    rec["parse_error"] = str(e)
            else:
                # keep the failure in the history rather than silently skipping, so a
                # gap in the curve is explained rather than mysterious
                rec["stderr_tail"] = (r.stderr or r.stdout or "")[-400:]

            with history.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(step)
            got = rec.get("mean_at_k")
            print(f"[step {step}] "
                  + (f"mean@{a.trials}={got:.4f} pass@k={rec['pass_at_k']:.4f}"
                     if got is not None else f"FAILED exit {r.returncode}")
                  + f"  ({dt/60:.1f} min)", flush=True)

            if not a.keep_checkpoints and rec.get("mean_at_k") is not None:
                # 8.8 GB each; keeping every 100-step checkpoint would be 88 GB.
                for f in ckpt.glob("*.safetensors"):
                    f.unlink()
                print(f"[step {step}] removed weights from {ckpt} "
                      f"(pass --keep-checkpoints to retain)", flush=True)

            # running curve, so a regression is visible without opening files
            rows = [json.loads(l) for l in history.read_text().splitlines()]
            curve = [(r["step"], r.get("mean_at_k")) for r in rows
                     if r.get("mean_at_k") is not None]
            if curve:
                print("  curve: " + "  ".join(f"{s}:{v:.3f}" for s, v in sorted(curve)),
                      flush=True)

            if a.stop_after and step >= a.stop_after:
                print("reached --stop-after; exiting", flush=True)
                return

        time.sleep(a.poll)


if __name__ == "__main__":
    main()
