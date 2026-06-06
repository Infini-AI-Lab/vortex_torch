#!/usr/bin/env python
"""GPU-pool scheduler for the algo2 experiment matrix.

Keeps every usable GPU busy: polls nvidia-smi, launches the next queued config on
any free GPU (excluding broken / other-user GPUs), reaps finished jobs, re-queues
device-flake failures, and regenerates examples/misc/algo2_results.md as runs land.

  conda activate vortex_glm
  ALGO2_GPUS=1,3,4,5 python examples/misc/run_algo2_pool.py            # full matrix
  ALGO2_GPUS=1,3 SMOKE=1 python examples/misc/run_algo2_pool.py        # quick smoke

Env:
  ALGO2_GPUS   candidate GPU ids (default "1,3,4,5"; 0/2 are broken, 6/7 often busy)
  MEM_FREE_MIB a candidate GPU counts as free below this many MiB used (default 2000)
  POLL_S       poll interval seconds (default 15)
  MAX_RETRY    re-queue attempts per config on device-flake (default 3)
  SMOKE        if "1": trials=1, gen=2048, only 3 configs — pipeline smoke test
"""
import os, sys, json, time, subprocess, shlex, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_DIR = "summary-glm4.7-flash"
LOGDIR = "/tmp/algo2_pool_logs"
SMOKE = os.environ.get("SMOKE") == "1"

GPUS = [int(g) for g in os.environ.get("ALGO2_GPUS", "1,3,4,5").split(",") if g.strip()]
MEM_FREE_MIB = int(os.environ.get("MEM_FREE_MIB", "2000"))
POLL_S = int(os.environ.get("POLL_S", "15"))
MAX_RETRY = int(os.environ.get("MAX_RETRY", "3"))

TRIALS = 1 if SMOKE else 16
GEN = 2048 if SMOKE else 32768
TOPK = [61] if SMOKE else [61, 93, 125, 157, 253]
SPARSE_MODULES = ["rope_aware_block_sparse_mla", "lserve_centroid_mla"]

COMMON = [
    "--trials", str(TRIALS), "--page-size", "16", "--block-size", "16",
    "--workload-chunk-size", "64", "--topk-ratio", "0.00",
    "--model-name", "zai-org/GLM-4.7-Flash", "--data-path", "examples/math/aime26_glm.jsonl",
    "--mem", "0.9", "--generation-max-new-tokens", str(GEN),
    "--max-input-length", "4096", "--tp-size", "1",
    "--summary-dir", SUMMARY_DIR, "--skip-already-finished-check",
]


def build_queue():
    q = []
    # 1. dense baseline (once)
    q.append({"name": "full_attention", "args": [
        "--vortex-module-name", "full_attention",
        "--attention-backend", "flashinfer", "--topk-val", "253"]})
    # 2/3. sparse modules × topk: cuda_mla + tensor-core indexer, no layer skip
    mods = SPARSE_MODULES[:1] if SMOKE else SPARSE_MODULES
    for mod in mods:
        for k in TOPK:
            q.append({"name": f"{mod}_tk{k}", "args": [
                "--vortex-module-name", mod, "--topk-val", str(k),
                "--attention-backend", "cuda_mla",
                "--vortex-impl-backend", "triton", "--vortex-use-tensor-core",
                "--vortex-layers-skip"]})   # must be last: no values => skip none
    for c in q:
        c["tries"] = 0
    return q


def gpu_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    m = {}
    for line in out.strip().splitlines():
        i, used = line.split(",")
        m[int(i)] = int(used)
    return m


def render():
    subprocess.run([sys.executable, "examples/misc/collect_algo2_results.py",
                    "--summary-dir", SUMMARY_DIR, "--out", "examples/misc/algo2_results.md"],
                   cwd=REPO, capture_output=True)


def launch(cfg, gpu):
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{cfg['name']}.out")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               TORCH_CUDA_ARCH_LIST="10.0", SGLANG_ENABLE_TORCH_COMPILE="0",
               HF_HOME="/raid/catalyst/models/", PYTHONPATH=".")
    cmd = [sys.executable, "examples/math/verify_algo.py"] + COMMON + cfg["args"]
    fh = open(log, "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    cfg["tries"] += 1
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {cfg['name']} on GPU{gpu} "
          f"(try {cfg['tries']}, pid {p.pid}) -> {log}", flush=True)
    return {"cfg": cfg, "gpu": gpu, "proc": p, "log": log}


def flaked(log):
    try:
        t = open(log, errors="ignore").read()
    except OSError:
        return False
    return "cudaErrorDevicesUnavailable" in t or "is/are busy or unavailable" in t


def main():
    os.chdir(REPO)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    queue = build_queue()
    running = {}   # gpu -> job
    failed = []
    print(f"pool: GPUs {GPUS} | {len(queue)} configs | SMOKE={SMOKE} "
          f"(trials={TRIALS}, gen={GEN})", flush=True)

    while queue or running:
        # reap finished
        for gpu in list(running):
            job = running[gpu]
            rc = job["proc"].poll()
            if rc is None:
                continue
            cfg = job["cfg"]
            del running[gpu]
            if rc == 0:
                print(f"[{time.strftime('%H:%M:%S')}] DONE  {cfg['name']} (GPU{gpu}, rc=0)", flush=True)
                render()
            elif flaked(job["log"]) and cfg["tries"] < MAX_RETRY:
                print(f"[{time.strftime('%H:%M:%S')}] FLAKE {cfg['name']} (GPU{gpu}) -> requeue", flush=True)
                queue.append(cfg)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] FAIL  {cfg['name']} (GPU{gpu}, rc={rc}) "
                      f"-- see {job['log']}", flush=True)
                failed.append(cfg["name"])

        # fill free GPUs (one launch per cycle => natural stagger, avoids init races)
        if queue:
            mem = gpu_mem()
            for gpu in GPUS:
                if gpu in running:
                    continue
                if mem.get(gpu, 10**9) < MEM_FREE_MIB:
                    running[gpu] = launch(queue.pop(0), gpu)
                    break  # one per cycle

        time.sleep(POLL_S)

    render()
    print(f"\n=== POOL DONE === failed: {failed or 'none'}", flush=True)
    print("report: examples/misc/algo2_results.md", flush=True)


if __name__ == "__main__":
    main()
