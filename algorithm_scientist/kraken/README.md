# Running vortex_torch on the Leviathan cluster

Two files, used together:

| file | role |
|---|---|
| [job.yaml](job.yaml) | kraken job spec — 1 node, 8x B200, ap-south-1 |
| [build_env.sh](build_env.sh) | builds the venv inside the container, then optionally runs the RULER sweeps |

Use this when the local host has no free GPU (`algorithm_scientist/free_gpus.sh`
returns nothing, or `nvidia-smi` is absent).

## One-time access setup

```bash
ada profile add \
  --account=643233519534 --provider=conduit \
  --role=ScientistRole-green-Leviathan-prod-16-us-east-1 \
  --profile leviathan-prod-16-us-east-1

aws eks update-kubeconfig \
  --name Leviathan-prod-16-us-east-1 --region us-east-1 \
  --profile leviathan-prod-16-us-east-1 \
  --alias leviathan-prod-16-us-east-1 \
  --user-alias leviathan-prod-16-us-east-1
```

Project is `green`. Check capacity before submitting — the B200 pool is shared:

```bash
kraken -p green queues list          # look at the Available column
kraken -p green jobs list            # what else is running
```

## Launch

`jobName` must be unique; edit it in `job.yaml` first.

```bash
kraken -p green jobs create -i algorithm_scientist/kraken/job.yaml
kraken -p green jobs update-kubeconfig -j <jobName>   # per-job kubectl context
kubectl get pods | grep <jobName>
```

The pod idles for 6h so it can be driven with `kubectl exec`. Stop it when done
— it holds a whole 8-GPU node:

```bash
kraken -p green jobs stop --job-name <jobName>
```

## Build the env and run the sweeps

`/scratch` is FSx and **persists across jobs**, so this is a one-time cost; a
later job reuses the venv, the HF cache and the repo.

```bash
POD=<jobName>-worker-0

# Ship the working tree (the vendored sglang is not on a remote branch).
tar czf /tmp/vortex_repo.tgz --exclude=.git --exclude=__pycache__ \
    --exclude='.venv*' --exclude=logs \
    --exclude=third_party/flashinfer --exclude=third_party/flash-attention .
kubectl cp /tmp/vortex_repo.tgz $POD:/scratch/vortex_repo.tgz
kubectl exec $POD -- bash -lc '
  mkdir -p /scratch/zhuominc/vortex_torch &&
  cd /scratch/zhuominc/vortex_torch && tar xzf /scratch/vortex_repo.tgz'

# Build + verify, then run both sweeps (SWEEP=none|mha|mla|both).
kubectl exec $POD -- bash -lc '
  cd /scratch/zhuominc/vortex_torch &&
  REPO=$PWD VENV=/scratch/zhuominc/venv-0516 SWEEP=both \
  HF_HOME=/scratch/zhuominc/hf \
    bash algorithm_scientist/kraken/build_env.sh 2>&1 | tail -40'
```

`build_env.sh` verifies the env before running anything: sglang/vortex versions,
that the vortex hooks are live in the vendored tree, and that all five attention
backends register. It refuses to run the sweeps if that fails.

## Gotchas

- **Pre-fetch models.** `huggingface_hub`'s 10s default timeout fails under
  concurrent load. Fetch once with `snapshot_download(...)` and
  `HF_HOME=/scratch/zhuominc/hf`, then run with `HF_HUB_OFFLINE=1`.
- **Don't `pkill -f` inside `kubectl exec`** — the pattern matches the exec'd
  shell's own process tree and kills your session (exit 143).
- **Long waits need a live pod.** A job whose command exits (or finishes its
  `sleep`) is reaped along with anything still running in it; size the idle
  window to the work, not the other way round.
- **`queue` goes under `resourceConfig`**, not at the top level, and is mutually
  exclusive with `nodePoolNames`.
