# Vortex Torch Docker

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t vortex-torch:cu128 .
```

For a specific GPU architecture, pass `TORCH_CUDA_ARCH_LIST`:

```bash
docker build -f docker/Dockerfile -t vortex-torch:h100 \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0" .
```

Common architecture values:

- `8.0`: A100
- `8.9`: RTX 4090, L40
- `9.0`: H100, H200

Run with GPU access:

```bash
docker run --gpus all -it --rm vortex-torch:cu128
```

For development with your local checkout mounted into the container:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/vortex_torch \
  vortex-torch:cu128
```

Inside the mounted container, rerun this after editing CUDA/C++ extension code:

```bash
pip install -e .
```

---

## `Dockerfile.runtime-min` — for hosts with a small image store

`Dockerfile` above builds a **self-contained** vortex image. That needs several GB
in the docker image store, which is not always where the free space is. Docker
splits its storage:

| what | path | typical here |
|---|---|---|
| volumes | `/var/lib/docker` | large |
| **image layers** | `/var/lib/containerd/…/overlayfs` | often the root LV, small |

On such a host, pulling even `nvidia/cuda:*-devel` fails with *no space left on
device* while `docker system df` still reports hundreds of GB of images (it does
not reflect the LV). The fix is to invert the layout: **models on a volume, image
kept tiny, toolchain bind-mounted from the host.**

```bash
docker volume create vortex_models          # lands on the roomy LV
docker build -f docker/Dockerfile.runtime-min -t vortex-runtime-min .

docker run --rm --gpus '"device=0"' \
    -v /home/<user>:/home/<user> \          # SAME path: editable installs + conda
    -v /usr/local/cuda:/usr/local/cuda:ro \ # nvcc for the runtime JIT builds
    -v vortex_models:/models \
    -w /home/<user>/vortex_torch \
    -e HOME=/home/<user> -e HF_HOME=/models \
    -e CUDA_HOME=/usr/local/cuda \
    -e PATH=/home/<user>/anaconda3/envs/vortex_v1/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    vortex-runtime-min python examples/ruler/run_ruler_mha.py --model <hf-id>
```

Verified end-to-end on Llama-3.1-8B-Instruct (RULER 16K, trtllm indexer).

**Three traps:**

1. **No `nvidia` runtime may be registered** and `--gpus` still works, via **CDI**
   (`/var/run/cdi/nvidia.yaml`). A missing runtime is not proof GPUs are unavailable.
2. **`nvcc` shells out to a bare `gcc`/`g++` on PATH and ignores `$CC`.** A prefixed
   conda toolchain (`x86_64-conda-linux-gnu-gcc`) is not enough — the image must
   ship `g++`. That is why `Dockerfile.runtime-min` installs it.
3. **Keep conda's `bin` OFF the container PATH.** It contains an `nvcc`, so
   `CUDA_HOME` gets inferred as `~/anaconda3`, whose `lib64` has no `libcudart`,
   and every extension link fails with `cannot find -lcudart`. Set `CUDA_HOME`
   explicitly.

**Gated HF repos:** pass `HF_TOKEN`, and fetch with
`allow_patterns=['*.safetensors','*.json'], ignore_patterns=['original/*']` — the
`original/*.pth` copies double the download. Then run with `HF_HUB_OFFLINE=1`, or
transformers probes the hub for `tokenizer.model` and 401s on the gated repo.
