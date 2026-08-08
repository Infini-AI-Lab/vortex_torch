# CUDA 13 base: sglang 0.5.16 pins cuda-python>=13 and the cu13 flashinfer /
# cutlass-dsl wheels (torch 2.11.0). The 12.8 base the 0.5.9 image used is not
# compatible with that stack.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Architectures the JIT/AOT CUDA extensions are built for.
#   9.0    = H100/H200 (sm_90)
#   10.0a  = B200 (sm_100a, the arch-specific variant the cuda_mla kernel needs;
#            see vortex_torch/engine/sgl/attention_backend/cuda_mla_kernel.py).
# Override at build time, e.g.  --build-arg TORCH_CUDA_ARCH_LIST="10.0a"  for
# a Blackwell-only image, or  "9.0"  for Hopper-only.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0a"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    wget \
    libnuma1 \
    libnuma-dev \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

RUN python -m pip install --upgrade pip setuptools wheel

# Default to v0.6 (the current vortex_torch release line). Override via
#   --build-arg VORTEX_TORCH_REF=<branch-or-tag>.
ARG VORTEX_TORCH_REF=v0.6

WORKDIR /workspace

RUN git clone -b "${VORTEX_TORCH_REF}" --recursive \
    https://github.com/Infini-AI-Lab/vortex_torch.git

# Install the vendored sglang together with vortex_torch in ONE pip resolve.
# sglang v0.5.16 does NOT ship an install.sh; the README's "Method 2: From
# source" (third_party/sglang/v0.5.16/sglang/docs_new/.../install.mdx, also
# https://docs.sglang.io/get_started/install.html) is just
#     pip install -e "python"
# which pulls torch 2.11.0, flashinfer_python[cu13] 0.6.14, sglang-kernel 0.4.5,
# transformers 5.12.1, cuda-python>=13 and the rest of the runtime tree. Both
# editables go in one command so pip solves the whole graph at once — 0.5.16's
# stack is mutually version-locked, and piecewise installs / later per-package
# upgrades silently produce an ABI-mismatched env.
WORKDIR /workspace/vortex_torch

RUN pip install --no-cache-dir \
    -e "third_party/sglang/v0.5.16/sglang/python" \
    -e .

# GLM-4.7-Flash works out of the box here: sglang 0.5.16 pins transformers
# 5.12.1, and `glm4_moe_lite` needs transformers >= 5. (Under the old 0.5.9
# image the 4.57.1 pin required the separate vortex_glm env / override step.)

RUN which python && python --version && which pip && pip --version

# Verify libnuma is visible at runtime
RUN python - <<'PY'
import ctypes
ctypes.CDLL("libnuma.so.1")
print("OK: libnuma.so.1 loaded")
PY


ARG AWS_EFA_INSTALLER_URL=https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz

ENV PATH=/opt/amazon/efa/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    iproute2 \
    cpio \
    linux-headers-aws \
    linux-aws \
    libhwloc15 \
    tar \
    unzip \
    pciutils environment-modules tcl \
    && rm -rf /var/lib/apt/lists/* \
    || (apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    cpio \
    iproute2 \
    libhwloc15 \
    tar \
    unzip \
    pciutils environment-modules tcl \
    && rm -rf /var/lib/apt/lists/*) \
    || true

RUN curl -fsSL -o aws-efa-installer-latest.tar.gz "${AWS_EFA_INSTALLER_URL}"
RUN tar --no-same-owner -xf aws-efa-installer-latest.tar.gz
RUN apt-get update && \
    apt-get install -y linux-aws linux-headers-aws || true
RUN cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --disable-ngc --skip-mpi

RUN plugin_path="$(find /opt/amazon /opt/aws-ofi-nccl -name 'libnccl-net.so' -print -quit 2>/dev/null || true)" \
    && test -n "${plugin_path}" \
    && ldconfig \
    && for required_lib in libhwloc.so.15 libfabric.so.1 libefa.so.1 libibverbs.so.1; do \
    ldconfig -p | grep -q "${required_lib}" || exit 1; \
    done \
    && ldd "${plugin_path}" \
    && ! ldd "${plugin_path}" | grep -q "not found" \
    && echo "Found NCCL net plugin at ${plugin_path}"

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]