#!/usr/bin/env bash
# Launch an sglang server with vortex sparse attention.
#
# Usage:  ./server_launch.sh <MODEL_NAME> <TP_SIZE>
# Example: ./server_launch.sh Qwen/Qwen3-4B 1
#
# The vortex_* hyper-parameters are no longer individual CLI flags: sglang's
# ServerArgs now carries a single aggregated `vortex` field, exposed on the CLI
# as `--vortex-config '<json>'` (see vortex_torch/engine/sgl/config.py and
# third_party/.../server_args.py). Passing the old per-knob `--vortex-*` flags
# fails argparse. We therefore write the knobs to a JSON file and feed it
# through `--vortex-config`. Keys are the VortexConfig field names (the
# `vortex_` prefix is stripped). Providing a non-null vortex config implicitly
# enables sparsity.
export OPENAI_API_KEY="None"
MODEL_NAME=$1
TP_SIZE=$2

VORTEX_CONFIG_FILE="$(mktemp /tmp/vortex_config.XXXXXX.json)"
trap 'rm -f "$VORTEX_CONFIG_FILE"' EXIT

cat > "$VORTEX_CONFIG_FILE" <<'JSON'
{
  "impl_backend": "triton",
  "use_tensor_core": true,
  "attention_backend": "trtllm",
  "layers_skip": [],
  "block_reserved_eos": 1,
  "block_reserved_bos": 2,
  "topk_val": 61,
  "block_size": 32,
  "workload_chunk_size": 64,
  "module_name": "block_sparse_attention",
  "max_seq_lens": 32768,
  "max_topk_val": 256,
  "dtype": "bfloat16",
  "compilation_cache_dir": "~/.vortex_compilation_cache"
}
JSON

# NOTE: we cannot use `python -m sglang.launch_server` directly. That entrypoint
# builds `ServerArgs` in the parent process before anything imports vortex_torch,
# so the `--vortex-config` JSON string is never folded into a VortexConfig (the
# `ServerArgs.__init__` adapter that does this is installed by `import
# vortex_torch`). The raw string then gets pickled to the spawned scheduler
# worker, where `server_args.vortex_block_size` -> `getattr(str, "block_size")`
# raises AttributeError. Importing vortex_torch FIRST, in this parent process,
# installs the adapter so the conversion happens before ServerArgs is pickled.
python -c '
import os, sys
import vortex_torch  # installs the ServerArgs adapter + backend integration
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

server_args = prepare_server_args(sys.argv[1:])
try:
    run_server(server_args)
finally:
    kill_process_tree(os.getpid(), include_parent=False)
' \
 --model-path "$MODEL_NAME" \
 --page-size 32 \
 --attention-backend "flashinfer" \
 --vortex-config "$(cat "$VORTEX_CONFIG_FILE")" \
 --context-length 32768 \
 --mem-fraction-static 0.9 \
 --tp-size "$TP_SIZE" \
 --port 30000 \
 --host 127.0.0.1
