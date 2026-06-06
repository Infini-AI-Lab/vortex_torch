
export OPENAI_API_KEY="None"
MODEL_NAME=$1
TP_SIZE=$2
TOPK_VAL=$3
PORT=$4
VORTEX_MODULE_NAME=${5:-block_sparse_attention}
BASE_URL="http://127.0.0.1:${PORT}/v1"
RESERVED_BOS=2
RESERVED_EOS=1
MODEL_TAG=$(basename "$MODEL_NAME")
MODEL_TAG=$(printf "%s" "$MODEL_TAG" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]._-' '-')
MODULE_TAG=$(printf "%s" "$VORTEX_MODULE_NAME" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]._-' '-')
OUTPUT_PREFIX="benchmark/benchmark_ttft_tpot_full_attention_topk${TOPK_VAL}_${MODEL_TAG}"

python -m sglang.launch_server \
 --model-path "$MODEL_NAME" \
 --page-size 16 \
 --disable-overlap-schedule \
 --attention-backend "flashinfer" \
 --vortex-layers-skip 0 \
 --vortex-block-reserved-eos "$RESERVED_EOS" \
 --vortex-block-reserved-bos "$RESERVED_BOS" \
 --vortex-topk-val "$TOPK_VAL" \
 --vortex-block-size 16 \
 --vortex-workload-chunk-size 32 \
 --vortex-module-name "$VORTEX_MODULE_NAME" \
 --vortex-max-seq-lens 32768 \
 --context-length 32768 \
 --mem-fraction-static 0.9 \
 --vortex-compilation-cache-dir "~/.vortex_compilation_cache" \
 --tp-size "$TP_SIZE" \
 --port "$PORT" \
 --host 127.0.0.1 &

SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

echo "Started server with PID ${SERVER_PID} on ${BASE_URL}"
echo "Benchmark model: ${MODEL_NAME}"
echo "Vortex module: ${VORTEX_MODULE_NAME}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Sleeping 120s before benchmarking..."
sleep 300

python examples/misc/benchmark_openai_ttft_tpot.py \
  --base-url "$BASE_URL" \
  --api-key None \
  --model "$MODEL_NAME" \
  --request-rates 1,2,3,4,5,6,7,8 \
  --duration-s 120 \
  --max-tokens 512 \
  --prompt-file examples/validation_4K.jsonl \
  --prompt-field input \
  --tokenizer "$MODEL_NAME" \
  --output-dir "${OUTPUT_PREFIX}_4k"

python examples/misc/benchmark_openai_ttft_tpot.py \
  --base-url "$BASE_URL" \
  --api-key None \
  --model "$MODEL_NAME" \
  --request-rates 1,2,3,4,5,6,7,8 \
  --duration-s 120 \
  --max-tokens 512 \
  --prompt-file examples/validation_8K.jsonl \
  --prompt-field input \
  --tokenizer "$MODEL_NAME" \
  --output-dir "${OUTPUT_PREFIX}_8k"


python examples/misc/benchmark_openai_ttft_tpot.py \
  --base-url "$BASE_URL" \
  --api-key None \
  --model "$MODEL_NAME" \
  --request-rates 1,2,3,4,5,6,7,8 \
  --duration-s 120 \
  --max-tokens 512 \
  --prompt-file examples/validation_16K.jsonl \
  --prompt-field input \
  --tokenizer "$MODEL_NAME" \
  --output-dir "${OUTPUT_PREFIX}_16k"
