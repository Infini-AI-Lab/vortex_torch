python examples/misc/benchmark_openai_ttft_tpot.py \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key None \
  --model Qwen/Qwen3-4B \
  --request-rates 1,2,3,4,5,6,7,8 \
  --duration-s 120 \
  --max-tokens 512 \
  --prompt-file examples/validation_4K.jsonl \
  --prompt-field input \
  --tokenizer Qwen/Qwen3-4B \
  --output-dir benchmark/benchmark_ttft_tpot_bsa64_qwen3-4b_4k


python examples/misc/benchmark_openai_ttft_tpot.py \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key None \
  --model Qwen/Qwen3-4B \
  --request-rates 1,2,3,4,5,6,7,8 \
  --duration-s 120 \
  --max-tokens 512 \
  --prompt-file examples/validation_8K.jsonl \
  --prompt-field input \
  --tokenizer Qwen/Qwen3-4B \
  --output-dir benchmark/benchmark_ttft_tpot_bsa64_qwen3-4b_8k



python examples/misc/benchmark_openai_ttft_tpot.py \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key None \
  --model Qwen/Qwen3-4B \
  --request-rates 1,2,3,4,5,6,7,8 \
  --duration-s 120 \
  --max-tokens 512 \
  --prompt-file examples/validation_16K.jsonl \
  --prompt-field input \
  --tokenizer Qwen/Qwen3-4B \
  --output-dir benchmark/benchmark_ttft_tpot_bsa64_qwen3-4b_16k

