#!/usr/bin/env bash
# Chunk 4/5 of examples/misc/run_minimax.sh — jobs [16..18] of 19 (last chunk, 3 jobs):
#   [16] gqa_quest_sparse_attention   topk=125  block=32  (budget 4096)
#   [17] gqa_quest_sparse_attention   topk=29   block=64  (budget 2048)
#   [18] gqa_quest_sparse_attention   topk=61   block=64  (budget 4096)
#
# Usage:
#   ./run_minimax_4.sh                                  # defaults
#   ./run_minimax_4.sh <summary_dir>                    # override summary dir
#   ./run_minimax_4.sh <summary_dir> <hf-model-id>      # override summary + model
#   ./run_minimax_4.sh <summary_dir> <hf-model-id> <data>  # also override data path
#   GPUS="0 1 2 3 4 5 6 7" ./run_minimax_4.sh
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 16 3
