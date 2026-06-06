#!/usr/bin/env bash
# Chunk 2/5 of examples/misc/run_minimax.sh — jobs [8..11] of 19:
#   [8]  gqa_quest_sparse_attention   topk=125  block=16
#   [9]  gqa_quest_sparse_attention   topk=157  block=16
#   [10] gqa_quest_sparse_attention   topk=253  block=16
#   [11] block_sparse_attention       topk=61   block=32  (budget 2048)
#
# Usage:
#   ./run_minimax_2.sh                                  # defaults
#   ./run_minimax_2.sh <summary_dir>                    # override summary dir
#   ./run_minimax_2.sh <summary_dir> <hf-model-id>      # override summary + model
#   ./run_minimax_2.sh <summary_dir> <hf-model-id> <data>  # also override data path
#   GPUS="0 1 2 3 4 5 6 7" ./run_minimax_2.sh
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 8 4
