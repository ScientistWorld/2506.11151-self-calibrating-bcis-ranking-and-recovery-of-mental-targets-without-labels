#!/bin/bash
# Train-slice evaluation entry point. Mirrors scripts/evaluate.sh: it reads
# pre-generated method output artifacts and writes a score JSON, without
# importing or running method code.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  INPUTS=("$@")
else
  INPUTS=(outputs/cursor_ranking.json outputs/cursor_ranking_eegnet_n500.json)
fi

python eval/train/evaluate_cursor.py --input "${INPUTS[@]}" --scores scoring/scores_train.json
