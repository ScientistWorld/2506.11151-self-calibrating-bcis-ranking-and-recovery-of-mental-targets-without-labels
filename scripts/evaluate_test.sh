#!/bin/bash
# Held-out test-slice evaluation entry point. It mirrors scripts/evaluate.sh's
# artifact contract, but writes scores_test.json and enforces the method keys
# that were visible in scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  INPUTS=("$@")
else
  INPUTS=(outputs/cursor_ranking.json outputs/cursor_ranking_eegnet_n500.json)
fi

python eval/test/evaluate_cursor.py \
  --input "${INPUTS[@]}" \
  --scores scoring/scores_test.json \
  --train-scores scoring/scores_train.json
