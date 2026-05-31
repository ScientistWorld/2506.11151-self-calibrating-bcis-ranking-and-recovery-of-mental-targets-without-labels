#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python method/run_cursor.py \
  --methods lr s_lr dummy \
  --n-targets "${N_TARGETS:-3}" \
  --n-observations "${N_OBSERVATIONS:-500}" \
  --n-candidates 60 \
  --n-splits 10 \
  --seed "${SEED:-0}" \
  --output outputs/cursor_ranking.json

bash scripts/evaluate.sh outputs/cursor_ranking.json
