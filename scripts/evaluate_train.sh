#!/bin/bash
# Train-slice evaluation: runs every solver registered in outputs/solvers/ on
# the visible train sessions (eval/train/) and writes scoring/scores_train.json.
# Use `python eval/train/evaluate_cursor.py --export-sessions DIR` to export
# these sessions together with their answers for development.

set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate_cursor.py --solvers-dir outputs/solvers --scores scoring/scores_train.json
