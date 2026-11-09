#!/bin/bash
# Reproduce the gym reference end to end (CPU only, ~10 minutes).
#
#   1. register CURSOR-LR and the paper's S-LR / Dummy controls as solvers;
#   2. score them on the visible train sessions (scores.json, scores_train.json);
#   3. score them on the hidden test sessions (scores_test.json);
#   4. run the paper's own simulation protocol at full scale as a
#      reproduction diagnostic (outputs/paper_protocol_check.json).

set -e

cd /home/user

rm -f outputs/solvers/*.json
python method/register_solvers.py lr
python baseline/run_baseline.py s_lr dummy
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python method/paper_protocol_check.py --output outputs/paper_protocol_check.json
