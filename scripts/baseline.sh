#!/bin/bash
# Register the paper's control conditions (S-LR and Dummy) as solvers and
# score them on the visible train sessions.
#
# Usage:
#   bash scripts/baseline.sh            # s_lr and dummy
#   bash scripts/baseline.sh s_lr       # one control

set -e

cd /home/user

python baseline/run_baseline.py "$@"
bash scripts/evaluate.sh
