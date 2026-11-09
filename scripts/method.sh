#!/bin/bash
# Run the paper's proposed method.
#
# CURSOR needs no training: it scores each session's candidates from that
# session's unlabeled stimulus/EEG pairs.  This script registers the CURSOR-LR
# solver (outputs/solvers/lr.json), which the evaluators run on their own
# sessions, and then scores the visible train sessions.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python method/register_solvers.py lr
bash scripts/evaluate.sh
