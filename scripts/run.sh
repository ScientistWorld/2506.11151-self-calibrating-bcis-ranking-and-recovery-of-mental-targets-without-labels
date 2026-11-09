#!/bin/bash
# Reproduce the gym reference end to end (CPU): register CURSOR-LR and the
# paper's S-LR / Dummy controls, score them on the train and hidden test
# sessions, run the paper-protocol diagnostic, and validate the workspace.
# The full hardening audit is `bash redteam/hardening/run_all.sh`.
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/reproduce.sh
python validate.py --post-design --compare
