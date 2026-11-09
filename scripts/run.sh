#!/bin/bash
# Smoke test of the hardened gym: tiny audit pass, then the full-size
# official reproduction (reference solvers through train and hidden test).
set -uo pipefail
cd /home/user
N=$(nproc)
export OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N
echo "cpus=$N"; df -h /tmp | tail -1
python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
OUT=redteam/hardening/smoke DESIGNS=k2 \
  SMOKE_ARGS="--n-aligned 4 --n-control 2 --n-observations 600" \
  PAPER_ARGS="--repeats 1 --n-observations 2000" \
  bash redteam/hardening/run_all.sh
time bash scripts/reproduce.sh > outputs/reproduce.log 2>&1; echo "reproduce rc=$?"
grep -E "solver|failed|Error" outputs/reproduce.log | tail -40
mkdir -p redteam/hardening/smoke
cp outputs/reproduce.log scoring/scores*.json outputs/paper_protocol_check.json redteam/hardening/smoke/ 2>/dev/null
python redteam/hardening/report.py --results redteam/hardening/smoke | tail -40
