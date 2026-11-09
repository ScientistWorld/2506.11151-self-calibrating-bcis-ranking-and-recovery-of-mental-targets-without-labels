#!/bin/bash
# Hardening audit job: equivalence check, paper-protocol reproduction, every
# candidate session design through train and hidden test (reference,
# variants, shortcuts), official reproduction of the default design, report.
set -uo pipefail
cd /home/user
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
bash redteam/hardening/run_all.sh
python validate.py || true
