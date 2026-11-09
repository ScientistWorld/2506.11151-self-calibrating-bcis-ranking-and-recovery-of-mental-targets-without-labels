#!/bin/bash
# Hidden-test evaluation: generates the hidden test sessions inside
# eval/test/, runs every solver registered in outputs/solvers/ on them, and
# writes scoring/scores_test.json.  Takes no arguments.

set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate_cursor.py --solvers-dir outputs/solvers --scores scoring/scores_test.json
