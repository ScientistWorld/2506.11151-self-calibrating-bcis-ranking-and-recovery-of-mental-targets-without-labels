#!/bin/bash
set -euo pipefail
cd /home/user
# This is an ordinary CPU job. No evaluator or Python-environment changes.
python redteam/audit.py equivalence
bash redteam/reference_seed0/run.sh > redteam/reference_seed0/training.log 2>&1
bash redteam/reference_seed1/run.sh > redteam/reference_seed1/training.log 2>&1
bash redteam/reference_seed2/run.sh > redteam/reference_seed2/training.log 2>&1
bash redteam/better_more_data/run.sh > redteam/better_more_data/training.log 2>&1
bash redteam/weaker_shuffled/run.sh > redteam/weaker_shuffled/training.log 2>&1
bash redteam/shortcut_no_eeg_centroid/run.sh > redteam/shortcut_no_eeg_centroid/training.log 2>&1
bash redteam/shortcut_label_oracle/run.sh > redteam/shortcut_label_oracle/training.log 2>&1
bash redteam/shortcut_supervision_declared/run.sh > redteam/shortcut_supervision_declared/training.log 2>&1
bash redteam/shortcut_bruteforce_catalogue/run.sh > redteam/shortcut_bruteforce_catalogue/training.log 2>&1
bash redteam/shortcut_control_degradation/run.sh > redteam/shortcut_control_degradation/training.log 2>&1
bash redteam/shortcut_negative_audits/run.sh > redteam/shortcut_negative_audits/training.log 2>&1
bash redteam/shortcut_fixed_dataset/run.sh > redteam/shortcut_fixed_dataset/training.log 2>&1
bash redteam/shortcut_candidate_collapse/run.sh > redteam/shortcut_candidate_collapse/training.log 2>&1
bash redteam/shortcut_test_size/run.sh > redteam/shortcut_test_size/training.log 2>&1
bash redteam/shortcut_scorefile_forgery/run.sh > redteam/shortcut_scorefile_forgery/training.log 2>&1
bash redteam/shortcut_nonfinite/run.sh > redteam/shortcut_nonfinite/training.log 2>&1
bash redteam/shortcut_null_audit/run.sh > redteam/shortcut_null_audit/training.log 2>&1
python redteam/report.py
python redteam/audit.py restore
