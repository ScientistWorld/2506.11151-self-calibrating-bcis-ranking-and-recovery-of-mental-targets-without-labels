#!/bin/bash
set -euo pipefail
cd /home/user
trap 'python redteam/audit.py restore' EXIT
if [ ! -f redteam/equivalence.json ]; then
    python redteam/audit.py equivalence
fi
for name in reference_seed0 reference_seed1 reference_seed2 better_more_data weaker_shuffled shortcut_no_eeg_centroid shortcut_label_oracle shortcut_supervision_declared shortcut_bruteforce_catalogue shortcut_control_degradation shortcut_negative_audits shortcut_fixed_dataset shortcut_candidate_collapse shortcut_test_size shortcut_scorefile_forgery shortcut_nonfinite shortcut_null_audit; do
    if [ -f "redteam/$name/status.json" ]; then
        echo "Preserving completed audit: $name"
        continue
    fi
    echo "Running audit: $name"
    bash "redteam/$name/run.sh" > "redteam/$name/training.log" 2>&1
    cat "redteam/$name/status.json"
done
python redteam/report.py
