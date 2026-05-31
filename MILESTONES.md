# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 11:45] - none
- Read the paper and released repository.
- Implemented a compact CURSOR ranking runner and independent evaluator.
- Prepared a CPU smoke job to verify the method end-to-end before claiming `method_runs`.

### [2026-05-31 11:47] - method_runs
- CPU test job completed without errors and wrote `outputs/cursor_ranking.json` plus `scoring/scores.json`.
- Verified that `scripts/evaluate.sh` scores method artifacts independently of `method/`.

### [2026-05-31 11:47] - core_claim
- On the compact one-target setting, the proposed label-free ranking method placed a much closer candidate first than both shuffled and dummy controls.
- Supervision audit metrics remained zero for labels and pretrained EEG decoders.
- Submitting an expanded CPU run over more targets to test robustness beyond the smoke setting.

### [2026-05-31 11:57] - core_claim
- The five-target target-session run completed but did not preserve the control gap; shuffled LR ranked closer top candidates than LR on average.
- Replaced the target-session shortcut with the released synthetic-train/random-candidate protocol used for Table 1 scale-downs.
- Restored the upstream shuffled/aligned RMSE ratio scoring and submitted a faithful five-target rerun.

## Stop Justification
