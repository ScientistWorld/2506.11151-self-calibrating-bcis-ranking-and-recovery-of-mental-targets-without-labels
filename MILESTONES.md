# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-31 12:04] - core_claim_plus
- The five-target synthetic-random run completed successfully with LR `target_rank=1.4`, `top_rank_distance=0.023`, and `control_gap_top_rank_distance=21.509`.
- The shuffled and dummy controls remained much worse while supervision audit metrics stayed at zero.
- Submitting an all-target synthetic-random run with 1000 observations per target to expand toward the paper's Table 1 setting.

### [2026-05-31 13:09] - core_claim_plus
- The all-target 1000-observation CPU run timed out after 3600 seconds before writing final scores.
- Added per-target progress logging and partial artifact writes so long runs are diagnosable.
- Retrying the all-target synthetic-random run with 500 observations per target, matching the successful five-target scale.

### [2026-05-31 14:10] - core_claim_plus
- The all-target 500-observation run timed out after completing 14 of 17 targets, so the method is progressing but still slightly too slow for the one-core limit.
- The partial artifact confirms 14 completed target runs were written before timeout.
- Retrying all 17 targets with 300 observations per target while preserving the 10-split CURSOR scoring protocol.

### [2026-05-31 14:18] - core_claim_plus
- The all-target 300-observation run completed successfully across all 17 targets.
- LR retained a useful advantage over shuffled and dummy controls: `top_rank_distance=15.577` vs `29.974` and `27.101`, with `control_gap_top_rank_distance=11.524`.
- Supervision and pretrained-decoder audit metrics remained zero.

## Stop Justification
- Completed at milestone `core_claim_plus`.
