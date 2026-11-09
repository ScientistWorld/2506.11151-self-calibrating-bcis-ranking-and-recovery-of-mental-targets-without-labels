# Evaluation Targets

All metrics are recomputed by `eval/` from raw candidate scores on
evaluator-owned sessions: 48 scored sessions plus 16 control sessions per
experiment and split. Ties count as their expected value under random
tie-breaking.

## Primary: unbiased_ranking_raw_eeg (weight 0.5)

Rank 60 candidates for a hidden target from 3000 unlabeled stimulus / raw-EEG
pairs per session.

- **Primary metric**: `top_rank_distance` (lower is better), the paper's
  preferred, perceptually grounded metric.
- **Supporting metrics**:
  - `target_rank` (lower)
  - `score_distance_pearson_r` (lower)
  - `target_top1_accuracy` (higher)
  - `target_top5_accuracy` (higher)
  - `reciprocal_target_rank` (higher)
  - `near_target_top_rank_rate` (higher)
- **Reference (CURSOR-LR, hidden test)**: distance 3.89, rank 2.69, r = −0.33,
  top-1 0.48. The paper's controls are at chance: S-LR scores distance 22.9 and
  rank 31.2; Dummy scores distance 22.5 and rank 30.5.
- **How to evaluate**: `bash scripts/evaluate_train.sh` (visible) and
  `bash scripts/evaluate_test.sh` (hidden). Both score every solver
  registered in `outputs/solvers/`.

## Secondary: unbiased_ranking_eegnet (weight 0.25)

The same sessions with the 176-D EEGNet embedding. This tests the paper's
claim that the framework is not tied to one EEG representation.

- **Metrics**: as above.
- **Reference**: distance 6.06, rank 2.52.

## Secondary: unbiased_ranking_raw_eeg_low_data (weight 0.25)

Independent sessions with 1000 pairs each. This is the paper's data-size axis
(Figure 3), where the method is weakest.

- **Metrics**: as above.
- **Reference**: distance 15.39, rank 14.94. Chance is distance 22.8, rank 30.5.

## Constraints: zero supervision, signal from the EEG/stimulus pairing (gates)

These are eligibility conditions, not traded-off metrics. A violation scores
the solver as failed on that experiment, with the worst value of every metric.

- **Labels never reach the solver.** Sessions carry no labels. EEG features
  arrive standardised and randomly rotated per session, so rows cannot be
  looked up in the public labelled file.
- **Valid output.** Every session must receive 60 finite scores, and the
  command must succeed.
- **Control sessions at chance.** Control sessions break the EEG/stimulus
  pairing. Their mean target rank must not fall below 30.5 − 4 SE (13.2).
  This is the paper's own control logic: the information must come from the
  EEG.
