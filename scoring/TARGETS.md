# Evaluation Targets

## Primary: unsupervised_ranking_synthetic_random

Tests whether synthetic-random candidate face embeddings can be ranked so that
candidates closer to the intended mental target receive better ranks using only
unlabeled brain/stimulus observations from the released BCI dataset.

- **Primary metric**: `target_top5_accuracy` (higher is better), because it is
  directly comparable to the paper's reported target-recovery rate.
- **Supporting metrics**: `control_gap_top_rank_distance` (higher is better),
  `top_rank_distance` (lower is better), `target_rank` (lower is better), and
  `score_distance_pearson_r` (lower is better; scores should decrease as
  candidate distance from the target increases).
- **How to evaluate**: run `scripts/evaluate.sh outputs/cursor_ranking.json`
  after a method writes ranking artifacts in the standard JSON format.

## Secondary: unsupervised_ranking_eegnet

Tests the same label-free candidate ranking task on the released EEGNet
embedding representation. This follows the paper's secondary claim that the
self-calibrating ranking framework is not tied to one EEG preprocessing
representation.

- **Primary metric**: `target_top5_accuracy` (higher is better)
- **Supporting metrics**: `control_gap_top_rank_distance`,
  `top_rank_distance`, `target_rank`, and `score_distance_pearson_r`
- **How to evaluate**: run
  `scripts/evaluate.sh outputs/cursor_ranking.json outputs/cursor_ranking_eegnet_n500.json`
  after both method artifacts exist.

## Constraint: zero_supervision

The task is self-calibrating: the ranking method must not use labels that reveal
which candidate is the intended mental target, and must not use a pretrained EEG
decoder.

- **Metric**: `label_supervision_count` (must remain zero)
- **Metric**: `pretrained_decoder_count` (must remain zero)
- **How to evaluate**: these audit counts are read from the method artifact and
  written to `scoring/scores.json` by `scripts/evaluate.sh`.

## Ablation: control_gap_top_rank_distance

Tests whether a deliberately broken control condition loses the ability to
place near-target candidates at the top of the ranking.

- **Metric**: `control_gap_top_rank_distance` (higher is better)
- **How to evaluate**: include both the main method and the shuffled-control
  result in the method artifact, then run `scripts/evaluate.sh`.
