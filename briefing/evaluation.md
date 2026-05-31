# Evaluation
# Evaluation

The main evaluation ranks a fixed candidate set of 60 face hypotheses for each
target session. The evaluator reads candidate scores produced by a method and
compares them with the true Euclidean distances from each candidate to the
held-out target.

Metrics:

- `score_distance_pearson_r`: Pearson correlation between method score and true
  distance. More negative values indicate better similarity ranking.
- `target_rank`: one-indexed rank of the exact target when candidates are sorted
  by descending score. Lower is better.
- `top_rank_distance`: Euclidean distance between the top-ranked candidate and
  the true target. Lower is better.
- `control_gap_top_rank_distance`: gap between the best control top-distance
  and the proposed method top-distance. Higher is better.
- `label_supervision_count` and `pretrained_decoder_count`: audit constraints;
  both must remain zero.
How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

<!-- What metrics are used? How are they computed? What does higher/lower mean? -->

## Baselines and Targets

<!-- What performance levels exist as reference points? Include the paper's reported numbers as targets, and any standard baselines. A scientist agent should know what "good" looks like. -->

## Evaluation Protocol

<!-- How should evaluation be run? What data split, what conditions? Any important details about how to compare fairly (e.g., fixed seeds, specific test sets, averaging over runs). -->
