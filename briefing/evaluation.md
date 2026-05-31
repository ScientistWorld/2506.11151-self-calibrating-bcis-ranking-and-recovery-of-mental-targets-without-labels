# Evaluation

The evaluation ranks a fixed candidate set of 60 face hypotheses for each target
condition. The evaluator reads candidate scores produced by a solver and
compares them with the true Euclidean distances from each candidate to the held
out target. The target is used only by the evaluator after scores are written.

## Metrics

- `score_distance_pearson_r`: Pearson correlation between solver score and true
  candidate distance. Lower is better because high scores should correspond to
  nearby candidates.
- `target_rank`: one-indexed rank of the exact target when candidates are sorted
  by descending score. Lower is better.
- `top_rank_distance`: Euclidean distance between the top-ranked candidate and
  the true target. Lower is better.
- `target_top5_accuracy`: fraction of target conditions where the exact target
  appears in the top five ranked candidates. Higher is better and this is the
  primary metric for paper-consistency checks.
- `control_gap_top_rank_distance`: difference between the best control
  top-ranked distance and the evaluated method's top-ranked distance. Higher is
  better and this keeps the comparative control claim in the gym score.
- `label_supervision_count`: number of target labels used by the solver. Must
  remain zero for the self-calibrating setting.
- `pretrained_decoder_count`: number of pretrained EEG decoders used by the
  solver. Must remain zero.

## Baselines and Targets

The reference values in `scoring/reference.json` are the paper's reported
ranking metrics for the random-candidate condition. Evaluated artifacts may
include solver variants and broken or random controls, but the evaluator only
assumes that each method key provides one score per candidate. Top-five target
recovery is the formal primary metric because it is directly comparable to the
paper tables. The control-gap and absolute-distance metrics remain scored so
the scaled reproduction still tests the comparative self-calibrating claim.

## Evaluation Protocol

Methods write JSON artifacts such as `outputs/cursor_ranking.json` and
`outputs/cursor_ranking_eegnet_n500.json` with candidate embeddings, candidate
scores, target indexes, and audit metadata. The artifacts must not supply true
candidate distances; `scripts/evaluate.sh` calls only code in `eval/`, reloads
the dataset through `data/`, recomputes target distances independently, and
writes `scoring/scores.json`, which mirrors the experiment and metric structure
of `scoring/reference.json`.
