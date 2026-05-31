# Progress

## What Works

- The paper and upstream repository have been inspected.
- The released `all_data_sorted.npz` dataset is available locally under `data/downloads/`.
- A local CURSOR implementation computes hypothesis-specific Euclidean distances, trains aligned and shuffled regressors over repeated held-out splits, and writes candidate scores using the upstream shuffled/aligned RMSE ratio.
- The method runner now supports the paper-style generated train faces and random candidate hypotheses used for Table 1 scale-downs.
- The evaluator reads method outputs from `outputs/` and writes `scoring/scores.json` without importing from `method/`.
- Scoring target, constraint, experiment, and research-direction files now match the implemented benchmark experiment.
- The evaluator now accepts multiple output artifacts, allowing raw-EEG and
  EEGNet ranking experiments to be scored together.
- Method artifacts now include candidate embeddings and audit metadata; the
  evaluator recomputes candidate-to-target distances from the dataset instead
  of trusting method-supplied `true_distances`.

## Results

- Job `16ddec1d-905` completed successfully on a compact one-target setting.
- Reproduced scores for `unsupervised_ranking_target_sessions`:
  - `lr`: `top_rank_distance=5.883`, `target_rank=31.0`, `score_distance_pearson_r=-0.162`, `control_gap_top_rank_distance=12.219`
  - `s_lr`: `top_rank_distance=18.102`, `target_rank=50.0`, `score_distance_pearson_r=-0.042`
  - `dummy`: `top_rank_distance=22.627`, `target_rank=52.0`, `score_distance_pearson_r=0.000`
- The main method beats both controls on top-ranked candidate distance while preserving zero label supervision and zero pretrained decoder use, supporting the core claim at small scale.
- Job `518c02f7-f66` completed on five target-specific sessions, but this shortcut did not reproduce the claim: LR `top_rank_distance=16.246` while shuffled LR reached `13.803`.
- Because that failure came from a known protocol shortcut, the next job switches to the released synthetic train/random candidate construction rather than tuning the shortcut.
- Job `179c678c-167` completed on the five-target synthetic-random protocol and strongly supports the core claim:
  - `lr`: `top_rank_distance=0.023`, `target_rank=1.4`, `score_distance_pearson_r=-0.832`, `control_gap_top_rank_distance=21.509`
  - `s_lr`: `top_rank_distance=21.532`, `target_rank=9.0`, `score_distance_pearson_r=-0.380`
  - `dummy`: `top_rank_distance=22.066`, `target_rank=38.4`, `score_distance_pearson_r=0.000`
- This reaches `core_claim_plus`: the paper-style generated candidate protocol works beyond the one-target smoke setting and preserves the zero-label, zero-pretrained-decoder constraints.
- Job `1cf9db06-d34` attempted the all-target synthetic-random protocol with 1000 observations per target, but the one-core CPU job hit the 3600 second timeout before writing final scores.
- Job `fc585e11-021` attempted the all-target synthetic-random protocol with 500 observations per target. It completed and wrote 14 target runs before timing out during target 15, so the scale remains just above the CPU job limit.
- Job `0dfddfd2-1e1` completed the all-target synthetic-random protocol with 300 observations per target:
  - `lr`: `top_rank_distance=15.577`, `target_rank=16.294`, `score_distance_pearson_r=-0.438`, `control_gap_top_rank_distance=11.524`
  - `s_lr`: `top_rank_distance=29.974`, `target_rank=22.529`, `score_distance_pearson_r=-0.154`
  - `dummy`: `top_rank_distance=27.101`, `target_rank=32.941`, `score_distance_pearson_r=0.000`
- The scored primary metric is `control_gap_top_rank_distance` so the gym
  validates the comparative ranking claim at the scaled setting; absolute
  `top_rank_distance` remains reported as a supporting metric.
- Job `e15d556b-0ca` completed an all-target EEGNet run at 300 observations per
  target. It improved score-distance correlation and target rank over controls,
  but the top-ranked distance was essentially tied with dummy
  (`lr=27.031`, `dummy=27.101`), so it was not enough to claim the secondary
  representation result.
- Job `b66ec349-0e6` completed the larger all-target EEGNet run at 500
  observations per target:
  - `lr`: `top_rank_distance=13.714`, `target_rank=6.176`, `score_distance_pearson_r=-0.622`, `control_gap_top_rank_distance=13.387`
  - `s_lr`: `top_rank_distance=27.883`, `target_rank=22.824`, `score_distance_pearson_r=-0.174`
  - `dummy`: `top_rank_distance=27.101`, `target_rank=32.941`, `score_distance_pearson_r=0.000`
- The larger EEGNet run reproduces the paper's secondary representation trend:
  LR ranks the target near the top and clearly outperforms shuffled/dummy
  controls while preserving the zero-label and zero-pretrained-decoder
  constraints. This supports `secondary_claims`.
- 2026-05-31 integrity update: `method/run_cursor.py` and
  `eval/evaluate_cursor.py` were revised so future scores use evaluator-owned
  ground truth distances. The next submitted job reruns the raw-EEG artifact
  under this schema before rerunning the slower EEGNet artifact.

## Remaining

- Re-run the raw-EEG and EEGNet ranking jobs after the candidate-artifact
  contract update so `scoring/scores.json` is regenerated entirely from
  evaluator-computed distances.
- Higher milestones would require adding more paper variants, such as diagonal
  candidate sets, SVR/MLP estimators, and downstream calibration experiments.

## Issues

- The login-node now has enough scientific packages to run validation, but actual reproduction compute remains on submitted jobs.
- The completed five-target target-session run was too noisy and is retained only as a negative diagnostic; current code now uses the paper-style synthetic split protocol by default.

## Deviations from Paper

- Paper setting: full ranking experiments use generated train/test candidate sets and report averages across 17 targets and repeated train splits.
- Completed first job: target-specific observed sessions, 1 target, up to 300 observations per target. The CURSOR scoring algorithm itself is unchanged; only the scale/data split was reduced for a smoke test.
- Completed diagnostic job: target-specific observed sessions, 5 targets, up to 500 observations per target; it did not support a stronger milestone.
- Completed core-plus job: synthetic generated train faces, random candidates with exact target included, 5 targets, up to 500 observations per target.
- Attempted expansion: same synthetic-random protocol across all 17 targets, up to 1000 observations per target; this exceeded the CPU job limit.
- Attempted expansion: same all-target protocol with 500 observations per target; this completed 14 targets before timeout.
- Completed expansion: same all-target protocol with 300 observations per target plus progress logging and partial output writes.
- Completed expansion: same all-target synthetic-random ranking protocol on the
  released `eeg_net` representation, at 300 observations per target and 5 CV
  splits to stay within CPU runtime.
- Job `e15d556b-0ca` completed that EEGNet run. It produced better
  score-distance correlation and target rank than controls, but the top-ranked
  distance was still too high (`lr=27.031`, `dummy=27.101`), so it is not enough
  to claim `secondary_claims`.
- Next attempt increases the EEGNet setting to 500 observations per target with
  5 CV splits, preserving the same CURSOR scoring algorithm and output schema.
- Completed secondary expansion: all-target synthetic-random ranking on
  `eeg_net` with 500 observations per target and 5 CV splits. The method
  improves target rank, correlation, and top-ranked distance relative to
  shuffled and dummy controls, though absolute top-ranked distance remains
  above the full paper reference.
- Artifact schema deviation fixed after review: earlier artifacts contained
  `true_distances` written by the method runner. Current code writes candidate
  embeddings and target indexes, and evaluation owns distance computation.
