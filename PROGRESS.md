# Progress

## What Works

- The paper and upstream repository have been inspected.
- The released `all_data_sorted.npz` dataset is available locally under `data/downloads/`.
- A local CURSOR implementation computes hypothesis-specific Euclidean distances, trains aligned and shuffled regressors over repeated held-out splits, and writes candidate scores.
- The evaluator reads method outputs from `outputs/` and writes `scoring/scores.json` without importing from `method/`.
- Scoring target, constraint, experiment, and research-direction files now match the implemented benchmark experiment.

## Results

- Job `16ddec1d-905` completed successfully on a compact one-target setting.
- Reproduced scores for `unsupervised_ranking_target_sessions`:
  - `lr`: `top_rank_distance=5.883`, `target_rank=31.0`, `score_distance_pearson_r=-0.162`, `control_gap_top_rank_distance=12.219`
  - `s_lr`: `top_rank_distance=18.102`, `target_rank=50.0`, `score_distance_pearson_r=-0.042`
  - `dummy`: `top_rank_distance=22.627`, `target_rank=52.0`, `score_distance_pearson_r=0.000`
- The main method beats both controls on top-ranked candidate distance while preserving zero label supervision and zero pretrained decoder use, supporting the core claim at small scale.

## Remaining

- Run the expanded CPU job over five targets and inspect whether the control gap persists.
- If the target-session scale remains noisy, switch to the upstream synthetic split generation protocol before attempting `core_claim_plus`.

## Issues

- The login-node now has enough scientific packages to run validation, but actual reproduction compute remains on submitted jobs.
- The reproduction currently uses target-specific observed sessions rather than the full upstream synthetic split generation. This is a scale/protocol deviation; it should be expanded or replaced for stronger milestones.

## Deviations from Paper

- Paper setting: full ranking experiments use generated train/test candidate sets and report averages across 17 targets and repeated train splits.
- Completed first job: target-specific observed sessions, 1 target, up to 300 observations per target. The CURSOR scoring algorithm itself is unchanged; only the scale/data split was reduced for a smoke test.
- Next job: target-specific observed sessions, 5 targets, up to 500 observations per target.
