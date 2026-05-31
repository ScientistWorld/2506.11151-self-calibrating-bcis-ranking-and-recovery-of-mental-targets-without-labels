# Progress

## What Works

- The paper and upstream repository have been inspected.
- The released `all_data_sorted.npz` dataset is available locally under `data/downloads/`.
- A local CURSOR implementation computes hypothesis-specific Euclidean distances, trains aligned and shuffled regressors over repeated held-out splits, and writes candidate scores using the upstream shuffled/aligned RMSE ratio.
- The method runner now supports the paper-style generated train faces and random candidate hypotheses used for Table 1 scale-downs.
- The evaluator reads method outputs from `outputs/` and writes `scoring/scores.json` without importing from `method/`.
- Scoring target, constraint, experiment, and research-direction files now match the implemented benchmark experiment.

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

## Remaining

- Run the all-target synthetic-random CPU job with 500 observations per target.
- If the all-target job works, consider adding a diagonal-candidate setting or more estimators for secondary coverage.

## Issues

- The login-node now has enough scientific packages to run validation, but actual reproduction compute remains on submitted jobs.
- The completed five-target target-session run was too noisy and is retained only as a negative diagnostic; current code now uses the paper-style synthetic split protocol by default.

## Deviations from Paper

- Paper setting: full ranking experiments use generated train/test candidate sets and report averages across 17 targets and repeated train splits.
- Completed first job: target-specific observed sessions, 1 target, up to 300 observations per target. The CURSOR scoring algorithm itself is unchanged; only the scale/data split was reduced for a smoke test.
- Completed diagnostic job: target-specific observed sessions, 5 targets, up to 500 observations per target; it did not support a stronger milestone.
- Completed core-plus job: synthetic generated train faces, random candidates with exact target included, 5 targets, up to 500 observations per target.
- Attempted expansion: same synthetic-random protocol across all 17 targets, up to 1000 observations per target; this exceeded the CPU job limit.
- Next job: same all-target protocol with 500 observations per target plus progress logging and partial output writes.
