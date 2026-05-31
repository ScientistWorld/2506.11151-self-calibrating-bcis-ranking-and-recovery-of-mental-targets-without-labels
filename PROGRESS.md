# Progress

## What Works

- The paper and upstream repository have been inspected.
- The released `all_data_sorted.npz` dataset is available locally under `data/downloads/`.
- A local CURSOR implementation computes hypothesis-specific Euclidean distances, trains aligned and shuffled regressors over repeated held-out splits, and writes candidate scores.
- The evaluator reads method outputs from `outputs/` and writes `scoring/scores.json` without importing from `method/`.

## Results

- No reproduced scores yet. A CPU test job is being submitted to run the first end-to-end ranking experiment.

## Remaining

- Run the CPU job and inspect `scoring/scores.json`.
- If the target-session scale is too small to support the claim, increase target count or observations, or switch to the upstream synthetic split generation protocol.
- Fill final scoring guidance files once scores exist.

## Issues

- The login-node Python lacks scientific packages, so numerical validation is deferred to the submitted container job.
- The reproduction currently uses target-specific observed sessions rather than the full upstream synthetic split generation. This is a scale/protocol deviation to validate quickly; it may need expansion before `core_claim`.

## Deviations from Paper

- Paper setting: full ranking experiments use generated train/test candidate sets and report averages across 17 targets and repeated train splits.
- Current first job: target-specific observed sessions, 3 targets, up to 500 observations per target. The CURSOR scoring algorithm itself is unchanged; only the scale/data split is reduced for a smoke test.
