# Evaluation

The evaluator owns every input of the comparison. It generates the sessions,
runs every registered solver on the same sessions, reads back only the raw
candidate scores, and recomputes every metric itself.

## Sessions

Each session contains:

- `stimuli`: N face latents (N x 512, float32);
- `eeg`: the EEG response recorded for each stimulus (N x p, row i belongs
  to stimulus i);
- `candidates`: 60 face latents (60 x 512), exactly one of which is the hidden
  target.

How sessions are built:

- **Geometry.** Stimuli and candidates are drawn from one target-independent
  distribution: uniform on a 2-sphere of radius 17.3 lying in a random 3-D
  affine subspace of the 512-D latent space. Every point on that sphere is
  equivalent, so geometry alone carries no information about the target. The
  target is a uniformly random candidate. The expected candidate-target
  distance is 23.1, the same as the chance level of the paper's ranking task.
- **EEG responses.** Each EEG row is a real recorded response. The released
  data contain responses recorded at 60 known stimulus-target distances
  between 0 and 46.2. Each stimulus receives an unused row recorded at the
  distance nearest to its own distance from the hidden target.
- **Session-specific EEG basis.** Within each session the EEG features are
  z-scored and multiplied by a random orthogonal matrix. Feature identity
  (channel, time window, embedding unit) is therefore not preserved, but
  rotation-invariant structure is.
- **Control sessions.** Mixed into every split are control sessions whose EEG
  rows are permuted across stimuli. The solver is not told which sessions are
  controls.

Each split holds 48 scored sessions and 16 control sessions per experiment. The
train split (`eval/train/`) is visible and can be exported with its answers:

```
python eval/train/evaluate_cursor.py --export-sessions DIR
```

The hidden test (`eval/test/`) uses the same generator with a hidden seed.

## Experiments

| Experiment | EEG representation | N per session | Weight |
|---|---|---:|---:|
| `unbiased_ranking_raw_eeg` | windowed raw EEG, 29 channels x 7 windows = 203 features | 3000 | 0.50 |
| `unbiased_ranking_eegnet` | released 176-D EEGNet embedding, same sessions as above | 3000 | 0.25 |
| `unbiased_ranking_raw_eeg_low_data` | windowed raw EEG, independent sessions | 1000 | 0.25 |

## Solver interface

A solver is registered by a manifest `outputs/solvers/<name>.json` of the form
`{"command": ["python", "my_solver.py", "--sessions", "{sessions}", "--output", "{output}"]}`.
The command runs from the workspace root. `{sessions}` is a directory holding
`index.json` (session ids and the EEG representation) and one `<id>.npz` per
session; `data/cursor_data.py:read_session_dir` reads it. The solver writes
`{"scores": {"<session id>": [60 floats]}}` to `{output}`. A higher score means
the candidate is more likely to be the target.

`scripts/evaluate.sh` / `scripts/evaluate_train.sh` score every registered
solver on the train sessions (`scoring/scores.json`, `scoring/scores_train.json`).
`scripts/evaluate_test.sh` scores them on the hidden test
(`scoring/scores_test.json`).

## Metrics (per experiment, mean over the 48 scored sessions)

Ties among scores are resolved as the expectation over uniformly random
tie-breaking.

- `top_rank_distance` (primary, lower is better): latent distance from the
  top-scored candidate to the target.
- `target_rank` (lower): rank of the target among the 60 candidates.
- `score_distance_pearson_r` (lower): Pearson correlation between scores and
  the candidates' distances to the target; constant scores count as 0.
- `target_top1_accuracy`, `target_top5_accuracy`, `reciprocal_target_rank`
  (higher).
- `near_target_top_rank_rate` (higher): fraction of sessions whose top
  candidate is within distance 5 of the target.

Random guessing gives rank 30.5, top-rank distance about 23, r = 0, and top-1
accuracy 1/60.

## Gates (a failed gate scores the solver as failed on that experiment)

These outcomes count as a failure:

- the command fails or times out;
- the output is unreadable;
- any session is missing;
- any session has the wrong number of scores;
- any score is non-finite.

The control-session gate also applies. On control sessions every solver is at
chance in expectation (mean rank 30.5). A solver whose mean control-session rank
is below 30.5 − 4 standard errors (13.2) gets its ranking from something other
than the EEG/stimulus pairing, and fails.

A failed solver receives the worst attainable value of every metric: r = +1,
rank 60, maximal distance, and zero accuracies.
