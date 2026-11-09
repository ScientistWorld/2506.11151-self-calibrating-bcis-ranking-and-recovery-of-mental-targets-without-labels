# Scientist Constraints

## Data

The only data source is the released BCI dataset that `scripts/download.sh`
fetches into `data/downloads/all_data_sorted.npz`, as presented to your solver
through the evaluator's sessions.

- Do not add external EEG datasets, face datasets, human labels, or target
  annotations.
- The released file contains the recorded stimulus-target distance of every
  EEG row. Using those distances, or any other label, to decode or look up the
  EEG of a session is supervision, and is out of bounds.

## Evaluation Protocol

- Submit a solver as a manifest in `outputs/solvers/<name>.json` whose command
  reads a session directory and writes one finite score per candidate per
  session. The evaluator runs it on the visible train sessions and on the
  hidden test sessions.
- Do not read, import, or reproduce anything from `eval/test/`.
- Do not change the session generator, the metrics or the gates in `eval/`.
- Your solver must not depend on knowing which sessions are controls.

## Supervision

- No training on labels that identify the target, candidate ranks, or
  candidate distances.
- No pretrained EEG-to-distance, EEG-to-face or EEG-to-embedding decoder.
- The visible train sessions and their exported answers are for development
  only. Your solver must not use them as training labels for a decoder that
  it then applies to new sessions.
- Each session must be calibrated from its own unlabeled stimulus/EEG pairs
  and candidates.

## Resource Scaling

The solver runs inside the hidden test.

- Keep it under the evaluator timeout of 3 hours per experiment on one CPU.
- Do not improve scores by adding candidate-generation sources or external
  models.
- More computation per session (more cross-validation, better estimators,
  search over candidates) is in scope.

## Compliance Checklist

- [ ] Uses only the evaluator-provided session inputs at scoring time
- [ ] Does not use recorded distances or other labels to decode EEG
- [ ] Does not introduce a pretrained EEG decoder or additional labeled EEG data
- [ ] Does not read or reproduce `eval/test/`
- [ ] Produces 60 finite scores for every session within the timeout
