# Scientist Constraints

## Data

Use only the released BCI face-embedding dataset fetched by
`scripts/download.sh` into `data/downloads/all_data_sorted.npz`. Do not add
external EEG datasets, face datasets, human labels, or manually curated target
annotations. The paper's claim is about recovering target rankings from the
unlabeled interaction records in this dataset, so extra labeled data would
change the problem rather than improve the method.

## Evaluation Protocol

Submit rankings through the artifact schema consumed by `scripts/evaluate.sh`.
Each evaluated target must use the same candidate count and the same metrics as
the benchmark experiment. For the EEGNet experiment, methods must use the
released `eeg_net` array from the same dataset rather than training a new EEG
embedding model. The evaluator must remain independent of `method/`; method
code may write artifacts, but it may not change metric definitions or read
reference scores.

## Supervision

No method may train on labels that identify the intended target, candidate
rank, or candidate distance-to-target for the evaluated target. No pretrained
EEG-to-image, EEG-to-face, or EEG-to-embedding decoder may be introduced. The
allowed input is the unlabeled set of EEG/stimulus observations and the
candidate face embeddings.

## Resource Scaling

Do not improve scores by increasing the problem resources outside the benchmark:
no additional candidate-generation source, no larger external face embedding
model, and no ensemble over separately trained supervised decoders. Changes
should improve the self-calibrating ranking procedure under the same data and
evaluation interface.

## Compliance Checklist

- [ ] Uses `data/downloads/all_data_sorted.npz` as the only data source
- [ ] Does not use target labels, candidate ranks, or candidate distances as
      training labels for the evaluated target
- [ ] Does not introduce a pretrained EEG decoder or additional labeled EEG data
- [ ] Writes method artifacts for `scripts/evaluate.sh` without modifying eval
      metric definitions
- [ ] Keeps the benchmark candidate count and metric names compatible with
      `scoring/reference.json`
