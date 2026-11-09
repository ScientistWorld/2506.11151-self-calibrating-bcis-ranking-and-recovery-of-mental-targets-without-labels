# Research Direction

## Research Problem

This benchmark concerns self-calibrating brain-computer interfaces for mental
target recovery. The question is whether a system can infer which candidate
mental target is most plausible from unlabeled neural responses and stimulus
embeddings, without explicit calibration labels for the user's target.

## Core Contribution

The paper contributes a label-free ranking approach for continuous target
spaces. Future work should improve how precisely and how data-efficiently the
target is recovered from one session's unlabeled brain/stimulus observations,
while preserving the self-calibrating setting.

## Approach Scope

In scope:

- better estimators or error measures for the hypothesis-consistency score;
- variance reduction (cross-validation schemes, repeated permutations);
- unsupervised representation learning on the session's own EEG;
- exploiting the structure of the candidate set or the score landscape;
- search over hypotheses;
- any method that turns unlabeled stimulus/EEG pairing into a better ranking,
  particularly with few observations.

## Out of Bounds

- Training or applying a supervised EEG decoder built from target identities,
  candidate ranks or recorded distances, including those in the released
  data file.
- Ranking candidates from geometry or priors alone instead of from the neural
  observations. The control-session gate fails solvers whose ranking does not
  come from the EEG/stimulus pairing.
- Reading or reproducing the hidden test generator.
- Adding external labeled EEG/face datasets or pretrained EEG models.
- Changing the evaluation metrics, candidate count, sessions or gates.
