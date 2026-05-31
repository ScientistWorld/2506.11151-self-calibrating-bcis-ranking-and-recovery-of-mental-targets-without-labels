# Research Direction

## Research Problem

This benchmark concerns self-calibrating brain-computer interfaces for mental
target recovery. The research question is whether a system can infer which
candidate mental target is most plausible from unlabeled neural responses and
stimulus embeddings, without explicit calibration labels for the user's target.

## Core Contribution

The paper contributes a label-free ranking approach for continuous target
spaces. Future work should improve target ranking from unlabeled
brain/stimulus observations while preserving the self-calibrating setting.

## Approach Scope

In-scope work may change how unlabeled observations are used to score candidate
targets, how candidate sets are sampled or normalized within the fixed
benchmark, and how robustness controls are incorporated. Methods should remain
within the same problem family: unsupervised or self-supervised target ranking
from paired neural responses and stimulus representations.

## Out of Bounds

- Training a supervised EEG decoder using target identities, candidate ranks, or
  target distances for the evaluated targets.
- Replacing the task with image recognition, face retrieval from visual inputs
  alone, or a generative prior that ignores the neural observations.
- Adding external labeled EEG/face datasets or pretrained EEG-to-image models to
  solve the benchmark by resource scaling.
- Changing the evaluation metrics or candidate-count protocol to make scores
  incomparable with the benchmark.
