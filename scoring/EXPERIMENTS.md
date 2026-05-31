# Experiment Overview

## unsupervised_ranking_synthetic_random

Tests whether a system can recover useful face-target rankings from unlabeled
BCI interaction records. For each target condition, the system receives a set of
synthetic-random candidate face embeddings and the available brain/stimulus
observations, then outputs scores that should prioritize candidates close to the
intended target.

This experiment is intended to establish both sides of the paper's central
claim: the ranking should be meaningfully better than broken or random controls,
and it should achieve that without target labels or pretrained EEG decoders.

## unsupervised_ranking_eegnet

Tests the same recovery setting using the released compact EEG embedding
features. This experiment is intended to check whether the benchmark remains
useful when the neural response representation changes while the task,
candidate set size, and supervision constraints stay fixed.
