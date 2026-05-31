# Overview

This reproduction implements CURSOR from “Self-Calibrating BCIs: Ranking and
Recovery of Mental Targets Without Labels.”

- **Paper ID:** 2506.11151
- **Title:** Self-Calibrating BCIs: Ranking and Recovery of Mental Targets
  Without Labels
- **Domain:** Brain-computer interfaces / EEG target recovery
- **TL;DR:** The paper ranks possible mental targets from unlabeled
  EEG-stimulus interactions by testing which candidate makes the observed brain
  responses internally predictive of candidate-specific distances.

## Short Summary

The benchmark asks a solver to rank candidate face embeddings for an unknown
mental target using only paired stimulus embeddings and EEG responses. CURSOR
scores each candidate by comparing held-out regression error for aligned EEG
against a shuffled control. This workspace reproduces the core ranking claim on
the released dataset with paper-style synthetic training stimuli and random
candidate hypotheses.

## Key Results

- Five-target synthetic-random run: LR achieved `target_rank=1.4` and
  `top_rank_distance=0.023`, far better than shuffled and dummy controls.
- All 17 targets at 300 observations per target: LR retained a positive control
  gap (`control_gap_top_rank_distance=11.524`) with zero label and decoder use.
