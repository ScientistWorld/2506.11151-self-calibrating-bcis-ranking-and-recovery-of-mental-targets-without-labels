# Problem Definition

The task is to recover a continuous mental target from an interaction session
containing paired visual stimuli and EEG responses, without calibration labels.
Each visual stimulus is represented by a face latent vector and each response by
an EEG feature vector. During each session, the participant has one target face
in mind, but the solver only receives the stimulus vectors and corresponding EEG
responses.

## Research Question

Can a system rank candidate face targets so that candidates closer to the
participant's intended mental target receive better scores, using only
unlabeled interaction data from the session?

## Why It Matters

A successful solution would reduce or remove explicit labeled calibration for
brain-computer interfaces that need to infer user intent over continuous target
spaces.

## Success Criteria

A good solver should rank near-target candidates above distant candidates,
perform better than broken or random controls, and preserve the self-calibrating
setting: no target labels and no pretrained EEG decoder.
