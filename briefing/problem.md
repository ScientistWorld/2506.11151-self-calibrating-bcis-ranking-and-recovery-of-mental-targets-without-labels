# Problem Definition

The task is to recover a continuous mental target from an interaction session
containing paired visual stimuli and EEG responses, without calibration labels.
Each visual stimulus is a face latent vector (512-D, from a face generator) and
each response is an EEG feature vector recorded while that face was shown.
During a session the participant holds one target face in mind; the solver
receives only the stimulus vectors, the paired EEG responses, and a set of 60
candidate face latents, exactly one of which is the participant's target.

## Research Question

Can a system rank candidate face targets so that candidates closer to the
participant's intended mental target receive better scores, using only the
unlabeled interaction data of that session?

## Why It Matters

A successful solution removes the labeled calibration phase of
brain-computer interfaces that must infer user intent over continuous target
spaces (here, any face in a generator's latent space).

## What Makes It Hard

- No labels: the solver is never told which stimuli resembled the target, nor
  the target's identity, and no pretrained EEG-to-distance decoder may be used.
- The stimuli and the candidates are drawn independently of the target, so the
  stimulus/candidate geometry by itself says nothing about which candidate is
  the target; the information is only in how the EEG responses co-vary with
  the stimuli.
- EEG responses are noisy single trials; how strongly they reflect the
  perceived similarity to the target is unknown and session-specific (EEG
  features arrive in a session-specific basis).
- Sessions can be short: one experiment has only 1000 stimulus/response pairs.

## Success Criteria

A good solver ranks the target and near-target candidates above distant ones,
does clearly better than chance and than broken controls, and gets its signal
from the pairing between EEG responses and stimuli rather than from any other
source.
