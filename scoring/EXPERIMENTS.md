# Experiment Overview

All three experiments use evaluator-generated sessions. In each session, a
solver gets unlabeled stimulus latents, the EEG response recorded for each
stimulus, and 60 candidate latents, exactly one of which is the hidden
target. Stimuli and candidates do not depend on the target. The scored
quantity is how well the solver's candidate scores single out the target and
nearby candidates. Control sessions with broken EEG/stimulus pairing are mixed
in. A solver that still ranks well on them is not using the EEG/stimulus
pairing, and its run is marked as failed.

## unbiased_ranking_raw_eeg (weight 0.5)

Main setting: 3000 stimulus/response pairs per session, with windowed raw EEG
(203 features). It establishes whether label-free target recovery from neural
responses works at all, and how precisely. It also checks that recovery is
clearly better than broken controls.

## unbiased_ranking_eegnet (weight 0.25)

The same sessions with the released EEGNet embedding of each response. It
checks whether the recovery still works, and how well, when the neural
representation changes while the task and supervision constraints stay
fixed.

## unbiased_ranking_raw_eeg_low_data (weight 0.25)

Independent sessions with only 1000 stimulus/response pairs each. This is the
data-scarce regime, where label-free recovery is far from solved. It measures
how efficiently a method uses limited interaction data.
