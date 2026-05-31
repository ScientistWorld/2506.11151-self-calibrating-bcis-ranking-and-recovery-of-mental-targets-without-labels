# Paper's Method

CURSOR scores a hypothetical mental target by constructing labels internally
from that hypothesis. For a candidate latent vector `h`, each observed stimulus
embedding `z_i` is mapped to a hypothetical distance `||h - z_i||_2`. A
regressor is trained to predict these distances from paired EEG features, and
the same task is repeated after deliberately misaligning EEG responses and
stimuli.

The released Table 1 analysis computes the candidate score as the held-out
shuffled/aligned RMSE ratio, averaged over repeated 90/10 random splits. If the
candidate matches the user's true mental target, aligned EEG should predict
candidate distances better than shuffled EEG and produce a ratio above chance.
Ranking candidates by this score recovers the target without using target
labels or a pretrained EEG decoder.

For the main ranking experiment, the released setup generates synthetic train
stimulus embeddings around each true target at the same distance distribution
as the recorded stimulus-target pairs. Main ranking test candidates are sampled
at uniformly random distances from the true target with the exact target
included among the candidate hypotheses. The paper reports ranking quality for
60 candidates per target using LR, shuffled LR, dummy, SVR, and MLP estimators.

## Key Contribution

The key contribution is a self-calibrating score for continuous mental-target
recovery: the method evaluates whether unlabeled EEG/stimulus observations are
more internally consistent for one candidate target than for alternatives.

## Main Claims

- CURSOR ranks hypotheses by similarity to the unknown mental target without
  calibration labels.
- Effective estimators such as linear regression outperform shuffled and dummy
  controls, showing that the signal comes from EEG/stimulus alignment.
- The recovered target distances are useful as labels for downstream
  calibration-style tasks.
