# Paper's Method
# Method

CURSOR scores a hypothetical target by constructing labels internally from that
hypothesis. For a candidate latent vector `h`, every observed stimulus `z_i` is
mapped to a hypothetical distance `||h - z_i||_2`. A regressor is trained to
predict those distances from the paired EEG features. The same regression task
is repeated after deliberately misaligning EEG responses and stimuli.

The CURSOR score is the ratio of held-out RMSE for the shuffled task to held-out
RMSE for the aligned task, averaged over repeated 90/10 splits. If the candidate
matches the participant's true mental target, EEG responses should predict the
candidate distances better than shuffled EEG, giving a larger score. Ranking
candidates by this score recovers the mental target without using labels or a
pretrained EEG decoder.
How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

<!-- What is the paper's main contribution? -->

## Approach

<!-- How does the paper solve the problem? Key ideas, architecture, algorithm. -->

## Main Claims

<!-- What are the paper's central claims? Which are primary vs. secondary? -->
