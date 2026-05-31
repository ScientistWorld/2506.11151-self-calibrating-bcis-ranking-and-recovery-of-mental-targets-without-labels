# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 14:20] - core_claim_plus
- Implemented CURSOR's actual hypothesis scoring algorithm with held-out aligned and shuffled regressors.
- Completed a five-target paper-style synthetic-random ranking run where LR strongly outperformed controls.
- Completed an all-target synthetic-random ranking run at 300 observations per target where LR retained a positive control gap.

## Stop Justification
- Not stopped; EEGNet N=300 did not strongly reproduce the secondary claim, so
  submitting a larger EEGNet N=500 run.
