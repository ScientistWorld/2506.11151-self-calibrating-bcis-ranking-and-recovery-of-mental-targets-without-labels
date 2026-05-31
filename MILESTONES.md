# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-31 14:20] - core_claim_plus
- Implemented CURSOR's actual hypothesis scoring algorithm with held-out aligned and shuffled regressors.
- Completed a five-target paper-style synthetic-random ranking run where LR strongly outperformed controls.
- Completed an all-target synthetic-random ranking run at 300 observations per target where LR retained a positive control gap.

### [2026-05-31 15:38] - secondary_claims
- Completed an all-target EEGNet representation run at 500 observations per target.
- Reproduced the secondary representation trend: LR reaches target rank 6.18 and top-distance control gap 13.39 while controls remain much worse.
- Preserved the self-calibrating constraints: zero target labels and zero pretrained EEG decoder use.

## Stop Justification
- Completed at milestone `secondary_claims`.
