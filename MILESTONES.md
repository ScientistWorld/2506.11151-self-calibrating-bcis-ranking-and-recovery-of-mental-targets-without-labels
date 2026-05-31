# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 15:42] - secondary_claims
- Completed all-target raw-EEG CURSOR ranking and all-target EEGNet ranking with zero label supervision and zero pretrained decoder use.
- LR outperformed shuffled and dummy controls on the primary control-gap metric in both scored experiments.
- Tightened the artifact contract so evaluation recomputes candidate distances from emitted candidate embeddings instead of trusting method-supplied distances.

### [2026-05-31 16:04] - core_claim_plus
- Follow-up audit found the scored raw-EEG and EEGNet artifacts were produced before the candidate-embedding artifact contract was enforced.
- Downgraded the active milestone to the reset `core_claim_plus` state until the all-target raw-EEG and EEGNet artifacts are regenerated and rescored under the independent evaluator.

## Stop Justification
- Not stopped; submitting a validation rerun for the revised artifact schema.
