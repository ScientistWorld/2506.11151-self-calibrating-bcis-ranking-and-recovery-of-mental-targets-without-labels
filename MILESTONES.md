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

### [2026-05-31 16:48] - core_claim_plus
- Job `f55aef50-ae1` regenerated the all-target raw-EEG artifact under the current artifact contract: 17 complete runs, candidate embeddings present, audit metadata present, and no method-supplied ground-truth distances.
- Submitting the all-target EEGNet rerun next so the secondary representation claim can be restored with both scored artifacts evaluated together.

## Stop Justification
- Not stopped; submitting the EEGNet validation rerun for the revised artifact schema.
