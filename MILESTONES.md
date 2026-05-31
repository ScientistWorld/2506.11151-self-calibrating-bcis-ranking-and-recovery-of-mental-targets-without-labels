# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-31 15:42] - secondary_claims
- Completed all-target raw-EEG CURSOR ranking and all-target EEGNet ranking with zero label supervision and zero pretrained decoder use.
- LR outperformed shuffled and dummy controls on the primary control-gap metric in both scored experiments.
- Tightened the artifact contract so evaluation recomputes candidate distances from emitted candidate embeddings instead of trusting method-supplied distances.

## Stop Justification
- Not stopped; submitting a validation rerun for the revised artifact schema.
