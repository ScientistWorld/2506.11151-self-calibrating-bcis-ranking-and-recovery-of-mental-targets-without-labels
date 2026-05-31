# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-05-31 17:00] - secondary_claims
- Job `5f2cbf4d-94d` regenerated the all-target EEGNet artifact under the same candidate-embedding contract.
- `scoring/scores.json` now contains both raw-EEG and EEGNet experiments scored by `eval/`, with LR outperforming shuffled and dummy controls while preserving zero label supervision and zero pretrained decoder use.

### [2026-05-31 17:13] - secondary_claims
- Tightened the main evaluator so the reusable `scoring/scores.json` output includes the full ranking metric set already used by the train/test split evaluators.
- Submitting a CPU validation job to regenerate all score files in the packaged environment and run `validate.py --compare`.

### [2026-05-31 17:18] - secondary_claims
- Final verification removed a method-specific phrase from the method-agnostic evaluation briefing and aligned `reference.json` primary metrics with the documented control-gap target.
- Refreshed the top-level score contract so `scoring/scores.json` includes the same ranking metric set as the train/test split score files; `python validate.py --compare` passes.

## Stop Justification
- Completed at milestone `secondary_claims`.
