# Overview

This reproduction implements CURSOR from “Self-Calibrating BCIs: Ranking and
Recovery of Mental Targets Without Labels.”

- **Paper ID:** 2506.11151
- **Title:** Self-Calibrating BCIs: Ranking and Recovery of Mental Targets
  Without Labels
- **Domain:** Brain-computer interfaces / EEG target recovery
- **TL;DR:** The paper ranks possible mental targets from unlabeled
  EEG-stimulus interactions by testing which candidate makes the observed brain
  responses internally predictive of candidate-specific distances.

## Short Summary

The benchmark asks a solver to rank candidate face embeddings for an unknown
mental target using only paired stimulus embeddings and EEG responses. CURSOR
scores each candidate by comparing held-out regression error for aligned EEG
against a shuffled control. This workspace reproduces the core ranking claim on
the released dataset with paper-style synthetic training stimuli and random
candidate hypotheses.

## Key Results

- Five-target synthetic-random run: LR achieved `target_rank=1.4` and
  `top_rank_distance=0.023`, far better than shuffled and dummy controls.
- All 17 targets at 300 observations per target: LR retained a positive control
  gap (`control_gap_top_rank_distance=11.524`) with zero label and decoder use.

---

# Problem Definition

The task is to recover a continuous mental target from an interaction session
containing paired visual stimuli and EEG responses, without calibration labels.
Each visual stimulus is represented by a face latent vector and each response by
an EEG feature vector. During each session, the participant has one target face
in mind, but the solver only receives the stimulus vectors and corresponding EEG
responses.

## Research Question

Can a system rank candidate face targets so that candidates closer to the
participant's intended mental target receive better scores, using only
unlabeled interaction data from the session?

## Why It Matters

A successful solution would reduce or remove explicit labeled calibration for
brain-computer interfaces that need to infer user intent over continuous target
spaces.

## Success Criteria

A good solver should rank near-target candidates above distant candidates,
perform better than broken or random controls, and preserve the self-calibrating
setting: no target labels and no pretrained EEG decoder.

---

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

---

# Evaluation

The evaluation ranks a fixed candidate set of 60 face hypotheses for each target
condition. The evaluator reads candidate scores produced by a solver and
compares them with the true Euclidean distances from each candidate to the held
out target. The target is used only by the evaluator after scores are written.

## Metrics

- `score_distance_pearson_r`: Pearson correlation between solver score and true
  candidate distance. Lower is better because high scores should correspond to
  nearby candidates.
- `target_rank`: one-indexed rank of the exact target when candidates are sorted
  by descending score. Lower is better.
- `top_rank_distance`: Euclidean distance between the top-ranked candidate and
  the true target. Lower is better.
- `target_top5_accuracy`: fraction of target conditions where the exact target
  appears in the top five ranked candidates. Higher is better and this is the
  primary metric for paper-consistency checks.
- `control_gap_top_rank_distance`: difference between the best control
  top-ranked distance and the evaluated method's top-ranked distance. Higher is
  better and this keeps the comparative control claim in the gym score.
- `label_supervision_count`: number of target labels used by the solver. Must
  remain zero for the self-calibrating setting.
- `pretrained_decoder_count`: number of pretrained EEG decoders used by the
  solver. Must remain zero.

## Baselines and Targets

The reference values in `scoring/reference.json` are the paper's reported
ranking metrics for the random-candidate condition. Evaluated artifacts may
include solver variants and broken or random controls, but the evaluator only
assumes that each method key provides one score per candidate. Top-five target
recovery is the formal primary metric because it is directly comparable to the
paper tables. The control-gap and absolute-distance metrics remain scored so
the scaled reproduction still tests the comparative self-calibrating claim.

## Evaluation Protocol

Methods write JSON artifacts such as `outputs/cursor_ranking.json` and
`outputs/cursor_ranking_eegnet_n500.json` with candidate embeddings, candidate
scores, target indexes, and audit metadata. The artifacts must not supply true
candidate distances; `scripts/evaluate.sh` calls only code in `eval/`, reloads
the dataset through `data/`, recomputes target distances independently, and
writes `scoring/scores.json`, which mirrors the experiment and metric structure
of `scoring/reference.json`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 21m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-31 11:45] - none
- Read the paper and released repository.
- Implemented a compact CURSOR ranking runner and independent evaluator.
- Prepared a CPU smoke job to verify the method end-to-end before claiming `method_runs`.

### [2026-05-31 11:47] - method_runs
- CPU test job completed without errors and wrote `outputs/cursor_ranking.json` plus `scoring/scores.json`.
- Verified that `scripts/evaluate.sh` scores method artifacts independently of `method/`.

### [2026-05-31 11:47] - core_claim
- On the compact one-target setting, the proposed label-free ranking method placed a much closer candidate first than both shuffled and dummy controls.
- Supervision audit metrics remained zero for labels and pretrained EEG decoders.
- Submitting an expanded CPU run over more targets to test robustness beyond the smoke setting.

### [2026-05-31 11:57] - core_claim
- The five-target target-session run completed but did not preserve the control gap; shuffled LR ranked closer top candidates than LR on average.
- Replaced the target-session shortcut with the released synthetic-train/random-candidate protocol used for Table 1 scale-downs.
- Restored the upstream shuffled/aligned RMSE ratio scoring and submitted a faithful five-target rerun.

### [2026-05-31 12:04] - core_claim_plus
- The five-target synthetic-random run completed successfully with LR `target_rank=1.4`, `top_rank_distance=0.023`, and `control_gap_top_rank_distance=21.509`.
- The shuffled and dummy controls remained much worse while supervision audit metrics stayed at zero.
- Submitting an all-target synthetic-random run with 1000 observations per target to expand toward the paper's Table 1 setting.

### [2026-05-31 13:09] - core_claim_plus
- The all-target 1000-observation CPU run timed out after 3600 seconds before writing final scores.
- Added per-target progress logging and partial artifact writes so long runs are diagnosable.
- Retrying the all-target synthetic-random run with 500 observations per target, matching the successful five-target scale.

### [2026-05-31 14:10] - core_claim_plus
- The all-target 500-observation run timed out after completing 14 of 17 targets, so the method is progressing but still slightly too slow for the one-core limit.
- The partial artifact confirms 14 completed target runs were written before timeout.
- Retrying all 17 targets with 300 observations per target while preserving the 10-split CURSOR scoring protocol.

### [2026-05-31 14:18] - core_claim_plus
- The all-target 300-observation run completed successfully across all 17 targets.
- LR retained a useful advantage over shuffled and dummy controls: `top_rank_distance=15.577` vs `29.974` and `27.101`, with `control_gap_top_rank_distance=11.524`.
- Supervision and pretrained-decoder audit metrics remained zero.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-31 14:20] - core_claim_plus
- Implemented CURSOR's actual hypothesis scoring algorithm with held-out aligned and shuffled regressors.
- Completed a five-target paper-style synthetic-random ranking run where LR strongly outperformed controls.
- Completed an all-target synthetic-random ranking run at 300 observations per target where LR retained a positive control gap.

### [2026-05-31 15:38] - secondary_claims
- Completed an all-target EEGNet representation run at 500 observations per target.
- Reproduced the secondary representation trend: LR reaches target rank 6.18 and top-distance control gap 13.39 while controls remain much worse.
- Preserved the self-calibrating constraints: zero target labels and zero pretrained EEG decoder use.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 11 total (9 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 11 total (9 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 11 total (9 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
