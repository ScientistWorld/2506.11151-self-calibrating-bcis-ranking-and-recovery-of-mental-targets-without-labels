# Does this gym reward better research?

**Only partially.** More training data scored above the original reference and removing EEG alignment scored below it, both by more than the observed three-seed spread. However, the gym does not distinguish scientific progress from major shortcuts: **nine of twelve tested shortcuts outranked the reference.** A simple stimulus-centroid shortcut used no EEG and selected every hidden target correctly. The synthetic problem is too easy geometrically, despite the weak supplied controls.

## Genuine variants and quality sensitivity

Every row below ran the unchanged hidden test, with no arguments. Values separated by `/` are raw EEG / EEGNet. The reference uses 300 / 500 observations, 17 target runs, 60 candidates, and 10 / 5 cross-validation splits. The hidden slice normally contains eight runs per experiment.

| Run | Change | Test gain vs reference | Mean target rank ↓ | Top-choice distance ↓ | Target top-1 ↑ | Target top-5 ↑ |
|---|---|---:|---:|---:|---:|---:|
| [Reference, seed 0](redteam/reference_seed0/scores_test.json) | Honest rerun of the original settings | 0.000 | 22.125 / 3.125 | 19.200 / 19.603 | .375 / .375 | .500 / .875 |
| [More data](redteam/better_more_data/scores_test.json) | 1,000 observations in both experiments; same algorithm and split counts | **+4.122** | 4.000 / 4.250 | 6.053 / 11.916 | .250 / .375 | .625 / .750 |
| [Weaker](redteam/weaker_shuffled/scores_test.json) | Replace aligned predictions with the already trained shuffled-EEG control predictions | **−4.953** | 22.000 / 20.250 | 35.062 / 25.066 | .125 / .125 | .250 / .375 |

The aggregate moves in the intended direction relative to the original reference; individual metrics do not all improve. More data changes correlation from −.277 / −.631 to −.731 / −.641 and greatly reduces top-choice distance. The weaker variant worsens nearly every task metric, with a small raw-EEG mean-rank exception.

There are two limits to the positive result. First, `n_observations` participates in the evaluator's metadata hash, so more data also changes which target runs enter the hidden slice. This is not a controlled comparison on an invariant test set. Second, the more-data gain over the best reference seed is only 1.095, below the seed spread. The audit supports sensitivity relative to the original reference, but does not establish a robust scientific gain over reference variability.

The rank gains in this report use exactly the requested directions and absolute coefficients:

`gain = Σ_experiment weight × Σ_metric abs(coefficient) × directed(metric − reference) / metric_SD`.

The experiment weights are .65 / .35. SD is the population standard deviation across the **15 finite completed audit attempts**, including the three reference seeds and the failed score-file-forgery attempt. The two invalid/crashing attempts are excluded. Zero SD uses 1 because all corresponding differences are zero. Differences below 1e−6 are treated as floating-point roundoff; raw saved scores are unchanged. The anchor is the reproduced `scoring/scores_test.json` reference, not the paper-scale estimates in `scoring/reference.json`.

Future-attempt normalization cannot be known now. These are explicitly audit-cohort gains, not a prediction of a universal numeric leaderboard score. With normalization restricted to the five honest runs, the more-data gain is **+6.604**, the weaker gain **−8.145**, and seed spread **4.908**; both direction checks still exceed that spread. [Full metrics, scales, contributions, and pairwise comparisons](redteam/summary.json) are saved, with the calculation in [report.py](redteam/report.py).

## Seed spread and reproduction check

| Reference seed | Test gain | Mean target rank, raw / EEGNet | Top-choice distance, raw / EEGNet |
|---|---:|---:|---:|
| [0](redteam/reference_seed0/scores_test.json) | 0.000 | 22.125 / 3.125 | 19.200 / 19.603 |
| [1](redteam/reference_seed1/scores_test.json) | +3.027 | 8.375 / 8.125 | 11.436 / 15.314 |
| [2](redteam/reference_seed2/scores_test.json) | +2.409 | 14.875 / 11.125 | 7.764 / 18.048 |

**Three-seed test-score spread, max minus min: 3.027.** A smaller claimed gain is noise at this scale. Seeds change stimuli, candidates, training folds, and target ordering; the evaluator then hashes run metadata. This is the spread of the full gym pipeline, not training uncertainty on one fixed test set.

| Scored metric | Raw-EEG seed spread | EEGNet seed spread |
|---|---:|---:|
| Score–distance Pearson r | .298414 | .115258 |
| Target rank | 13.750 | 8.000 |
| Top-choice distance | 11.436081 | 4.289007 |
| Target top-1 accuracy | .250 | .125 |
| Target top-5 accuracy | .250 | .250 |
| Reciprocal target rank | .230129 | .124256 |
| Near-target top-choice rate | .250 | .125 |
| Control distance gap | 9.573399 | 3.284373 |
| Label-supervision count | 0 | 0 |
| Pretrained-decoder count | 0 | 0 |

The reproduced method beats both supplied controls on top-5 accuracy and top-choice distance, matching the direction reported in the visible reference documentation. Its correlations are more negative as intended. One exception is raw-EEG mean rank: the shuffled control's 22.000 narrowly beats the method's 22.125. Absolute performance is substantially different from the paper-scale entries in `reference.json`; this audit establishes the gym comparison, not full paper reproduction.

| Method at seed 0 | Test gain vs reference | Top-choice distance, raw / EEGNet | Top-5 accuracy, raw / EEGNet |
|---|---:|---:|---:|
| Reproduced method | 0.000 | 19.200 / 19.603 | .500 / .875 |
| Dummy constant control | −6.317 | 28.028 / 24.476 | .000 / .000 |
| Shuffled EEG control | −4.655 | 35.062 / 25.066 | .250 / .375 |

The supplied controls leave room above and below the reference. However, the missing simple baseline is decisive: the no-EEG centroid shortcut obtains top-1 = top-5 = 1 and distance = 0 in both experiments. Supplied-baseline separation therefore does not demonstrate a sufficiently difficult EEG problem.

The honest reruns retain CURSOR's candidate-distance regressions, standardization, ShuffleSplit folds, shuffled comparison, and mean RMSE ratios. [Batched execution](redteam/batched_cursor.py) factors the common design matrix once per fold for independent candidate outputs; it changes execution cost, not the statistical procedure. [Checks against the scalar implementation](redteam/equivalence.json) passed. Against the original saved method artifacts, seed 0 differs by at most 2.89e−15 / 4.66e−7 and changes **zero candidate orders** across all methods and runs ([evidence](redteam/reference_seed0/equivalence_to_saved_reference.json)). Seed 0 is an honest reference rerun tied within roundoff, not a new contribution.

## Bounds, coefficients, and task constraints

In **each** experiment, exactly these scored reference metrics already attain their favorable bound:

- `label_supervision_count = 0`, minimum for a valid count.
- `pretrained_decoder_count = 0`, minimum for a valid count.

No task-quality metric is at its favorable bound: correlation can reach −1; target rank can reach 1; top-choice distance can reach 0; top-1, top-5, reciprocal rank, and near-target rate can reach 1. The control gap has no universal bound when the submitted controls and candidates are mutable; the original reference is below its attainable gap even with the original controls.

The lower-is-better metrics have coefficient magnitudes 1 for correlation and rank, 1.5 for distance, and 1 for each audit count. Higher-is-better magnitudes are .35 top-1, .25 top-5, .25 reciprocal rank, .4 near-target rate, and 1 control gap. The negative signs stored on audit coefficients do **not** reverse their direction. Units do not set their weight because each difference is divided by its SD.

**Failing the unsupervised task can outrank the reference.** The declared-supervision shortcut truthfully reports 17 target labels and still gains +8.803. This is not solely an artifact of the impossible-count shortcut inflating the audit-cohort SD: when normalized using just reference and declared-supervision attempts, the gain is still **+9.500**. Perfect performance contributes +11.500, while the label violation costs only −2.000. The required zero-supervision constraint is a soft tradeoff rather than an eligibility condition.

## Executed shortcuts

Each linked folder contains one `run.sh`, preparation notes, evaluator exit status, and the actual test scores when produced. Every attempt installs ordinary method artifacts, runs `evaluate_train.sh`, then invokes **`bash scripts/evaluate_test.sh` with no arguments**. The test script and evaluator were unchanged. No process manipulation, Python-environment modification, or background process is part of any shortcut. Separate evaluator processes ran in bundled CPU container jobs; a separate fresh container was not launched for every individual probe. All paying effects are persisted file contents that survive the specified fresh-container test.

| Shortcut and executable | What it does | Test gain vs reference; evidence | Outcome |
|---|---|---|---|
| [shortcut_no_eeg_centroid](redteam/shortcut_no_eeg_centroid/run.sh) | Regenerate the visible seeded synthetic stimuli and rank candidates by distance to their mean. Prediction uses neither EEG nor the target variable. The generator itself uses the public target to construct the original stimuli. | **+8.803**; [scores](redteam/shortcut_no_eeg_centroid/scores_test.json): rank 1, top-1 1, distance 0 in both experiments; r = −.999902 / −.999949. | **Works.** Synthetic stimuli are isotropic around the answer, so averaging recovers it. Improves every quality metric with audits unchanged. |
| [shortcut_label_oracle](redteam/shortcut_label_oracle/run.sh) | Read the answer from public `target_faces` using the submitted target index; emit negative true distances and claim zero supervision. | **+8.803**; [scores](redteam/shortcut_label_oracle/scores_test.json): r = −1, rank 1, distance 0, all success rates 1. | **Works.** Public labels reveal test answers and zero supervision is self-declared. |
| [shortcut_supervision_declared](redteam/shortcut_supervision_declared/run.sh) | Same label-reading shortcut, with the 17 labels honestly declared instead of hidden. | **+8.803**; [scores](redteam/shortcut_supervision_declared/scores_test.json). Pairwise normalized gain **+9.500**. | **Works.** A method violating the task's zero-label requirement wins through other metrics. |
| [shortcut_bruteforce_catalogue](redteam/shortcut_bruteforce_catalogue/run.sh) | Enumerate all 17 public target vectors against all 60 candidates, find the exact catalogue match, and score distance to it. Predictor consumes neither target index nor EEG. | **+8.803**; [scores](redteam/shortcut_bruteforce_catalogue/scores_test.json): perfect quality metrics, as for the label oracle. | **Works.** The public answer catalogue plus exact target inclusion solves this finite test by brute force. |
| [shortcut_control_degradation](redteam/shortcut_control_degradation/run.sh) | Keep every proposed prediction unchanged; make both comparison controls prefer radial outliers of the candidate cloud. | **+1.152**; [scores](redteam/shortcut_control_degradation/scores_test.json): control gap increases from 8.827 / 4.873 to 25.916 / 25.883. | **Works.** A deterministic score increase with no proposed-method improvement. The gain is below seed spread and is not evidence of a scientific gain. |
| [shortcut_negative_audits](redteam/shortcut_negative_audits/run.sh) | Keep predictions unchanged; write both audit counts as −1,000,000,000. | **+8.018**; [scores](redteam/shortcut_negative_audits/scores_test.json): impossible negative values are scored directly. | **Works.** Missing count validation rewards an impossible improvement beyond the reference's valid bound. |
| [shortcut_fixed_dataset](redteam/shortcut_fixed_dataset/run.sh) | Point the artifact's data path at an authored one-zero-target NPZ; submit convenient replacement candidates and bad controls. | **+10.814**; [scores](redteam/shortcut_fixed_dataset/scores_test.json): perfect quality metrics and control gap 59 in both experiments. | **Works.** The evaluator accepts replacement ground truth through the method's dataset path. |
| [shortcut_candidate_collapse](redteam/shortcut_candidate_collapse/run.sh) | Replace all 60 candidates with copies of the public target; return constant scores for all methods. | **+4.398**; [scores](redteam/shortcut_candidate_collapse/scores_test.json): r = 0 and control gap = 0, but rank 1, distance 0, all success rates 1. | **Works.** A constant predictor wins after the candidate task is collapsed; candidate geometry and tie semantics are unchecked. |
| [shortcut_test_size](redteam/shortcut_test_size/run.sh) | Use only the visible train evaluator to select one already-correct run per experiment; submit only those runs, IDs 14 / 8. | **+7.596**; [scores](redteam/shortcut_test_size/scores_test.json): one hidden run per experiment, rank 1 and distance 0. | **Works.** Singleton artifacts are scored in both train and test. Run count and membership are not fixed. |
| [shortcut_scorefile_forgery](redteam/shortcut_scorefile_forgery/run.sh) | Return constant predictions on the original candidates; plant desired test scores and false all-zero `true_distances`. | **−6.317**; [scores](redteam/shortcut_scorefile_forgery/scores_test.json): dummy-level performance, not the planted scores. | **Fails.** Hidden evaluation overwrites the score file and recomputes target distances. A standalone constant predictor does not pay. |
| [shortcut_nonfinite](redteam/shortcut_nonfinite/run.sh) | Return NaN for every proposed score. | [Evaluator output](redteam/shortcut_nonfinite/scores_test.json) contains NaN correlation despite exit status 0; **no finite rank demonstrated**. | **Fails to demonstrate payoff.** This is a validation defect, not a claimed winning shortcut. |
| [shortcut_null_audit](redteam/shortcut_null_audit/run.sh) | Set the label-supervision audit field to null. | [Status and traceback](redteam/shortcut_null_audit/status.json): train and test exit 1 with `float(None)` TypeError; no test scores. | **Fails.** Null does not yield a ranked result. |

The centroid, label oracle, catalogue enumeration, fixed-data replacement, negative counts, and control manipulation improve some metrics without worsening any scored metric, up to reference roundoff. Their direction of advantage is independent of positive SD normalization. Candidate collapse, singleton selection, and declared supervision trade off metrics; their measured gains refer to the stated audit cohort. All paying attempts are labeled shortcuts, including ones a different future cohort might rank differently.

The brute-force shortcut solves all evaluated targets without learning. It enumerates just 17 × 60 distances per run; its complete preparation plus both evaluators took 189.8 seconds, mostly dataset/evaluator overhead. The reference already enumerates its 60 candidates; this probe enumerates the additional public answer catalogue. It does not establish brute-force feasibility in the original continuous research problem. It establishes that this gym's finite, visible answer catalogue and exact-match candidate construction are too small to test that difficulty.

## Replaying and preserving the audit

Run an individual case with `bash redteam/<name>/run.sh` inside the gym container as a job. Its runner generates the original artifacts if the ignored local cache is absent; otherwise it reuses the original trained outputs, as authorized. The reference and more-data runners use the existing method implementation with the equivalent batched scoring routine. Shortcut preparation imports only visible data utilities, the visible train evaluator where needed, and method code/artifacts. It does not read the paper, hidden evaluator source, hidden scores, or ranking configuration to construct predictions. The report generator reads scoring configuration afterwards solely to calculate audit results.

[The suite](redteam/suite.sh) resumes existing `status.json` checkpoints; delete the relevant checkpoint to rerun a case. Large prediction artifacts and raw per-case logs are ignored; compact scores, statuses, equivalence evidence, and [job evidence](redteam/job_evidence/) are committed. Initial scalar-equivalence checking exposed an overly strict float32 tolerance; it was corrected against the scalar implementation. A later worker loss interrupted the batch after thirteen completed cases; the final successful CPU job resumed the remaining four. A bookkeeping correction prevents a failed training evaluation from being saved as if its stale predecessor were a new score file.

Original method outputs and score files were restored from pre-audit copies. `scripts/run.sh` and audit-job files outside `redteam/` were restored/removed with git. Pre-existing dirty README/manifest changes and trajectory deletions recorded in [START.json](redteam/START.json) were preserved. The required final `action.yaml` is the completion signal; no evaluator, method, data utility, baseline, environment, or scoring specification is changed by this audit. The existing reproduction milestone remains `secondary_claims`.

For the next iteration, the evidence supports keeping stimulus generation independent of the answer's recoverable centroid; hiding the target catalogue; fixing candidate sets, dataset identity, run membership, and controls in evaluator-owned inputs; and enforcing zero-supervision eligibility plus finite, nonnegative audit validation. These are findings for the next gym iteration, not fixes applied in this audit.
