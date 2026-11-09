# Hardening report

The test now rewards science: a genuinely better method scores higher than the
reference, by much more than the test's noise. A weaker method and the paper's
broken controls score lower. Every shortcut the red team found is either at
chance or scored as failed.

## What was wrong

`REDTEAM.md` found that nine of twelve shortcuts outranked the reference.
There were two root causes.

1. **The method owned the test inputs.** Method artifacts supplied the
   candidates, the dataset path, the run list and the control scores, plus
   self-declared supervision counts. The hidden "test" was a hash-selected half
   of whatever runs the method chose to write.
2. **The geometry gave the answer away.** The gym followed the paper's
   simulation protocol (Section 6.1). Every stimulus is placed at a recorded
   distance in a random direction around the target, and every candidate at a
   uniform distance from it. Stimulus and candidate clouds are therefore
   centred on the target. On this protocol at the paper's full scale
   (N = 9234, 17 targets × 2 repeats, `redteam/hardening/results/paper_protocol_check.json`):
   - CURSOR-LR ranks the target first in every run (r = −0.973, distance 0);
   - a no-EEG ranking by distance to the stimulus centroid gets rank 1.03 and
     distance 0.

   The paper's own protocol cannot tell CURSOR from a trivial shortcut. The
   paper acknowledges this "acquisition bias" itself.

The 17 recorded targets were also public, so the target catalogue was
enumerable.

## What changed

1. **The evaluator owns every input.** `eval/train/` and `eval/test/` are
   independent copies sharing no code; they differ only in seed.
   - The evaluator generates the sessions itself. Each has stimuli, real EEG
     rows and 60 candidates, one of them a fresh hidden target.
   - It runs every solver registered in `outputs/solvers/*.json` on identical
     sessions and reads back only raw scores.
   - It recomputes every metric.

   No method name is hardcoded. `scripts/evaluate_test.sh` takes no
   arguments. The hidden seed and generator live only in `eval/test/`.
2. **Target-independent, homogeneous geometry.** Stimuli and candidates are
   uniform on a 2-sphere of radius 17.3 in a random 3-D subspace of the 512-D
   latent space. The target is a uniformly random candidate. The expected
   candidate-target distance is 23.07, matching the paper's random baseline of
   23.08.
   - Each stimulus gets an unused recorded EEG row at the recorded distance
     nearest its distance to the target. This is the paper's
     "equivalent dataset" assumption that EEG encodes distance to the target.
   - The CURSOR algorithm and its procedure are unchanged.
3. **Session-specific EEG basis.** Within each session, EEG features are
   z-scored and rotated by a random orthogonal matrix, so rows cannot be
   looked up in the public labelled file. CURSOR-LR is invariant: scores
   change by at most 3.3e-9 and candidate order is identical
   (`redteam/hardening/equivalence.json`).
4. **Constraints became gates, not tradeoffs.** The self-reported
   `label_supervision_count`, the self-reported `pretrained_decoder_count` and
   the method-supplied `control_gap` were removed. A solver run is scored as
   failed, with the worst attainable value of every metric, when:
   - any score is missing, misshapen or non-finite;
   - the command fails or times out;
   - its 16 hidden control sessions (EEG rows permuted across stimuli) rank
     better than chance − 4 SE.
5. **Ties count at their expected value under random tie-breaking.** Constant
   scores are therefore exactly at chance.
6. **Larger test.** Each experiment now has 48 scored + 16 control hidden
   sessions (previously 8 runs), all paired across solvers.
7. **Headroom experiment.** The paper calls data quantity its main practical
   limitation (Fig. 3, App. F.1). `unbiased_ranking_raw_eeg_low_data` uses
   independent sessions with N = 1000 pairs. Weights are now 0.50 (raw EEG,
   paper's Table 1), 0.25 (EEGNet, Appendix Table 2) and 0.25 (low data),
   previously 0.65 / 0.35. Primary metric: `top_rank_distance`, which the paper
   calls its most informative metric.
8. **`scoring/reference.json`.** The paper's numbers are kept unchanged under
   `_paper_results`, since the redesigned setup is not on the paper's scale.
   `results` holds the gym reference run.
   - The old file listed LR `target_top5_accuracy = 0.67` as a paper number.
     The paper's Table 3 reports 0.565, recorded under `_paper_results`.
9. **Reproduction speed.** The batched CURSOR equals scikit-learn per-candidate
   LR to 7e-16, which made full-scale runs feasible.

## Does a better result now score higher?

Gain = Σ_exp weight × Σ_metric |coef| × improvement / SD. The improvement is
taken in each metric's better direction; SD is the population SD over the 19
valid attempts in this audit. All runs are on the unchanged hidden test with no
arguments. The CI is a 95% bootstrap over the 48 paired sessions.

| Run | Before (old gym) | After | Paired 95% CI, sessions + / − |
|---|---:|---:|---|
| Reference CURSOR-LR, seed 0 | 0 | **0** | |
| Reference seeds 1 / 2 | +3.03 / +2.41 (spread 3.03) | +0.75 / −0.10 (spread **0.86**) | n.s. |
| **Better**: exact leave-one-out CV, 20 shuffles (`better_loo`) | — | **+3.70** | [+2.73, +4.67], 41 / 7 |
| **Better**: 100 ShuffleSplit folds (`better_cv100`) | — | **+2.19** | [+1.44, +2.91], 41 / 7 |
| Better (old gym): 1000 instead of 300/500 observations | +4.12, but below seed spread vs best seed | n/a: eval now fixes N | |
| Kernel smoothing of scores (tried, not better) | — | −0.24 | [−0.73, +0.21] |
| **Weaker**: 25% of each session's observations | — | **−6.37** | [−7.62, −5.07], 4 / 44 |
| Weaker (old gym): shuffled-EEG predictions | −4.95 | ≈ S-LR below | |
| Paper control S-LR | −4.66 | **−8.55** | [−9.68, −7.42], 0 / 48 |
| Paper control Dummy | −6.32 | **−8.73** | [−9.55, −7.93], 0 / 48 |

Both better variants beat the reference by 2.5–4× the seed spread, with CIs
excluding zero. The weaker variant and the controls fall far below it.

Raw / EEGNet / low-data details for leave-one-out vs the reference:

| | Reference | Leave-one-out |
|---|---|---|
| Rank | 2.69 / 2.52 / 14.94 | 1.10 / 1.35 / 4.23 |
| Top-rank distance | 3.89 / 6.06 / 15.39 | 0.38 / 0.86 / 8.66 |

Train and test agree for an honest method (test vs train):

| Experiment | Test rank / distance | Train rank / distance |
|---|---|---|
| Raw EEG | 2.69 / 3.89 | 2.56 / 6.09 |
| EEGNet | 2.52 / 6.06 | 3.40 / 8.67 |
| Low data | 14.94 / 15.39 | 13.85 / 13.49 |

## Shortcuts

Each shortcut was rerun as an ordinary registered solver through the hidden
test. Code: `redteam/hardening/shortcuts.py`; scores:
`redteam/hardening/results/s2/test_shortcuts.json`.

| Shortcut (red-team name) | Before | Fix | After |
|---|---:|---|---:|
| No-EEG stimulus centroid (`shortcut_no_eeg_centroid`) | +8.80 | target-independent geometry | −8.83 (rank 30.4) |
| Stimulus density / nearest stimulus (new) | — | same | −8.91 |
| Candidate centrality, no EEG (new) | — | homogeneous sphere instead of a Gaussian cloud | −7.07 (r −0.18, rank 27.4) |
| CURSOR + centrality prior, weight 0.3 / 1.0 (new) | — | same | −0.32 [−1.08, +0.37] / −2.36 |
| Label oracle (`shortcut_label_oracle`, `shortcut_supervision_declared`) | +8.80 / +8.80 | targets are fresh, hidden in `eval/test/`; sessions carry no labels; supervision counts no longer scored | not expressible |
| Public catalogue enumeration (`shortcut_bruteforce_catalogue`) | +8.80 | fresh hidden targets | −8.88 |
| EEG-row lookup in the public labelled file (new) | trivially perfect once eval owns sessions | per-session z-score + random rotation | −7.72 (chance) |
| Negative / null audit counts | +8.02 / crash | audit counts removed | not expressible |
| Fixed dataset path (`shortcut_fixed_dataset`) | +10.81 | eval loads its own data; method supplies only scores | not expressible |
| Candidate collapse (`shortcut_candidate_collapse`) | +4.40 | eval owns candidates; ties at expectation | constant scores −8.73 (exact chance) |
| Run-subset selection (`shortcut_test_size`) | +7.60 | eval owns run set and size; missing session = failed | `shortcut_missing`: **failed** (worst values, −21.08) |
| Control degradation (`shortcut_control_degradation`) | +1.15 | controls are not method-supplied; constraint is an eval-run control gate | not expressible |
| Non-finite scores (`shortcut_nonfinite`) | NaN accepted, exit 0 | non-finite = failed | **failed** (−21.08) |
| Score-file forgery (`shortcut_scorefile_forgery`) | −6.32 | eval writes last | forged file overwritten; −8.73 |
| Solver that reads the hidden generator (new probe) | — | control-session gate | **failed**: control rank 1.00 < 13.18 |

On the earlier Gaussian-cloud version of the redesign (`k2`), the geometry
still carried a prior: central candidates are closer to any random target.
- A no-EEG centrality ranking reached r = −0.58 and distance 16.0 (chance
  22.6).
- Blending it into CURSOR paid +0.50 [+0.15, +0.87].

That is why the final geometry is a sphere, where every candidate is
equivalent. All designs' results are in `redteam/hardening/results/summary.json`.

## Shortcuts left open, and why

- **Registration attack on the public file.** A solver could estimate the
  session rotation by aligning the session's EEG covariance eigenvectors with
  those of the public file, then match rows and read their recorded distances.
  This is a deliberate use of labels, forbidden by `CONSTRAINTS.md`. It is
  research-level work, not a few lines; the naive lookup is at chance. A
  general invertible (non-orthogonal) mixing would make it harder, but would
  also destroy PCA/ridge-type structure that honest methods may use. Left for
  the grader.
- **Reading `eval/test/`.** A solver runs inside the hidden test and could
  read the generator. A naive oracle is failed by the control gate. One that
  also regenerates the control sessions and answers randomly on them would
  evade the gate. Forbidden by `CONSTRAINTS.md`; left for the grader.
- **Residual sample-level centrality on the sphere.** The 60 sampled
  candidates are not perfectly uniform, so a no-EEG centrality ranking keeps
  r ≈ −0.18. It is at chance on rank, and blending it into CURSOR does not
  pay. Left.

## Reference score, before and after

The tests differ, so the two rows are not comparable.

| | Raw EEG | EEGNet | Low data |
|---|---|---|---|
| Before (8 hidden runs, method-chosen) | rank 22.1, distance 19.2, top-5 0.50 | rank 3.1, distance 19.6, top-5 0.88 | — |
| After (48 hidden sessions each) | rank 2.69, distance 3.89, top-1 0.48, top-5 0.90, r −0.33 | rank 2.52, distance 6.06, top-1 0.42 | rank 14.94, distance 15.39, top-1 0.08 |

S-LR and Dummy are at chance in every experiment, which is the paper's
direction: LR ≫ S-LR ≈ Dummy.

## Headroom check

- **Metrics at their bound.** The reference is at the bound of no scored
  metric. Closest is `target_top5_accuracy` = 0.896 (raw and EEGNet).
- **One simple variant nearly saturates the raw-EEG experiment.** Exact
  leave-one-out reaches top-5 1.00, near-target rate 0.98 and rank 1.10 there.
  So I added the low-data experiment, built from the paper's own data-size
  ablation. There, the reference is at rank 14.9 and leave-one-out at 4.2,
  both far from rank 1. Correlation also stays far from its bound everywhere:
  leave-one-out reaches −0.40 against a bound of −1.
- **Baselines make sense.** S-LR and Dummy sit at chance (distance ~22.5,
  rank ~30). The reference is clearly below them, on a scale that leaves room
  above it.
- **Brute force no longer solves the test.** Enumerating candidates without
  EEG gives chance, because the geometry is exchangeable.

## Design sensitivity

All rows below are hidden test, CURSOR-LR raw / EEGNet, with the paired
leave-one-out gain.

| Design | Rank | Distance | Leave-one-out gain | No-EEG centroid r / distance |
|---|---|---|---|---|
| Gaussian 2-D, σ 13, N 3000 | 3.56 / 2.83 | 5.65 / 6.66 | +1.39 | −0.56 / 16.4 |
| Gaussian 2-D, N 6000 | 3.23 / 4.06 | 5.72 / 5.22 | +0.61 | −0.51 / 16.9 |
| Gaussian 2-D, σ 9 | 2.50 / 3.04 | 2.40 / 3.01 | +1.12 | −0.57 / 11.0 |
| Gaussian 3-D | 3.88 / 4.29 | 5.76 / 9.97 | +2.05 | −0.51 / 17.6 |
| Gaussian 5-D | 10.38 / 6.27 | 15.24 / 14.06 | +5.16 | −0.49 / 17.2 |
| **Sphere (chosen)** | 2.69 / 2.52 | 3.89 / 6.06 | +3.70 | +0.01 / 23.1 |

## What I could not fix, plainly

- **The scored setup is not the paper's simulation protocol.** That protocol
  is solvable without EEG, so its numbers (paper: rank 6.63, distance 2.90;
  ours at N = 9234: rank 1.0, distance 0) appear only as a diagnostic. Our
  regenerated stimuli make the paper's protocol easier than the paper reports,
  though the direction (LR ≫ controls) matches.
- **Gains depend on the cohort.** The SDs used for normalisation come from
  this audit's 19 attempts; future cohorts will normalise differently.
- **Validator warnings.** `validate.py` warns that `eval/` "regenerates the
  experiment". This is by design: eval runs the registered solvers and scores
  their raw output. All checks pass with `--post-design --compare`.
- **Old red-team scripts.** `redteam/<case>/run.sh` from the previous
  iteration target the removed artifact interface (`method/run_cursor.py`) and
  no longer replay. Their saved evidence is kept. The new audit replays with
  `bash redteam/hardening/run_all.sh`.
- **A smaller milestone item.** The paper's optimisation and user-study claims
  (CMA-ES recovery, H-ID) are not part of the gym. The milestone stays
  `secondary_claims`.

## Reproduce

`bash scripts/reproduce.sh` (CPU, ~8 min) registers `lr`, `s_lr` and `dummy`,
writes `scoring/scores{,_train,_test}.json`, and runs the paper-protocol
diagnostic. The full audit is `bash redteam/hardening/run_all.sh`, which takes
about 1.5 h on one CPU and resumes from `.done` markers.
