# Paper's Method

CURSOR (Consistency-based Unsupervised Regression for Similarity-based
Optimization and Ranking) scores a hypothetical mental target by constructing
labels internally from that hypothesis. For a candidate latent `h`, each
observed stimulus `z_i` is mapped to a hypothetical distance `||h - z_i||_2`
(standardised). A regressor is trained to predict these distances from the
paired EEG features, and the same task is repeated after deliberately
misaligning EEG responses and stimuli (a fixed random permutation).

The score `S(h)` is the held-out shuffled/aligned RMSE ratio averaged over 10
random 90/10 ShuffleSplit folds (aligned and shuffled errors are computed on
the identical held-out fold). If `h` is the participant's target, aligned EEG
predicts the hypothesis distances better than shuffled EEG and `S(h)` is
larger. Candidates are ranked by `S(h)`; no target labels and no pretrained
EEG decoder are used.

## Implementation in this gym

- `method/cursor.py`: `cursor_score` is the scalar scikit-learn version
  (`StandardScaler` + `LinearRegression` per candidate); `cursor_scores_fast`
  runs the identical procedure for all 60 candidates at once (shared folds,
  normal equations). They agree to 7e-16 (`redteam/hardening/equivalence.json`).
  OLS with an intercept is invariant to the evaluator's per-session EEG basis.
- `method/score_sessions.py`: the solver (`--estimator lr|s_lr|dummy`).
  `s_lr` (shuffled pairs) and `dummy` (mean predictor) are the paper's
  control conditions; `baseline/run_baseline.py` registers them.
- `method/register_solvers.py`: writes `outputs/solvers/{lr,s_lr,dummy}.json`.
- `method/paper_protocol_check.py`: runs CURSOR on the paper's own
  simulation protocol (target-centred stimuli, uniform-distance candidates,
  N = 9234) as a reproduction diagnostic; not scored.

## Key Contribution

A self-calibrating score for continuous mental-target recovery: the method
asks which candidate makes the unlabeled EEG responses most internally
predictive of candidate-specific stimulus distances.

## Main Claims

- CURSOR ranks hypotheses by similarity to the unknown mental target without
  calibration labels (Table 1: LR R = -0.77, rank 6.63/60, top-rank distance
  2.90 at N = 9234).
- Linear regression outperforms the shuffled (S-LR) and Dummy controls, so the
  signal comes from EEG/stimulus alignment.
- The result holds for EEGNet embeddings (Appendix Table 2) and degrades as the
  number of observations shrinks (Figure 3).
