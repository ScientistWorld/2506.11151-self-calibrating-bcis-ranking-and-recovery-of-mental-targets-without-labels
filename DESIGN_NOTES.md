# Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The test script points at `eval/test/`, writes `scoring/scores_test.json`, and uses `scoring/scores_train.json` only to enforce the visible method keys.
- Checked train/test Python evaluators for cross-import leakage. They duplicate the split logic and import only neutral workspace/data dependencies, not each other and not `method/`.
- Checked that split evaluators read method output artifacts rather than regenerating results or importing the paper method.
- Checked `scoring/reference.json` for paper-agreement metric names and formulas. The active metrics measure ranking quality or self-calibration constraints, not deviation from published values.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.
- Ran static validation only: `bash -n scripts/evaluate_train.sh`, `bash -n scripts/evaluate_test.sh`, and `python validate.py --reference-only`.

## Changed

- Added four orthogonal ranking metrics to both train and test evaluators:
  `target_top1_accuracy`, `target_top5_accuracy`, `reciprocal_target_rank`, and
  `near_target_top_rank_rate`. These make it harder to optimize only mean
  top-ranked distance or only average target rank while missing exact retrieval
  behavior.
- Added `complete: true` enforcement in both split evaluators so partial method
  artifacts cannot be scored as if they were finished runs.
- Updated `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` defaults to
  evaluate both benchmark experiments:
  `outputs/cursor_ranking.json` and `outputs/cursor_ranking_eegnet_n500.json`.
  The first pass otherwise could silently omit the EEGNet slice unless extra
  CLI arguments were supplied.
- Added the required `description` field to every metric in
  `scoring/reference.json` and registered the new ranking metrics in both
  experiments with modest coefficients.
- After the first real split run, changed both experiments' primary metric from
  `control_gap_top_rank_distance` to `target_top5_accuracy`. The control-gap
  metric remains scored, but it was too sensitive to which broken-control runs
  landed in the eight-run held-out slice to serve as the split-ranking primary.

## Runtime Concerns For Next Iteration

- The split is a deterministic target/run hash split over the runs present in
  the artifact. With the current 17-run artifacts it should be usable, but if a
  future run uses very few targets the train/test scores will be noisy and may
  become uninformative.
- The new default split scripts assume both default artifacts exist. The runner
  should ensure the EEGNet artifact is generated before invoking split
  evaluation, or pass explicit artifact paths for the intended experiments.
- The supervision constraint metrics are audit fields read from the method
  artifact. They are useful for honest benchmark submissions but are not a
  cryptographic guard against a malicious artifact writer.

## Real Split Run

- Job `ef043e1b-b2e` ran `scripts/evaluate_train.sh` and
  `scripts/evaluate_test.sh` on `outputs/cursor_ranking.json` and
  `outputs/cursor_ranking_eegnet_n500.json` with `gpus: 0`; exit code was 0.
- Generated `scoring/scores_train.json` and `scoring/scores_test.json`.
  Both files cover both experiments, the same three method keys
  (`dummy`, `lr`, `s_lr`), and ten metrics per method.
- `python validate.py` passed, including train/test method-key parity,
  split-value differentiation, and primary metric discrimination on train.
- The split is not byte-identical: the proposed `lr` method's primary
  `target_top5_accuracy` changed from 0.67 train to 0.50 test on
  `unsupervised_ranking_synthetic_random`, and from 0.67 train to 0.88 test on
  `unsupervised_ranking_eegnet`.
- The control-gap diagnostic also differs substantially across slices: 13.09
  train to 8.83 test on `unsupervised_ranking_synthetic_random`, and 20.95 train
  to 4.87 test on `unsupervised_ranking_eegnet`.
- The proposed method remains stronger than controls on both slices, so the gym
  has a meaningful baseline to improve. The main caveat is sample size: each
  experiment has only 9 train runs and 8 test runs, so top-k rates move in
  coarse increments.
