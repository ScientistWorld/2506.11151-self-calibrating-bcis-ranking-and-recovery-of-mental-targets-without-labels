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
