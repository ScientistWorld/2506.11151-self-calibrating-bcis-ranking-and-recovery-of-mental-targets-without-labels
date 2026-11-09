> Superseded: this verification predates the hardening iteration. See `HARDENING.md` for the current gym and its checks.

# Verification Report

## Initial Assessment

The workspace started from a claimed `secondary_claims` milestone. The claim is
justified: the implemented ranking method is the paper's CURSOR scoring
procedure, completed artifacts exist for both the raw EEG and EEGNet
representations, and the reusable evaluator scores artifacts without importing
from `method/`.

## Issues Found

- Minor: `scoring/scores.json`, `scores_train.json`, and `scores_test.json`
  still used the older raw experiment key
  `unsupervised_ranking_target_sessions` after `reference.json` and the docs
  were corrected to `unsupervised_ranking_synthetic_random`.
- Cosmetic: an ignored stale output file from an older EEGNet run remained in
  `outputs/` and was not used by any script.
- Residual system artifact: `git status` still shows system-managed root files
  (`README.md`, `manifest.json`, and `trajectory/`) as modified/deleted. These
  are outside the research gym payload and were not edited or restored.

## Fixes Applied

- Renamed the raw experiment key in all three tracked score files to
  `unsupervised_ranking_synthetic_random`, preserving the already-computed
  metric values from the completed artifacts.
- Removed the unused ignored `outputs/cursor_ranking_eegnet.json` artifact.
- Verified `scripts/download.sh` points to a real GitHub raw URL returning HTTP
  200 with the expected 31,141,744 byte dataset.
- Ran `python validate.py --compare` and
  `python validate.py --post-design --compare`; both passed.

## Final Judgment

This workspace is ready to serve as a research gym in both improve and
from-scratch modes. The method-agnostic problem and evaluation documents avoid
algorithm details, `eval/` and `data/` are independent of `method/`, scoring
captures both ranking benefit and self-calibration constraints, and the
train/test split artifacts are present and discriminative.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: scaled ranking reproduction only; the full optimization and
  larger estimator/ablation matrix remain future extensions.
