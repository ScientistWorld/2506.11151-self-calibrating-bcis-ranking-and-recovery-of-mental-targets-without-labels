"""Check the fast multi-candidate CURSOR against the scalar sklearn version.

Also checks that CURSOR-LR is unchanged by the evaluator's per-session EEG
basis (standardise + random rotation), so the hardened sessions do not alter
the reference method's computation.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from method.cursor import CursorConfig, cursor_score, cursor_scores_fast  # noqa: E402


def main() -> None:
    spec = importlib.util.spec_from_file_location("train_eval", WORKSPACE / "eval/train/evaluate_cursor.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    pool = train.load_pool(WORKSPACE / "data/downloads/all_data_sorted.npz")
    sessions = train.make_sessions(pool, train.DEFAULT_DESIGN, 12345, 1, 0)
    session = sessions[0]
    stimuli = session["center"] + session["obs"] @ session["basis"].T
    candidates = session["center"] + session["cand"] @ session["basis"].T
    report = {}
    for representation in ("eeg_raw", "eeg_net"):
        rows_eeg = pool["eeg"][representation][session["rows"]]
        rng = np.random.default_rng(int(session["eeg_seeds"][representation]))
        rotated = train._session_basis_eeg(rows_eeg, rng).astype(float)
        fast = cursor_scores_fast(rotated, stimuli, candidates, n_splits=10, seed=0)
        fast_unrotated = cursor_scores_fast(rows_eeg, stimuli, candidates, n_splits=10, seed=0)
        fast_dummy = cursor_scores_fast(rotated, stimuli, candidates, n_splits=10, seed=0, estimator="dummy")
        check = [0, 7, 19, 33, 46, 59, int(session["target"])]
        scalar = np.array([cursor_score(rotated, stimuli, candidates[j], CursorConfig("lr", 10, 0)) for j in check])
        scalar_dummy = np.array([cursor_score(rotated, stimuli, candidates[j], CursorConfig("dummy", 10, 0))
                                 for j in check])
        report[representation] = {
            "candidates_checked": check,
            "max_abs_diff_fast_vs_sklearn_lr": float(np.max(np.abs(fast[check] - scalar))),
            "max_abs_diff_fast_vs_sklearn_dummy": float(np.max(np.abs(fast_dummy[check] - scalar_dummy))),
            "max_abs_diff_rotated_vs_unrotated_eeg": float(np.max(np.abs(fast - fast_unrotated))),
            "order_identical_rotated_vs_unrotated": bool(np.array_equal(np.argsort(-fast), np.argsort(-fast_unrotated))),
            "score_range": [float(fast.min()), float(fast.max())],
        }
    ok = all(v["max_abs_diff_fast_vs_sklearn_lr"] < 1e-5 and v["max_abs_diff_rotated_vs_unrotated_eeg"] < 1e-5
             and v["max_abs_diff_fast_vs_sklearn_dummy"] < 1e-12 for v in report.values())
    report["passed"] = bool(ok)
    out = WORKSPACE / "redteam/hardening/equivalence.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not ok:
        print("WARNING: fast CURSOR differs from the scalar implementation beyond tolerance")


if __name__ == "__main__":
    main()
