"""CURSOR solver for evaluator-owned sessions.

Usage (normally invoked by eval/ through a manifest in outputs/solvers/):
    python method/score_sessions.py --sessions DIR --output FILE [--estimator lr]

Reads every session in DIR (see data/cursor_data.py:read_session_dir) and
writes {"scores": {session_id: [score per candidate]}}; higher = more likely
to be the participant's mental target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from data.cursor_data import read_session_dir  # noqa: E402
from method.cursor import cursor_scores_fast, cursor_scores_loo, smooth_scores  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--estimator", choices=["lr", "s_lr", "dummy"], default="lr",
                        help="lr: CURSOR with linear regression; s_lr: paper's shuffled-pairs "
                             "control; dummy: paper's mean-predictor control.")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Use only this fraction of each session's observations.")
    parser.add_argument("--cv", choices=["shufflesplit", "loo"], default="shufflesplit",
                        help="shufflesplit: paper's repeated 90/10 splits; loo: exact leave-one-out.")
    parser.add_argument("--smooth", action="store_true",
                        help="Kernel-average scores over neighbouring candidates.")
    args = parser.parse_args()

    scores = {}
    for session_id, session in read_session_dir(args.sessions):
        eeg = session["eeg"]
        stimuli = session["stimuli"]
        candidates = session["candidates"]
        if args.subsample < 1.0:
            keep = np.random.default_rng(args.seed + 11).choice(
                len(eeg), max(20, int(round(args.subsample * len(eeg)))), replace=False)
            eeg, stimuli = eeg[keep], stimuli[keep]
        estimator = args.estimator
        if estimator == "s_lr":
            eeg = eeg[np.random.default_rng(args.seed + 3001).permutation(len(eeg))]
            estimator = "lr"
        if args.cv == "loo" and estimator == "lr":
            values = cursor_scores_loo(eeg, stimuli, candidates, n_shuffles=args.n_splits, seed=args.seed)
        else:
            values = cursor_scores_fast(eeg, stimuli, candidates, n_splits=args.n_splits,
                                        seed=args.seed, estimator=estimator)
        if args.smooth:
            values = smooth_scores(values, candidates)
        scores[session_id] = [float(v) for v in values]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"scores": scores}))
    print(f"Scored {len(scores)} sessions -> {args.output}")


if __name__ == "__main__":
    main()
