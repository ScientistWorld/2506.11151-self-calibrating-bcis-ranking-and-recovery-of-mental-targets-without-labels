"""Reproduction diagnostic on the paper's own simulation protocol (not scored).

Section 6.1 of the paper regenerates, for each of the 17 recorded targets,
synthetic stimuli at every recorded stimulus-target distance (N = 9234) and
ranks 60 hypotheses at uniformly random distances in [0, 46.16] from the
target, the target itself included.  This script runs CURSOR (LR) and the
paper's S-LR / Dummy controls on exactly that protocol at full scale, plus a
no-EEG "stimulus centroid" ranking that shows why the protocol cannot serve
as a benchmark: the stimuli are centred on the answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from data.cursor_data import (  # noqa: E402
    load_arrays,
    paired_distance_distribution,
    random_candidate_set,
    synthetic_training_split,
    unique_targets,
)
from method.cursor import cursor_scores_fast  # noqa: E402


def ranking_metrics(scores: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    target = int(np.argmin(distances))
    better = int(np.sum(scores > scores[target]))
    tied = int(np.sum(scores == scores[target])) - 1
    positions = np.arange(better + 1, better + tied + 2, dtype=float)
    top = np.flatnonzero(scores == scores.max())
    r = float(np.corrcoef(scores, distances)[0, 1]) if np.std(scores) > 0 else 0.0
    return {
        "score_distance_pearson_r": r,
        "target_rank": float(positions.mean()),
        "top_rank_distance": float(distances[top].mean()),
        "target_top1_accuracy": float(np.mean(positions <= 1)),
        "target_top5_accuracy": float(np.mean(positions <= 5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(WORKSPACE / "outputs/paper_protocol_check.json"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-observations", type=int, default=9234)
    args = parser.parse_args()

    arrays = load_arrays()
    targets = unique_targets(arrays["target_faces"])
    recorded = paired_distance_distribution(arrays["target_faces"], arrays["observed_faces"])
    per_run: dict[str, dict[str, list[dict]]] = {}
    for representation in ("eeg_raw", "eeg_net"):
        eeg_all = np.asarray(arrays[representation], dtype=float)
        runs: dict[str, list[dict]] = {m: [] for m in ("lr", "s_lr", "dummy", "no_eeg_centroid")}
        for repeat in range(args.repeats):
            for t, target in enumerate(targets):
                seed = 1000 * repeat + t
                observed, rows = synthetic_training_split(target, recorded, args.n_observations, seed)
                candidates, distances = random_candidate_set(target, float(np.max(recorded)), 60, seed + 7919)
                eeg = eeg_all[rows]
                scores = {
                    "lr": cursor_scores_fast(eeg, observed, candidates, seed=repeat),
                    "s_lr": cursor_scores_fast(
                        eeg[np.random.default_rng(seed + 3001).permutation(len(eeg))],
                        observed, candidates, seed=repeat),
                    "dummy": cursor_scores_fast(eeg, observed, candidates, seed=repeat, estimator="dummy"),
                    "no_eeg_centroid": -np.linalg.norm(candidates - observed.mean(axis=0), axis=1),
                }
                for method, values in scores.items():
                    runs[method].append(ranking_metrics(values, distances))
            print(f"[paper-protocol] {representation} repeat {repeat} done", flush=True)
        per_run[representation] = runs

    summary = {}
    for representation, runs in per_run.items():
        summary[representation] = {}
        for method, rows in runs.items():
            summary[representation][method] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in rows])),
                    "sd": float(np.std([row[metric] for row in rows])),
                }
                for metric in rows[0]
            }
    out = {
        "protocol": "paper Section 6.1: target-centred synthetic stimuli, uniform-distance candidates",
        "n_observations": args.n_observations,
        "n_runs_per_method": len(targets) * args.repeats,
        "summary": summary,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
