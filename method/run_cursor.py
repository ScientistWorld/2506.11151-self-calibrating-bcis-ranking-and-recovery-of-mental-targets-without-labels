from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKSPACE))

from data.cursor_data import (  # noqa: E402
    candidate_set_for_target,
    diagonal_candidate_set,
    paired_distance_distribution,
    random_candidate_set,
    load_arrays,
    rows_for_target,
    subsample_rows,
    synthetic_training_split,
    unique_diagonal_shifts,
    unique_targets,
)
from method.cursor import CursorConfig, score_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CURSOR ranking experiments.")
    parser.add_argument("--data", default=str(WORKSPACE / "data/downloads/all_data_sorted.npz"))
    parser.add_argument("--output", default=str(WORKSPACE / "outputs/cursor_ranking.json"))
    parser.add_argument("--eeg", choices=["eeg_raw", "eeg_net"], default="eeg_raw")
    parser.add_argument("--methods", nargs="+", default=["lr", "s_lr", "dummy"])
    parser.add_argument(
        "--protocol",
        choices=["synthetic_random", "synthetic_shifts", "target_sessions"],
        default="synthetic_random",
        help="synthetic_random follows the released Table 1 random-candidate protocol.",
    )
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--n-observations", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_arrays(args.data)
    eeg = arrays[args.eeg]
    if eeg is None:
        raise ValueError(f"{args.eeg} is unavailable in {args.data}")

    targets = unique_targets(arrays["target_faces"])
    distance_distribution = paired_distance_distribution(
        arrays["target_faces"],
        arrays["observed_faces"],
    )
    diagonal_shifts = unique_diagonal_shifts(
        arrays["target_faces"],
        arrays["observed_faces"],
    )
    rng = np.random.default_rng(args.seed)
    target_order = rng.permutation(len(targets))[: args.n_targets]

    runs = []
    for run_number, target_index in enumerate(target_order):
        target = targets[target_index]
        if args.protocol == "target_sessions":
            target_rows = rows_for_target(arrays["target_faces"], target)
            obs_rows = subsample_rows(target_rows, args.n_observations, args.seed + run_number)
            candidates, true_distances = candidate_set_for_target(
                arrays["observed_faces"],
                target,
                target_rows,
                args.n_candidates,
                args.seed + 7919 + run_number,
            )
            observed = arrays["observed_faces"][obs_rows]
            observed_eeg = eeg[obs_rows]
        else:
            observed, obs_rows = synthetic_training_split(
                target,
                distance_distribution,
                args.n_observations,
                args.seed + 1543 + run_number,
            )
            if args.protocol == "synthetic_shifts":
                candidates, true_distances = diagonal_candidate_set(
                    target,
                    diagonal_shifts,
                    args.n_candidates,
                    args.seed + 7919 + run_number,
                )
            else:
                candidates, true_distances = random_candidate_set(
                    target,
                    float(np.max(distance_distribution)),
                    args.n_candidates,
                    args.seed + 7919 + run_number,
                )
            observed_eeg = eeg[obs_rows]

        method_scores = {}
        for method in args.methods:
            method_eeg = np.array(observed_eeg, copy=True)
            estimator_name = method
            if method == "s_lr":
                np.random.default_rng(args.seed + 3001 + run_number).shuffle(method_eeg)
                estimator_name = "lr"
            scores = score_candidates(
                method_eeg,
                observed,
                candidates,
                CursorConfig(estimator=estimator_name, n_splits=args.n_splits, seed=args.seed + run_number),
            )
            method_scores[method] = scores.tolist()

        runs.append(
            {
                "run_id": run_number,
                "target_index": int(target_index),
                "protocol": args.protocol,
                "n_observations": int(len(obs_rows)),
                "n_candidates": int(len(candidates)),
                "true_distances": true_distances.tolist(),
                "scores": method_scores,
            }
        )

    output = {
        "experiment": "unsupervised_ranking_target_sessions",
        "data": str(Path(args.data).resolve()),
        "eeg": args.eeg,
        "seed": args.seed,
        "runs": runs,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
