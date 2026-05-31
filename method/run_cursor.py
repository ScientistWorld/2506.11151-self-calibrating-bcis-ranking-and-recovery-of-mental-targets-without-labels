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


def write_output(path: Path, args: argparse.Namespace, runs: list[dict], complete: bool) -> None:
    experiment = f"unsupervised_ranking_{args.protocol}"
    if args.eeg == "eeg_net":
        experiment = "unsupervised_ranking_eegnet"
    output = {
        "experiment": experiment,
        "data": str(Path(args.data).resolve()),
        "eeg": args.eeg,
        "seed": args.seed,
        "protocol": args.protocol,
        "methods": args.methods,
        "n_observations": args.n_observations,
        "n_candidates": args.n_candidates,
        "n_splits": args.n_splits,
        "audit": {
            "label_supervision_count": 0,
            "pretrained_decoder_count": 0,
        },
        "complete": complete,
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))


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
    parser.add_argument(
        "--continue-existing",
        action="store_true",
        help="Resume from an incomplete output file with the same run schema.",
    )
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

    out_path = Path(args.output)
    runs = []
    if args.continue_existing and out_path.exists():
        previous = json.loads(out_path.read_text())
        if (
            previous.get("protocol") == args.protocol
            and previous.get("eeg") == args.eeg
            and int(previous.get("seed", args.seed)) == args.seed
            and previous.get("methods") == args.methods
            and int(previous.get("n_observations", -1)) == args.n_observations
            and int(previous.get("n_candidates", -1)) == args.n_candidates
            and int(previous.get("n_splits", -1)) == args.n_splits
            and all("candidates" in run for run in previous.get("runs", []))
        ):
            runs = previous.get("runs", [])
            print(f"[cursor] resuming {out_path} with {len(runs)} completed runs", flush=True)
        else:
            print(f"[cursor] ignoring incompatible existing output at {out_path}", flush=True)

    for run_number, target_index in enumerate(target_order):
        if run_number < len(runs):
            continue
        target = targets[target_index]
        print(
            f"[cursor] target {run_number + 1}/{len(target_order)} "
            f"(index={int(target_index)}, protocol={args.protocol}, n={args.n_observations})",
            flush=True,
        )
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
            print(f"[cursor]   scoring {method}", flush=True)
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
                "candidates": candidates.tolist(),
                "scores": method_scores,
            }
        )
        write_output(out_path, args, runs, complete=False)

    write_output(out_path, args, runs, complete=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
