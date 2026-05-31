from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE))

from data.cursor_data import load_arrays, unique_targets  # noqa: E402


SLICE_NAME = "train"
SPLIT_ID = "cursor_target_hash_v1"
CONTROL_METHODS = {"dummy", "random", "s_lr"}
EXPERIMENT_ALIASES = {
    "unsupervised_ranking_target_sessions": "unsupervised_ranking_synthetic_random",
}


def _split_digest(run: dict, fallback_index: int) -> str:
    key = {
        "split_id": SPLIT_ID,
        "run_id": run.get("run_id", fallback_index),
        "target_index": run.get("target_index"),
        "protocol": run.get("protocol"),
        "n_observations": run.get("n_observations"),
        "n_candidates": run.get("n_candidates"),
    }
    payload = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _slice_runs(runs: list[dict]) -> list[dict]:
    if len(runs) < 2:
        return list(runs)
    ordered = sorted(
        enumerate(runs),
        key=lambda item: (_split_digest(item[1], item[0]), item[0]),
    )
    cutoff = (len(ordered) + 1) // 2
    return [run for rank, (_, run) in enumerate(ordered) if rank < cutoff]


def _is_control_method(method: str) -> bool:
    name = method.lower()
    return (
        name in CONTROL_METHODS
        or name.endswith("_control")
        or name.startswith("control_")
        or "shuffle" in name
        or "shuffled" in name
    )


def _rank_of_target(scores: np.ndarray, distances: np.ndarray) -> int:
    target_index = int(np.argmin(distances))
    order = np.argsort(-scores)
    return int(np.flatnonzero(order == target_index)[0] + 1)


def _target_distances(output: dict, run: dict) -> np.ndarray:
    if "candidates" not in run:
        raise ValueError(
            "Method artifact is missing candidate embeddings. Re-run the method "
            "with the current artifact schema so eval can compute ground truth."
        )
    arrays = load_arrays(output["data"])
    targets = unique_targets(arrays["target_faces"])
    target = targets[int(run["target_index"])]
    candidates = np.asarray(run["candidates"], dtype=float)
    return np.linalg.norm(candidates - target, axis=1)


def _summarize_method(output: dict, runs: list[dict], method: str) -> dict[str, float]:
    audit = output.get("audit", {})
    pearsons = []
    ranks = []
    top_distances = []
    for run in runs:
        if method not in run["scores"]:
            continue
        distances = _target_distances(output, run)
        scores = np.asarray(run["scores"][method], dtype=float)
        if len(scores) != len(distances):
            raise ValueError(
                f"Method {method} produced {len(scores)} scores for "
                f"{len(distances)} candidates"
            )
        if np.std(scores) == 0 or np.std(distances) == 0:
            r = 0.0
        else:
            r = float(pearsonr(scores, distances).statistic)
        pearsons.append(r)
        ranks.append(_rank_of_target(scores, distances))
        top_distances.append(float(distances[int(np.argmax(scores))]))

    if not pearsons:
        raise ValueError(f"No {SLICE_NAME} runs found for method {method}")

    rank_array = np.asarray(ranks, dtype=float)
    top_distance_array = np.asarray(top_distances, dtype=float)
    return {
        "score_distance_pearson_r": float(np.mean(pearsons)),
        "target_rank": float(np.mean(ranks)),
        "top_rank_distance": float(np.mean(top_distances)),
        "target_top1_accuracy": float(np.mean(rank_array == 1)),
        "target_top5_accuracy": float(np.mean(rank_array <= 5)),
        "reciprocal_target_rank": float(np.mean(1.0 / rank_array)),
        "near_target_top_rank_rate": float(np.mean(top_distance_array <= 5.0)),
        "label_supervision_count": float(audit["label_supervision_count"]),
        "pretrained_decoder_count": float(audit["pretrained_decoder_count"]),
    }


def _add_control_gaps(results: dict[str, dict[str, float]]) -> None:
    control_distances = [
        method_result["top_rank_distance"]
        for method, method_result in results.items()
        if _is_control_method(method)
    ]
    best_control = min(control_distances) if control_distances else None
    for method, method_result in results.items():
        if best_control is None or _is_control_method(method):
            method_result["control_gap_top_rank_distance"] = 0.0
        else:
            method_result["control_gap_top_rank_distance"] = float(
                best_control - method_result["top_rank_distance"]
            )


def _experiment_name(output: dict) -> str:
    name = output.get("experiment", "unsupervised_ranking_synthetic_random")
    return EXPERIMENT_ALIASES.get(name, name)


def _score_output(input_name: str) -> tuple[str, dict]:
    output = json.loads(Path(input_name).read_text())
    if output.get("complete") is not True:
        raise ValueError(
            f"{input_name} is not marked complete; evaluate only finished method artifacts"
        )
    if "label_supervision_count" not in output.get("audit", {}):
        raise ValueError(f"{input_name} is missing audit.label_supervision_count")
    if "pretrained_decoder_count" not in output.get("audit", {}):
        raise ValueError(f"{input_name} is missing audit.pretrained_decoder_count")

    runs = _slice_runs(output["runs"])
    if not runs:
        raise ValueError(f"{input_name} has no runs in the {SLICE_NAME} slice")

    methods = sorted({name for run in runs for name in run["scores"]})
    results = {method: _summarize_method(output, runs, method) for method in methods}
    _add_control_gaps(results)

    experiment = _experiment_name(output)
    return experiment, {
        "slice": SLICE_NAME,
        "split_id": SPLIT_ID,
        "n_runs": len(runs),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CURSOR train-slice ranking outputs.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(WORKSPACE / "outputs/cursor_ranking.json")],
        help="One or more CURSOR output JSON files.",
    )
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring/scores_train.json"))
    args = parser.parse_args()

    experiments = {}
    for input_name in args.input:
        experiment, score_block = _score_output(input_name)
        experiments[experiment] = score_block

    scores = {"_slice": SLICE_NAME, "experiments": experiments}
    score_path = Path(args.scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
