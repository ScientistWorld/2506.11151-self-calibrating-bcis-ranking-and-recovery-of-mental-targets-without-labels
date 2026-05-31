from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKSPACE))

from data.cursor_data import load_arrays, unique_targets  # noqa: E402


CONTROL_METHODS = {"dummy", "random", "s_lr"}
EXPERIMENT_ALIASES = {
    "unsupervised_ranking_target_sessions": "unsupervised_ranking_synthetic_random",
}


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


def _summarize_method(output: dict, method: str) -> dict[str, float]:
    runs = output["runs"]
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
        raise ValueError(f"No runs found for method {method}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CURSOR ranking outputs.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(WORKSPACE / "outputs/cursor_ranking.json")],
        help="One or more CURSOR output JSON files.",
    )
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring/scores.json"))
    args = parser.parse_args()

    experiments = {}
    for input_name in args.input:
        output = json.loads(Path(input_name).read_text())
        if "label_supervision_count" not in output.get("audit", {}):
            raise ValueError(f"{input_name} is missing audit.label_supervision_count")
        if "pretrained_decoder_count" not in output.get("audit", {}):
            raise ValueError(f"{input_name} is missing audit.pretrained_decoder_count")
        methods = sorted({name for run in output["runs"] for name in run["scores"]})

        results = {method: _summarize_method(output, method) for method in methods}
        _add_control_gaps(results)
        experiments[_experiment_name(output)] = {"results": results}

    scores = {"experiments": experiments}
    score_path = Path(args.scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
