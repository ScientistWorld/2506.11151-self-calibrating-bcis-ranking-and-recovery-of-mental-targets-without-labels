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
        if np.std(scores) == 0 or np.std(distances) == 0:
            r = 0.0
        else:
            r = float(pearsonr(scores, distances).statistic)
        pearsons.append(r)
        ranks.append(_rank_of_target(scores, distances))
        top_distances.append(float(distances[int(np.argmax(scores))]))

    if not pearsons:
        raise ValueError(f"No runs found for method {method}")

    return {
        "score_distance_pearson_r": float(np.mean(pearsons)),
        "target_rank": float(np.mean(ranks)),
        "top_rank_distance": float(np.mean(top_distances)),
        "label_supervision_count": float(audit["label_supervision_count"]),
        "pretrained_decoder_count": float(audit["pretrained_decoder_count"]),
    }


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
        if "lr" in results:
            controls = [
                results[name]["top_rank_distance"]
                for name in ("s_lr", "dummy")
                if name in results
            ]
            if controls:
                results["lr"]["control_gap_top_rank_distance"] = float(
                    min(controls) - results["lr"]["top_rank_distance"]
                )
        for name in results:
            results[name].setdefault("control_gap_top_rank_distance", 0.0)
        experiments[output.get("experiment", "unsupervised_ranking_target_sessions")] = {
            "results": results
        }

    scores = {"experiments": experiments}
    score_path = Path(args.scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
