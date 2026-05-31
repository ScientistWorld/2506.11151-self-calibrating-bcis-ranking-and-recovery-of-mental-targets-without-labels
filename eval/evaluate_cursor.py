from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


WORKSPACE = Path(__file__).resolve().parents[1]


def _rank_of_target(scores: np.ndarray, distances: np.ndarray) -> int:
    target_index = int(np.argmin(distances))
    order = np.argsort(-scores)
    return int(np.flatnonzero(order == target_index)[0] + 1)


def _summarize_method(runs: list[dict], method: str) -> dict[str, float]:
    pearsons = []
    ranks = []
    top_distances = []
    for run in runs:
        if method not in run["scores"]:
            continue
        distances = np.asarray(run["true_distances"], dtype=float)
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
        "label_supervision_count": 0.0,
        "pretrained_decoder_count": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CURSOR ranking outputs.")
    parser.add_argument("--input", default=str(WORKSPACE / "outputs/cursor_ranking.json"))
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring/scores.json"))
    args = parser.parse_args()

    output = json.loads(Path(args.input).read_text())
    runs = output["runs"]
    methods = sorted({name for run in runs for name in run["scores"]})

    results = {method: _summarize_method(runs, method) for method in methods}
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

    scores = {
        "experiments": {
            "unsupervised_ranking_target_sessions": {
                "results": results,
            }
        }
    }
    score_path = Path(args.scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
