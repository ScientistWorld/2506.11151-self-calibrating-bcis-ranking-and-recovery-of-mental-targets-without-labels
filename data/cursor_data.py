from __future__ import annotations

from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = WORKSPACE / "data" / "downloads" / "all_data_sorted.npz"


def load_arrays(path: str | Path = DEFAULT_DATA) -> dict[str, np.ndarray]:
    """Load the released self-calibrating BCI arrays."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {data_path}. Run scripts/download.sh first."
        )
    loaded = np.load(data_path)
    return {name: loaded[name] for name in loaded.files}


def unique_targets(target_faces: np.ndarray) -> np.ndarray:
    return np.unique(target_faces, axis=0)


def rows_for_target(target_faces: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.all(np.isclose(target_faces, target), axis=1))


def candidate_set_for_target(
    observed_faces: np.ndarray,
    target: np.ndarray,
    row_ids: np.ndarray,
    n_candidates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate hypotheses and true distances, always including target."""
    rng = np.random.default_rng(seed)
    target_observed = observed_faces[row_ids]
    distances = np.linalg.norm(target_observed - target, axis=1)

    n_from_observed = max(0, n_candidates - 1)
    if n_from_observed >= len(target_observed):
        chosen = np.arange(len(target_observed))
    else:
        # Stratify by distance so ranking is not dominated by near duplicates.
        order = np.argsort(distances)
        quantile_positions = np.linspace(0, len(order) - 1, n_from_observed)
        chosen = order[np.rint(quantile_positions).astype(int)]
        if len(np.unique(chosen)) < n_from_observed:
            chosen = rng.choice(len(target_observed), n_from_observed, replace=False)

    candidates = np.vstack([target_observed[chosen], target[None, :]])
    true_distances = np.linalg.norm(candidates - target, axis=1)

    perm = rng.permutation(len(candidates))
    return candidates[perm], true_distances[perm]


def subsample_rows(row_ids: np.ndarray, n_observations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_observations >= len(row_ids):
        return np.array(row_ids, copy=True)
    return rng.choice(row_ids, n_observations, replace=False)
