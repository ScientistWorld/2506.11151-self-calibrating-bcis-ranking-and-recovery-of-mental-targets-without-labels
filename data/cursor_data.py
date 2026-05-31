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


def paired_distance_distribution(
    target_faces: np.ndarray,
    observed_faces: np.ndarray,
) -> np.ndarray:
    """Distances paired with the released EEG rows."""
    return np.linalg.norm(observed_faces - target_faces, axis=1)


def unique_diagonal_shifts(
    target_faces: np.ndarray,
    observed_faces: np.ndarray,
    decimals: int = 6,
) -> np.ndarray:
    """Approximate upstream's unique target-to-observed shift set."""
    shifts = observed_faces - target_faces
    return np.unique(np.round(shifts, decimals=decimals), axis=0)


def faces_at_distances(
    target: np.ndarray,
    distances: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate latent faces at prescribed Euclidean distances from target."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(len(distances), target.shape[0]))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    return target[None, :] + directions * distances[:, None]


def synthetic_training_split(
    target: np.ndarray,
    distance_distribution: np.ndarray,
    n_observations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style generated train faces plus EEG row indexes.

    The upstream release generates one synthetic face per released EEG row at
    the row's original target distance, then samples rows for a training size.
    Direct radial sampling preserves that distance constraint without the
    expensive numerical optimizer used in the original script.
    """
    synthetic_faces = faces_at_distances(target, distance_distribution, seed)
    row_ids = np.arange(len(distance_distribution))
    selected = subsample_rows(row_ids, n_observations, seed + 17)
    return synthetic_faces[selected], selected


def diagonal_candidate_set(
    target: np.ndarray,
    shifts: np.ndarray,
    n_candidates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style diagonal candidate hypotheses around a target."""
    rng = np.random.default_rng(seed)
    candidates = target[None, :] + shifts
    distances = np.linalg.norm(candidates - target, axis=1)

    if n_candidates < len(candidates):
        target_idx = int(np.argmin(distances))
        remaining = np.array([i for i in range(len(candidates)) if i != target_idx])
        order = remaining[np.argsort(distances[remaining])]
        positions = np.linspace(0, len(order) - 1, max(0, n_candidates - 1))
        chosen = order[np.rint(positions).astype(int)]
        if len(np.unique(chosen)) < len(chosen):
            chosen = rng.choice(remaining, len(chosen), replace=False)
        chosen = np.concatenate([[target_idx], chosen])
        candidates = candidates[chosen]
        distances = distances[chosen]

    perm = rng.permutation(len(candidates))
    return candidates[perm], distances[perm]


def random_candidate_set(
    target: np.ndarray,
    max_distance: float,
    n_candidates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper Table 1 random candidates: uniform distances plus exact target."""
    rng = np.random.default_rng(seed)
    n_random = max(0, n_candidates - 1)
    distances = rng.random(n_random) * max_distance
    candidates = faces_at_distances(target, distances, seed + 31)
    candidates = np.vstack([candidates, target[None, :]])
    true_distances = np.linalg.norm(candidates - target, axis=1)
    perm = rng.permutation(len(candidates))
    return candidates[perm], true_distances[perm]
