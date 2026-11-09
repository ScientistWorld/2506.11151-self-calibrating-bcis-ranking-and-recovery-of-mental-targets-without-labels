from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


EstimatorName = Literal[
    "lr",
    "s_lr",
    "dummy",
    "svr",
    "mlp",
    "best_mlp",
    "best_svr",
]


@dataclass(frozen=True)
class CursorConfig:
    estimator: EstimatorName = "lr"
    n_splits: int = 10
    seed: int = 0


def _estimator(name: EstimatorName):
    if name in {"lr", "s_lr"}:
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR())
    if name == "best_svr":
        return make_pipeline(StandardScaler(), SVR(C=0.01, gamma="scale", kernel="linear"))
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(100, 50, 25), random_state=0, max_iter=300),
        )
    if name == "best_mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                activation="identity",
                alpha=0.1,
                hidden_layer_sizes=(100, 100),
                learning_rate="adaptive",
                random_state=0,
                max_iter=300,
            ),
        )
    raise ValueError(f"Unknown estimator: {name}")


def cursor_score(
    eeg: np.ndarray,
    observed_faces: np.ndarray,
    hypothesis: np.ndarray,
    config: CursorConfig,
) -> float:
    """Compute the held-out shuffled/aligned RMSE ratio for one h."""
    distances = np.linalg.norm(observed_faces - hypothesis, axis=1)
    model = _estimator(config.estimator)
    distances = StandardScaler().fit_transform(distances.reshape(-1, 1)).ravel()
    splitter = ShuffleSplit(n_splits=config.n_splits, test_size=0.1, random_state=config.seed)
    shuffled_eeg = np.array(eeg, copy=True)
    shuffled_eeg = shuffled_eeg[np.random.default_rng(config.seed).permutation(len(shuffled_eeg))]
    fold_ratios: list[float] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(eeg)):
        aligned = clone(model)
        aligned.fit(eeg[train_idx], distances[train_idx])
        aligned_pred = aligned.predict(eeg[test_idx])
        aligned_rmse = float(np.sqrt(mean_squared_error(distances[test_idx], aligned_pred)))

        shuffled = clone(model)
        shuffled.fit(shuffled_eeg[train_idx], distances[train_idx])
        shuffled_pred = shuffled.predict(shuffled_eeg[test_idx])
        shuffled_rmse = float(np.sqrt(mean_squared_error(distances[test_idx], shuffled_pred)))
        if aligned_rmse == 0:
            fold_ratios.append(float("inf"))
        else:
            fold_ratios.append(shuffled_rmse / aligned_rmse)

    return float(np.mean(fold_ratios))


def score_candidates(
    eeg: np.ndarray,
    observed_faces: np.ndarray,
    candidates: np.ndarray,
    config: CursorConfig,
) -> np.ndarray:
    scores = [
        cursor_score(eeg, observed_faces, candidate, config)
        for candidate in candidates
    ]
    return np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Fast multi-candidate execution of the same procedure.
#
# Every candidate's distance regression shares the EEG design matrix, the
# ShuffleSplit folds and the shuffled-row permutation, so the per-fold normal
# equations are formed once and solved for all 60 candidate columns at once.
# Ordinary least squares with an intercept is invariant to the per-fold EEG
# StandardScaler, so the predictions equal those of
# make_pipeline(StandardScaler(), LinearRegression()) fitted per candidate
# (checked against ``cursor_score`` in redteam/hardening/equivalence.py).
# ---------------------------------------------------------------------------


def _psd_solve(gram: np.ndarray, rhs: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Minimum-norm least-squares solution from normal equations."""
    eigval, eigvec = np.linalg.eigh(gram)
    keep = eigval > rcond * max(float(eigval[-1]), 1e-300)
    inv = np.zeros_like(eigval)
    inv[keep] = 1.0 / eigval[keep]
    return eigvec @ (inv[:, None] * (eigvec.T @ rhs))


def candidate_distance_targets(observed_faces: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Standardized hypothetical distances d_i^h = ||h - z_i||, one column per h."""
    from scipy.spatial.distance import cdist

    distances = cdist(np.asarray(observed_faces, dtype=float), np.asarray(candidates, dtype=float))
    mean = distances.mean(axis=0)
    scale = distances.std(axis=0)
    scale[scale == 0] = 1.0
    return (distances - mean) / scale


def cursor_scores_fast(
    eeg: np.ndarray,
    observed_faces: np.ndarray,
    candidates: np.ndarray,
    n_splits: int = 10,
    seed: int = 0,
    estimator: str = "lr",
    test_size: float = 0.1,
) -> np.ndarray:
    """CURSOR S(h) for every candidate: mean over folds of shuffled/aligned RMSE."""
    targets = candidate_distance_targets(observed_faces, candidates)
    eeg = np.asarray(eeg, dtype=float)
    n = len(eeg)
    center = eeg.mean(axis=0)
    scale = eeg.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.hstack([np.ones((n, 1)), (eeg - center) / scale])
    shuffled = design[np.random.default_rng(seed).permutation(n)]
    gram = design.T @ design  # identical for the row-permuted design
    cross = {"aligned": design.T @ targets, "shuffled": shuffled.T @ targets}
    total = targets.sum(axis=0)
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    ratios = []
    for train_idx, test_idx in splitter.split(design):
        y_test = targets[test_idx]
        rmse = {}
        for kind, x in (("aligned", design), ("shuffled", shuffled)):
            x_test = x[test_idx]
            if estimator == "dummy":
                pred = np.broadcast_to((total - y_test.sum(axis=0)) / len(train_idx), y_test.shape)
            elif estimator == "lr":
                beta = _psd_solve(gram - x_test.T @ x_test, cross[kind] - x_test.T @ y_test)
                pred = x_test @ beta
            else:
                raise ValueError(f"Unsupported fast estimator: {estimator}")
            rmse[kind] = np.sqrt(np.mean((y_test - pred) ** 2, axis=0))
        ratios.append(
            np.divide(rmse["shuffled"], rmse["aligned"],
                      out=np.full_like(rmse["aligned"], np.inf), where=rmse["aligned"] != 0)
        )
    return np.mean(ratios, axis=0)


def smooth_scores(scores: np.ndarray, candidates: np.ndarray, neighbor: int = 3) -> np.ndarray:
    """Gaussian-kernel average of scores over nearby candidate hypotheses."""
    from scipy.spatial.distance import cdist

    pairwise = cdist(np.asarray(candidates, dtype=float), np.asarray(candidates, dtype=float))
    bandwidth = float(np.median(np.sort(pairwise, axis=1)[:, neighbor]))
    if bandwidth <= 0:
        return np.asarray(scores, dtype=float)
    kernel = np.exp(-(pairwise**2) / (2 * bandwidth**2))
    return kernel @ np.asarray(scores, dtype=float) / kernel.sum(axis=1)
