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
    """Compute Algorithm 1's held-out shuffled/aligned RMSE ratio for one h."""
    distances = np.linalg.norm(observed_faces - hypothesis, axis=1)
    model = _estimator(config.estimator)
    rng = np.random.default_rng(config.seed)
    distances = StandardScaler().fit_transform(distances.reshape(-1, 1)).ravel()
    splitter = ShuffleSplit(n_splits=config.n_splits, test_size=0.1, random_state=config.seed)
    fold_ratios: list[float] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(eeg)):
        aligned = clone(model)
        aligned.fit(eeg[train_idx], distances[train_idx])
        aligned_pred = aligned.predict(eeg[test_idx])
        aligned_rmse = float(np.sqrt(mean_squared_error(distances[test_idx], aligned_pred)))

        rng_fold = np.random.default_rng(config.seed + 1009 * (fold_id + 1))
        shuffled_eeg = np.array(eeg, copy=True)
        shuffled_eeg = shuffled_eeg[rng_fold.permutation(len(shuffled_eeg))]

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
