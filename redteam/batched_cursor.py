"""Numerically equivalent multi-output execution of the released LR procedure.

Each response column is one independent candidate-distance regression. All
normalization, folds, shuffled rows, and mean-of-RMSE-ratios remain unchanged.
The shared design matrix is factorized once per fold rather than 60 times.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from method.cursor import _estimator
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1)
def score_candidates(eeg, observed_faces, candidates, config):
    distances = np.linalg.norm(observed_faces[:, None, :] - candidates[None, :, :], axis=2)
    distances = StandardScaler().fit_transform(distances)
    shuffled_eeg = eeg[np.random.default_rng(config.seed).permutation(len(eeg))]
    folds = ShuffleSplit(n_splits=config.n_splits, test_size=0.1, random_state=config.seed)
    model = _estimator(config.estimator)
    ratios = []
    for train_idx, test_idx in folds.split(eeg):
        aligned = clone(model).fit(eeg[train_idx], distances[train_idx])
        shuffled = clone(model).fit(shuffled_eeg[train_idx], distances[train_idx])
        a_rmse = np.sqrt(mean_squared_error(distances[test_idx], aligned.predict(eeg[test_idx]), multioutput='raw_values'))
        s_rmse = np.sqrt(mean_squared_error(distances[test_idx], shuffled.predict(shuffled_eeg[test_idx]), multioutput='raw_values'))
        ratios.append(np.divide(s_rmse, a_rmse, out=np.full_like(a_rmse, np.inf), where=a_rmse != 0))
    return np.mean(ratios, axis=0)
