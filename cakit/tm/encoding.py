"""
Thermometer encoding of continuous features into bits.

For each feature, k thresholds are placed at evenly spaced quantiles of the
training data (k = 4 -> 20/40/60/80%). Each threshold gives one bit,
"is the value above this threshold?". Fit on the training split only.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class ThermometerEncoder:
    def __init__(self, n_thresholds: int = 4):
        if n_thresholds < 1:
            raise ValueError("n_thresholds must be >= 1")
        self.n_thresholds = n_thresholds
        self.thresholds_: Optional[np.ndarray] = None  # (n_features, k)

    def fit(self, X: np.ndarray) -> "ThermometerEncoder":
        X = np.asarray(X, dtype=float)
        k = self.n_thresholds
        qs = np.arange(1, k + 1) / (k + 1)
        self.thresholds_ = np.quantile(X, qs, axis=0).T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Bits of shape (n_samples, n_features * k), grouped by feature."""
        if self.thresholds_ is None:
            raise RuntimeError("ThermometerEncoder must be fitted first")
        X = np.asarray(X, dtype=float)
        bits = X[:, :, None] > self.thresholds_[None, :, :]
        return bits.reshape(X.shape[0], -1).astype(np.int64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def feature_names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        """Readable bit names, e.g. ``"petal_length>4.60"``."""
        if self.thresholds_ is None:
            raise RuntimeError("ThermometerEncoder must be fitted first")
        n_features = self.thresholds_.shape[0]
        if input_names is None:
            input_names = [f"f{i + 1}" for i in range(n_features)]
        return [
            f"{input_names[i]}>{self.thresholds_[i, t]:.2f}"
            for i in range(n_features)
            for t in range(self.n_thresholds)
        ]
