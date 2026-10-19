"""
Datasets for the relaxed Tsetlin Machine experiments.

- Clean XOR (E0): 2 input bits, no noise.
- Noisy XOR (E1-E3): Granmo's released files (12 bits, 5,000 + 5,000
  examples, 40% of training labels inverted, clean test labels), or a
  generator with the same structure for other noise levels.
- Iris (E4): the 150 original measurements, thermometer-encoded per
  training split; Granmo's Binary Iris file for the encoding ablation.

Downloaded files are cached in ``experiments/relaxed_tm/data/`` (git-ignored).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
CAIR_RAW = "https://raw.githubusercontent.com/cair/TsetlinMachine/master/"


def _download(filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        urllib.request.urlretrieve(CAIR_RAW + filename, path)
    return path


# ----------------------------------------------------------------------
# XOR
# ----------------------------------------------------------------------
def clean_xor(n_repeats: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """All 4 XOR inputs, repeated: X (4*n_repeats, 2), y (4*n_repeats,)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    y = X[:, 0] ^ X[:, 1]
    return np.tile(X, (n_repeats, 1)), np.tile(y, n_repeats)


def load_noisy_xor() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Granmo's released noisy XOR split: X_train, y_train, X_test, y_test."""
    train = np.loadtxt(_download("NoisyXORTrainingData.txt")).astype(np.int64)
    test = np.loadtxt(_download("NoisyXORTestData.txt")).astype(np.int64)
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]


def generate_noisy_xor(
    n_train: int = 5000,
    n_test: int = 5000,
    n_noise_bits: int = 10,
    label_noise: float = 0.4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same structure as Granmo's dataset: y = x1 XOR x2, plus random bits.
    A fraction ``label_noise`` of training labels is inverted; test is clean.
    """
    rng = np.random.default_rng(seed)
    n_bits = 2 + n_noise_bits
    X_train = rng.integers(0, 2, (n_train, n_bits))
    X_test = rng.integers(0, 2, (n_test, n_bits))
    y_train = X_train[:, 0] ^ X_train[:, 1]
    y_test = X_test[:, 0] ^ X_test[:, 1]
    flip = rng.random(n_train) < label_noise
    y_train = np.where(flip, 1 - y_train, y_train)
    return X_train, y_train, X_test, y_test


# ----------------------------------------------------------------------
# Iris
# ----------------------------------------------------------------------
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def load_iris_raw() -> Tuple[np.ndarray, np.ndarray]:
    """Original Iris measurements X (150, 4) and labels y (150,)."""
    from sklearn.datasets import load_iris

    data = load_iris()
    return data.data, data.target.astype(np.int64)


def load_granmo_binary_iris() -> Tuple[np.ndarray, np.ndarray]:
    """Granmo's Binary Iris file: X (150, 16) bits, y (150,)."""
    data = np.loadtxt(_download("BinaryIrisData.txt")).astype(np.int64)
    return data[:, :16], data[:, 16]


def split_80_20(X: np.ndarray, y: np.ndarray, seed: int):
    """Random 80/20 split, as in Granmo's Iris protocol (redrawn per run)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]
