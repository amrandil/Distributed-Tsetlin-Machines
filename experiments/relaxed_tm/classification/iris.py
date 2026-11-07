"""
Iris splits for E4, shared by the standard-TM and relaxed-TM runs so that
seed k gives the same flowers to every model.

Two encodings, 16 bits each:
  binary       Granmo's Binary Iris file: each measurement binned and written
               as a 4-bit binary number (most significant bit first)
  thermometer  the original measurements, 4 thresholds per measurement at the
               20/40/60/80% quantiles of the training flowers only

Split for seed k: random 80/20 (120 training, 30 test flowers), as in Granmo's
protocol. With ``validation=True``, 24 of the 120 training flowers are held out
as validation data and the model trains on the other 96; the 30 test flowers
are dropped. Used only for tuning.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from experiments.relaxed_tm.classification.data import (
    IRIS_FEATURES,
    load_granmo_binary_iris,
    load_iris_raw,
    split_80_20,
)
from cakit.tm import ThermometerEncoder

ENCODINGS = ("binary", "thermometer")
N_VALIDATION = 24
BINARY_NAMES = [f"{f}.bit{b}" for f in IRIS_FEATURES for b in range(1, 5)]  # bit1 = most significant


def iris_split(encoding: str, seed: int, validation: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Returns X_fit, y_fit, X_eval, y_eval, feature_names. X_eval is the test
    split, or the validation flowers if ``validation``.
    """
    if encoding == "binary":
        X, y = load_granmo_binary_iris()
    elif encoding == "thermometer":
        X, y = load_iris_raw()
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    X_fit, y_fit, X_eval, y_eval = split_80_20(X, y, seed)
    if validation:
        perm = np.random.default_rng([seed, 1]).permutation(len(X_fit))
        val, fit = perm[:N_VALIDATION], perm[N_VALIDATION:]
        X_fit, y_fit, X_eval, y_eval = X_fit[fit], y_fit[fit], X_fit[val], y_fit[val]
    if encoding == "thermometer":
        enc = ThermometerEncoder(4).fit(X_fit)
        X_fit, X_eval = enc.transform(X_fit), enc.transform(X_eval)
        names: List[str] = enc.feature_names(IRIS_FEATURES)
    else:
        names = list(BINARY_NAMES)
    return X_fit.astype(np.int64), y_fit.astype(np.int64), X_eval.astype(np.int64), y_eval.astype(np.int64), names
