"""
Standard Tsetlin Machine baseline, via the ``tmu`` library (CAIR, Granmo et al.).

This is the only file that imports tmu. It trains a standard Multi-Class TM
and converts the learned clauses into our own ``HardMultiClassTM``, so the
standard TM's rules are printed and checked with the same code as the
relaxed TM's.

Notes on tmu (pinned in requirements.txt to a commit of cair/tmu):

- The PyPI release (0.7.1) and the latest tag (0.8.3) fail under NumPy 2;
  the pinned commit of the main branch fixes this.
- ``number_of_clauses`` in tmu is per class. Granmo's "20 clauses" for noisy
  XOR are 2 classes x 10 clauses (his MultiClassTsetlinMachine divides the
  total over the classes).
- tmu's random generator never returns when seeded with 0, so our seed k is
  passed to tmu as k + 1.
- Tsetlin Automata have 2^8 = 256 states (8 state bits, tmu's default);
  Granmo (2018) used 100 states per action (200 in total).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from cakit.tm import HardMultiClassTM, HardTM

TMU_SOURCE = "git+https://github.com/cair/tmu.git@acff83ce6d72000a33026ff3b6a7eaec8926c1bf"


@dataclass
class StandardTMConfig:
    clauses_per_class: int = 10   # noisy XOR: 2 classes x 10 = Granmo's 20 clauses
    T: int = 15
    s: float = 3.9
    epochs: int = 200
    boost_true_positive_feedback: int = 0   # Granmo's noisy XOR demo: off
    number_of_state_bits_ta: int = 8
    eval_every: int = 10                    # epochs between test-accuracy logs (curve only)


def _import_tmu():
    # tmu logs a CUDA warning on import when pycuda is absent; CPU is used either way.
    logging.getLogger("tmu").setLevel(logging.ERROR)
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    finally:
        logging.disable(previous)
    return TMClassifier


def tmu_seed(seed: int) -> int:
    """tmu hangs when seeded with 0; shift every seed by one."""
    return seed + 1


def to_hard(tm, n_classes: int, n_inputs: int) -> HardMultiClassTM:
    """
    Convert a trained tmu TMClassifier into a ``HardMultiClassTM``.

    tmu stores, per class, clauses 0..m/2-1 as positive and m/2..m-1 as
    negative, and literals as [x1..xn, not x1..not xn]: the same layout as
    ``cakit.tm``.
    """
    m = tm.number_of_clauses
    polarity = torch.cat([torch.ones(m // 2), -torch.ones(m // 2)])
    banks = []
    for c in range(n_classes):
        bank = tm.clause_banks[c]
        include = torch.tensor(
            [[bool(bank.get_ta_action(j, k)) for k in range(2 * n_inputs)] for j in range(m)]
        )
        banks.append(HardTM(include=include, polarity=polarity.clone(), bias=0.0))
    return HardMultiClassTM(banks)


def run_standard_tm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: StandardTMConfig,
    seed: int,
) -> Dict:
    """
    Train one standard TM for ``config.epochs`` epochs and report the final
    epoch (Granmo's protocol). Test accuracy is also logged every
    ``eval_every`` epochs for a learning curve; it is never used to pick a model.
    """
    TMClassifier = _import_tmu()
    Xtr, ytr = X_train.astype(np.uint32), y_train.astype(np.uint32)
    Xte, yte = X_test.astype(np.uint32), y_test.astype(np.uint32)

    tm = TMClassifier(
        config.clauses_per_class,
        config.T,
        config.s,
        boost_true_positive_feedback=config.boost_true_positive_feedback,
        number_of_state_bits_ta=config.number_of_state_bits_ta,
        platform="CPU",
        seed=tmu_seed(seed),
    )

    curve: List[float] = []
    start = time.perf_counter()
    for epoch in range(config.epochs):
        tm.fit(Xtr, ytr)
        last = epoch == config.epochs - 1
        if config.eval_every and ((epoch + 1) % config.eval_every == 0 or last):
            curve.append(float((tm.predict(Xte) == yte).mean()))
    train_time = time.perf_counter() - start

    n_classes = int(max(ytr.max(), yte.max()) + 1)
    hard = to_hard(tm, n_classes, X_train.shape[1])
    return {
        "test": float((tm.predict(Xte) == yte).mean()),
        "train": float((tm.predict(Xtr) == ytr).mean()),
        "test_curve": curve,
        "train_time_s": train_time,
        "hard": hard,
        "hard_matches_tmu": bool((hard.predict(X_test.astype(np.int64)).numpy() == tm.predict(Xte)).all()),
    }
