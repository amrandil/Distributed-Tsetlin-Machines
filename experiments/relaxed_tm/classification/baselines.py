"""
Standard Tsetlin Machine baseline, via the ``tmu`` library (CAIR, Granmo et al.).

It can also run Granmo's original 2018 code (granmo2018/), for comparison.
This is the only file that imports tmu or the 2018 code. It trains a standard Multi-Class TM
and converts the learned clauses into our own ``HardMultiClassTM``, so the
standard TM's rules are printed and checked with the same code as the
relaxed TM's.

Notes on tmu (installed with install_tmu.sh, pinned to a commit of cair/tmu):

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


# ----------------------------------------------------------------------
# Granmo's original 2018 code (granmo2018/MultiClassTsetlinMachine.pyx)
# ----------------------------------------------------------------------
@dataclass
class Granmo2018Config:
    clauses_total: int = 20      # his code splits these over the classes: 2 x 10
    T: int = 15
    s: float = 3.9
    states: int = 100            # states per action (200 in total)
    epochs: int = 200
    eval_every: int = 10         # epochs between test-accuracy logs (curve only)


def granmo2018_to_hard(tm, n_classes: int, clauses_total: int, n_inputs: int, states: int) -> HardMultiClassTM:
    """
    Convert a trained 2018 MultiClassTsetlinMachine into a ``HardMultiClassTM``.

    Layout in his code: class i owns clauses i*m .. i*m + m - 1 (m = clauses per
    class); within a class, even positions are positive and odd ones negative.
    A literal is included iff its automaton's state is above ``states``.
    """
    m = clauses_total // n_classes
    banks = []
    for c in range(n_classes):
        include = torch.zeros(m, 2 * n_inputs, dtype=torch.bool)
        for j in range(m):
            clause = c * m + j
            for k in range(n_inputs):
                include[j, k] = tm.get_state(clause, k, 0) > states
                include[j, n_inputs + k] = tm.get_state(clause, k, 1) > states
        polarity = torch.tensor([1.0 if j % 2 == 0 else -1.0 for j in range(m)])
        banks.append(HardTM(include=include, polarity=polarity, bias=0.0))
    return HardMultiClassTM(banks)


def run_granmo2018_tm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Granmo2018Config,
    seed: int,
) -> Dict:
    """
    Same protocol and return format as ``run_standard_tm``, using Granmo's
    original code. Seeds both random-number generators his code uses: NumPy
    (initial automaton states, example order) and C rand() (all feedback).

    His fit() is called one epoch at a time so the test accuracy can be logged.
    Each call shuffles a fresh 0..n-1 index instead of re-shuffling the previous
    order; both give a uniformly random order every epoch.
    """
    from experiments.relaxed_tm.classification.granmo2018 import MultiClassTsetlinMachine as G

    Xtr, ytr = np.ascontiguousarray(X_train, dtype=np.int32), np.ascontiguousarray(y_train, dtype=np.int32)
    Xte, yte = np.ascontiguousarray(X_test, dtype=np.int32), np.ascontiguousarray(y_test, dtype=np.int32)
    n_classes = int(max(ytr.max(), yte.max()) + 1)

    np.random.seed(seed)
    G.seed(seed + 1)  # glibc treats srand(0) like srand(1); shift as for tmu
    tm = G.MultiClassTsetlinMachine(
        n_classes, config.clauses_total, Xtr.shape[1], config.states, config.s, config.T
    )

    curve: List[float] = []
    start = time.perf_counter()
    for epoch in range(config.epochs):
        tm.fit(Xtr, ytr, len(ytr), epochs=1)
        last = epoch == config.epochs - 1
        if config.eval_every and ((epoch + 1) % config.eval_every == 0 or last):
            curve.append(float(tm.evaluate(Xte, yte, len(yte))))
    train_time = time.perf_counter() - start

    hard = granmo2018_to_hard(tm, n_classes, config.clauses_total, Xtr.shape[1], config.states)
    own_pred = np.array([tm.predict(np.ascontiguousarray(x)) for x in Xte])
    return {
        "test": float(tm.evaluate(Xte, yte, len(yte))),
        "train": float(tm.evaluate(Xtr, ytr, len(ytr))),
        "test_curve": curve,
        "train_time_s": train_time,
        "hard": hard,
        "hard_matches_original": bool((hard.predict(X_test.astype(np.int64)).numpy() == own_pred).all()),
    }
