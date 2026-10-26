"""Standard-TM baseline wrapper (skipped if tmu is not installed)."""

import numpy as np
import pytest

pytest.importorskip("tmu")

from experiments.relaxed_tm.classification.baselines import StandardTMConfig, run_standard_tm, tmu_seed
from experiments.relaxed_tm.classification.data import generate_noisy_xor
from experiments.relaxed_tm.classification.xor_rules import expected_rule


def test_seed_zero_is_shifted():
    # tmu's random generator hangs when seeded with 0
    assert tmu_seed(0) != 0


def test_standard_tm_learns_clean_xor_and_converts():
    X_tr, y_tr, X_te, y_te = generate_noisy_xor(n_train=1000, n_test=500, label_noise=0.0, seed=0)
    r = run_standard_tm(X_tr, y_tr, X_te, y_te, StandardTMConfig(epochs=10, eval_every=0), seed=0)
    assert r["test"] > 0.95
    # our HardMultiClassTM built from tmu's include actions predicts exactly like tmu
    assert r["hard_matches_tmu"]


def test_seed_reproduces():
    X_tr, y_tr, X_te, y_te = generate_noisy_xor(n_train=500, n_test=200, seed=1)
    cfg = StandardTMConfig(epochs=3, eval_every=1)
    a = run_standard_tm(X_tr, y_tr, X_te, y_te, cfg, seed=3)
    b = run_standard_tm(X_tr, y_tr, X_te, y_te, cfg, seed=3)
    assert a["test_curve"] == b["test_curve"]


def test_expected_rule_mirrors():
    assert expected_rule(1) == {"+ : x1 ∧ ¬x2", "+ : ¬x1 ∧ x2", "− : x1 ∧ x2", "− : ¬x1 ∧ ¬x2"}
    assert expected_rule(0) == {"− : x1 ∧ ¬x2", "− : ¬x1 ∧ x2", "+ : x1 ∧ x2", "+ : ¬x1 ∧ ¬x2"}
