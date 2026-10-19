"""
Unit tests for the relaxed Tsetlin Machine (cakit.tm).
"""

import numpy as np
import pytest
import torch

from cakit.tm import (
    MultiClassRelaxedTM,
    RelaxedTM,
    ThermometerEncoder,
    format_rules,
    harden,
    make_literals,
    undecided_fraction,
)

BIG = 50.0  # |w| this large gives p within ~1e-22 of 0 or 1

ALL_XOR_INPUTS = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_LABELS = torch.tensor([0, 1, 1, 0])


def set_clauses(model: RelaxedTM, clauses):
    """
    Set w so each clause includes exactly the given literals.
    clauses: list of literal-index lists, one per clause. For n inputs,
    literal k < n is x_{k+1} and literal n + k is ¬x_{k+1}.
    """
    with torch.no_grad():
        model.w.fill_(-BIG)
        for j, lits in enumerate(clauses):
            for k in lits:
                model.w[j, k] = BIG


def xor_model() -> RelaxedTM:
    """Hand-set XOR: + x1¬x2, + ¬x1x2, − x1x2, − ¬x1¬x2."""
    model = RelaxedTM(n_inputs=2, n_clauses=4)
    # literals: 0 = x1, 1 = x2, 2 = ¬x1, 3 = ¬x2
    set_clauses(model, [[0, 3], [2, 1], [0, 1], [2, 3]])
    return model


class TestLiterals:
    def test_literals_are_x_then_negation(self):
        x = torch.tensor([[1, 0, 1]])
        assert make_literals(x).tolist() == [[1, 0, 1, 0, 1, 0]]


class TestSoftClauses:
    def test_log_space_matches_direct_product(self):
        torch.manual_seed(0)
        model = RelaxedTM(n_inputs=5, n_clauses=6, init_mean=0.0, init_std=2.0)
        x = torch.randint(0, 2, (32, 5))
        lit = make_literals(x)
        p = model.include_probs()
        direct = torch.prod(1 - p[None, :, :] * (1 - lit[:, None, :]), dim=-1)
        assert torch.allclose(model.clause_outputs(x), direct, atol=1e-6)

    def test_all_excluded_clause_outputs_one_soft(self):
        model = RelaxedTM(n_inputs=3, n_clauses=2)
        set_clauses(model, [[], []])
        out = model.clause_outputs(torch.randint(0, 2, (10, 3)))
        assert torch.allclose(out, torch.ones_like(out))

    def test_contradictory_clause_outputs_zero(self):
        model = RelaxedTM(n_inputs=3, n_clauses=2)
        set_clauses(model, [[0, 3], [1]])  # clause 0: x1 ∧ ¬x1
        out = model.clause_outputs(torch.randint(0, 2, (10, 3)))
        assert torch.allclose(out[:, 0], torch.zeros(10), atol=1e-6)

    def test_half_included_zero_literal_halves_clause(self):
        model = RelaxedTM(n_inputs=1, n_clauses=2)
        with torch.no_grad():
            model.w.fill_(-BIG)
            model.w[0, 0] = 0.0  # p = 0.5 on x1
        out = model.clause_outputs(torch.tensor([[0]]))
        assert out[0, 0].item() == pytest.approx(0.5)

    def test_polarity_halves(self):
        model = RelaxedTM(n_inputs=2, n_clauses=6)
        assert model.polarity.tolist() == [1, 1, 1, -1, -1, -1]

    def test_odd_clause_count_rejected(self):
        with pytest.raises(ValueError):
            RelaxedTM(n_inputs=2, n_clauses=3)


class TestHandSetXOR:
    def test_soft_model_solves_xor(self):
        model = xor_model()
        y_hat = model(ALL_XOR_INPUTS)
        assert ((y_hat > 0.5).long() == XOR_LABELS).all()

    def test_hard_model_solves_xor(self):
        hard = harden(xor_model())
        assert (hard.predict(ALL_XOR_INPUTS) == XOR_LABELS).all()

    def test_rules_printed(self):
        rules = format_rules(harden(xor_model()))
        assert rules == ["+ : x1 ∧ ¬x2", "+ : ¬x1 ∧ x2", "− : x1 ∧ x2", "− : ¬x1 ∧ ¬x2"]


class TestHardening:
    def test_soft_equals_hard_when_p_is_exact(self):
        torch.manual_seed(1)
        model = RelaxedTM(n_inputs=6, n_clauses=10)
        with torch.no_grad():
            model.w.copy_(torch.where(torch.rand_like(model.w) < 0.2, BIG, -BIG))
        x = torch.randint(0, 2, (64, 6))
        hard = harden(model)
        soft = model.clause_outputs(x)
        # Training convention: empty clauses output 1, like the soft model.
        assert torch.allclose(soft, hard.clause_outputs(x, predict=False), atol=1e-6)

    def test_empty_clause_does_not_vote_at_prediction(self):
        model = RelaxedTM(n_inputs=2, n_clauses=2)
        set_clauses(model, [[], [0]])
        hard = harden(model)
        x = torch.tensor([[0, 0], [1, 1]])
        assert hard.clause_outputs(x)[:, 0].tolist() == [0.0, 0.0]
        assert hard.clause_outputs(x, predict=False)[:, 0].tolist() == [1.0, 1.0]

    def test_stats_count_empty_and_contradictory(self):
        model = RelaxedTM(n_inputs=2, n_clauses=4)
        set_clauses(model, [[], [0, 2], [0, 3], [1]])
        stats = harden(model).stats()
        assert stats["n_empty"] == 1
        assert stats["n_contradictory"] == 1
        assert stats["n_active"] == 2
        assert stats["total_literals"] == 3

    def test_undecided_fraction(self):
        model = RelaxedTM(n_inputs=1, n_clauses=2)
        with torch.no_grad():
            model.w.copy_(torch.tensor([[0.0, BIG], [-BIG, 0.0]]))
        assert undecided_fraction(model) == pytest.approx(0.5)


class TestMultiClass:
    def test_shapes_and_probabilities(self):
        model = MultiClassRelaxedTM(n_inputs=4, n_classes=3, clauses_per_class=6)
        x = torch.randint(0, 2, (8, 4))
        probs = model(x)
        assert probs.shape == (8, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8))

    def test_hard_prediction_is_argmax(self):
        model = MultiClassRelaxedTM(n_inputs=1, n_classes=2, clauses_per_class=2)
        set_clauses(model.banks[0], [[1], []])  # class 0 votes for ¬x1
        set_clauses(model.banks[1], [[0], []])  # class 1 votes for x1
        hard = harden(model)
        assert hard.predict(torch.tensor([[0], [1]])).tolist() == [0, 1]


class TestThermometerEncoder:
    def test_shape_and_monotone_bits(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        enc = ThermometerEncoder(n_thresholds=4).fit(X)
        bits = enc.transform(X)
        assert bits.shape == (100, 12)
        grouped = bits.reshape(100, 3, 4)
        # Thermometer code: once a bit is 0, all higher thresholds are 0.
        assert (np.diff(grouped, axis=-1) <= 0).all()

    def test_thresholds_from_training_data_only(self):
        enc = ThermometerEncoder(n_thresholds=1).fit(np.array([[0.0], [2.0]]))
        assert enc.thresholds_[0, 0] == pytest.approx(1.0)
        assert enc.transform(np.array([[100.0], [-100.0]])).tolist() == [[1], [0]]

    def test_feature_names(self):
        enc = ThermometerEncoder(n_thresholds=1).fit(np.array([[0.0], [2.0]]))
        assert enc.feature_names(["petal"]) == ["petal>1.00"]
