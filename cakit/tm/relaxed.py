"""
Relaxed (differentiable) Tsetlin Machine.

In a standard Tsetlin Machine, a Tsetlin Automaton decides for every
(clause, literal) pair whether the literal is included in the clause.
Here that hard decision is replaced by a free real number ``w``; its
include-probability is ``p = sigmoid(w)``. With ``p`` strictly between
0 and 1 the whole model is smooth, so it can be trained by gradient
descent. After training, :func:`cakit.tm.harden.harden` rounds every
``p`` back to 0/1 and returns an ordinary (hard) Tsetlin Machine.

Forward pass for one input ``x`` of ``n`` bits:

1. Literals: ``l = [x_1 .. x_n, 1 - x_1 .. 1 - x_n]`` (2n values).
2. Soft clause j (soft AND, Payani & Fekri 2019)::

       c_j = prod_k (1 - p_jk * (1 - l_k))

   Each literal contributes the blend of its two hard cases,
   ``(1 - p) * 1 + p * l``: 1 when excluded, its own value when included.
   Because literals are exactly 0 or 1, and ``1 - sigmoid(w) = sigmoid(-w)``,
   this is computed in log space as one matrix product::

       log c_j = sum_k (1 - l_k) * log sigmoid(-w_jk)

3. Vote: the first half of the clauses vote +1, the second half -1;
   ``s = sum(c_pos) - sum(c_neg)``.
4. Output: ``y_hat = sigmoid((s - b) / tau)``.

Inputs must be binary (0/1). The log-space formula is exact only then.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_literals(x: torch.Tensor) -> torch.Tensor:
    """Return ``[x, 1 - x]`` along the last dimension, as float."""
    x = x.float()
    return torch.cat([x, 1.0 - x], dim=-1)


class RelaxedTM(nn.Module):
    """
    Single-output relaxed Tsetlin Machine.

    Args:
        n_inputs: Number of input bits n (the model has 2n literals).
        n_clauses: Number of clauses m (even). Clauses 0..m/2-1 have positive
            polarity, clauses m/2..m-1 negative polarity.
        tau: Output smoothness in ``sigmoid((s - b) / tau)``.
        learn_bias: If True, the threshold b is learned; otherwise b = 0,
            as in a basic Tsetlin Machine.
        init_mean, init_std: Normal initialization of w. A negative mean
            starts most literals mostly excluded (mean -1.5 gives p ~ 0.18).
    """

    def __init__(
        self,
        n_inputs: int,
        n_clauses: int,
        tau: float = 1.0,
        learn_bias: bool = False,
        init_mean: float = -1.5,
        init_std: float = 1.0,
    ):
        super().__init__()
        if n_clauses < 2 or n_clauses % 2 != 0:
            raise ValueError("n_clauses must be an even number >= 2")
        if tau <= 0:
            raise ValueError("tau must be positive")

        self.n_inputs = n_inputs
        self.n_clauses = n_clauses
        self.n_literals = 2 * n_inputs
        self.tau = tau

        self.w = nn.Parameter(
            torch.randn(n_clauses, self.n_literals) * init_std + init_mean
        )
        polarity = torch.ones(n_clauses)
        polarity[n_clauses // 2:] = -1.0
        self.register_buffer("polarity", polarity)

        if learn_bias:
            self.bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("bias", torch.zeros(()))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def include_probs(self) -> torch.Tensor:
        """Include-probabilities p = sigmoid(w), shape (m, 2n)."""
        return torch.sigmoid(self.w)

    def clause_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Soft clause outputs in [0, 1], shape (batch, m)."""
        lit = make_literals(x)
        log_c = (1.0 - lit) @ F.logsigmoid(-self.w).T
        return torch.exp(log_c)

    def vote_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Soft vote sum s, shape (batch,)."""
        return self.clause_outputs(x) @ self.polarity

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(s - b) / tau, shape (batch,). Use with BCE-with-logits."""
        return (self.vote_sum(x) - self.bias) / self.tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft output y_hat in (0, 1), shape (batch,)."""
        return torch.sigmoid(self.logits(x))

    # ------------------------------------------------------------------
    # Regularization terms
    # ------------------------------------------------------------------
    def l1_penalty(self) -> torch.Tensor:
        """Sum of include-probabilities: pushes unneeded literals out."""
        return self.include_probs().sum()

    def entropy_penalty(self) -> torch.Tensor:
        """Sum of binary entropies H(p): pushes every p toward 0 or 1."""
        log_p = F.logsigmoid(self.w)
        log_q = F.logsigmoid(-self.w)
        p = torch.exp(log_p)
        return -(p * log_p + (1.0 - p) * log_q).sum()


class MultiClassRelaxedTM(nn.Module):
    """
    Multi-class relaxed Tsetlin Machine, as Granmo's Multi-Class TM:
    one independent clause bank (a :class:`RelaxedTM`) per class, no shared
    clauses. Training uses a softmax over the class vote sums divided by
    tau (cross-entropy); prediction is the class with the largest vote sum.

    Args:
        n_inputs: Number of input bits.
        n_classes: Number of classes.
        clauses_per_class: Clauses in each bank (even).
        tau, learn_bias, init_mean, init_std: As in :class:`RelaxedTM`;
            with ``learn_bias`` each class gets its own threshold.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int,
        clauses_per_class: int,
        tau: float = 1.0,
        learn_bias: bool = False,
        init_mean: float = -1.5,
        init_std: float = 1.0,
    ):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.clauses_per_class = clauses_per_class
        self.tau = tau
        self.banks = nn.ModuleList(
            [
                RelaxedTM(
                    n_inputs,
                    clauses_per_class,
                    tau=tau,
                    learn_bias=learn_bias,
                    init_mean=init_mean,
                    init_std=init_std,
                )
                for _ in range(n_classes)
            ]
        )

    def vote_sums(self, x: torch.Tensor) -> torch.Tensor:
        """Soft vote sums minus thresholds, shape (batch, n_classes)."""
        return torch.stack([bank.vote_sum(x) - bank.bias for bank in self.banks], dim=-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits (s_c - b_c) / tau, shape (batch, n_classes)."""
        return self.vote_sums(x) / self.tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft class probabilities, shape (batch, n_classes)."""
        return torch.softmax(self.logits(x), dim=-1)

    def l1_penalty(self) -> torch.Tensor:
        return sum(bank.l1_penalty() for bank in self.banks)

    def entropy_penalty(self) -> torch.Tensor:
        return sum(bank.entropy_penalty() for bank in self.banks)

    def include_probs(self) -> List[torch.Tensor]:
        return [bank.include_probs() for bank in self.banks]
