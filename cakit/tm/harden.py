"""
Hardening: turning a relaxed Tsetlin Machine back into hard logic.

Literal k is included in clause j iff p_jk > 0.5, i.e. w_jk > 0. A hard
clause is the AND of its included literals.

Empty clauses (no literal included) follow Granmo's Tsetlin Machine code:
they output 1 during training (so they keep receiving feedback) and 0 at
prediction (so they do not vote). Hard models built here are used for
prediction, so ``predict=True`` is the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .relaxed import MultiClassRelaxedTM, RelaxedTM, make_literals

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


@dataclass
class HardTM:
    """
    A hard (ordinary) single-output Tsetlin Machine.

    Attributes:
        include: Bool tensor (m, 2n); include[j, k] = literal k is in clause j.
        polarity: Tensor (m,) of +1 / -1.
        bias: Threshold b; prediction is 1 iff vote sum >= b.
    """

    include: torch.Tensor
    polarity: torch.Tensor
    bias: float = 0.0

    @property
    def n_clauses(self) -> int:
        return self.include.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.include.shape[1] // 2

    def clause_outputs(self, x: ArrayLike, predict: bool = True) -> torch.Tensor:
        """Hard clause outputs (0/1 float), shape (batch, m)."""
        lit = make_literals(_as_tensor(x))
        # A clause fails iff an included literal is 0.
        violations = (1.0 - lit) @ self.include.float().T
        out = (violations == 0).float()
        if predict:
            empty = ~self.include.any(dim=1)
            out[:, empty] = 0.0
        return out

    def vote_sum(self, x: ArrayLike, predict: bool = True) -> torch.Tensor:
        return self.clause_outputs(x, predict) @ self.polarity.float()

    def predict(self, x: ArrayLike) -> torch.Tensor:
        """Hard prediction (0/1 long), shape (batch,)."""
        return (self.vote_sum(x) >= self.bias).long()

    def stats(self) -> Dict[str, float]:
        """Interpretability counts for this clause bank."""
        n = self.n_inputs
        sizes = self.include.sum(dim=1)
        contradictory = (self.include[:, :n] & self.include[:, n:]).any(dim=1)
        empty = sizes == 0
        active = ~empty & ~contradictory
        return {
            "n_clauses": int(self.n_clauses),
            "n_empty": int(empty.sum()),
            "n_contradictory": int(contradictory.sum()),
            "n_active": int(active.sum()),
            "total_literals": int(sizes[active].sum()),
            "mean_clause_length": float(sizes[active].float().mean()) if active.any() else 0.0,
        }


@dataclass
class HardMultiClassTM:
    """A hard Multi-Class Tsetlin Machine: one :class:`HardTM` bank per class."""

    banks: List[HardTM]

    @property
    def n_classes(self) -> int:
        return len(self.banks)

    def vote_sums(self, x: ArrayLike) -> torch.Tensor:
        """Hard vote sums minus thresholds, shape (batch, n_classes)."""
        return torch.stack([b.vote_sum(x) - b.bias for b in self.banks], dim=-1)

    def predict(self, x: ArrayLike) -> torch.Tensor:
        """Class with the largest vote sum (ties: lowest index)."""
        return self.vote_sums(x).argmax(dim=-1)

    def stats(self) -> Dict[str, float]:
        per_bank = [b.stats() for b in self.banks]
        keys = ["n_clauses", "n_empty", "n_contradictory", "n_active", "total_literals"]
        total = {k: int(sum(s[k] for s in per_bank)) for k in keys}
        total["mean_clause_length"] = (
            total["total_literals"] / total["n_active"] if total["n_active"] else 0.0
        )
        return total


def harden_bank(model: RelaxedTM) -> HardTM:
    with torch.no_grad():
        return HardTM(
            include=(model.w > 0).clone(),
            polarity=model.polarity.clone(),
            bias=float(model.bias),
        )


def harden(model: Union[RelaxedTM, MultiClassRelaxedTM]):
    """Round a relaxed model into a hard one (p > 0.5 -> include)."""
    if isinstance(model, MultiClassRelaxedTM):
        return HardMultiClassTM([harden_bank(b) for b in model.banks])
    if isinstance(model, RelaxedTM):
        return harden_bank(model)
    raise TypeError(f"Cannot harden object of type {type(model).__name__}")


def undecided_fraction(model: Union[RelaxedTM, MultiClassRelaxedTM], low: float = 0.1, high: float = 0.9) -> float:
    """Share of include-probabilities still undecided (low < p < high)."""
    with torch.no_grad():
        probs = model.include_probs()
        if isinstance(probs, list):
            probs = torch.cat([p.flatten() for p in probs])
        return float(((probs > low) & (probs < high)).float().mean())
