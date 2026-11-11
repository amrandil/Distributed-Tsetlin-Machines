"""
Training loop for relaxed Tsetlin Machines.

Fully soft training: loss and gradients use the smooth model; the hard
(rounded) model is only evaluated. Following Granmo's protocol, the model
after the final epoch is reported (no best-epoch picking). Soft and hard
accuracy are logged per epoch for diagnosis only.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from cakit.tm import MultiClassRelaxedTM, RelaxedTM, harden, undecided_fraction

Model = Union[RelaxedTM, MultiClassRelaxedTM]


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.02
    lambda_l1: float = 0.0
    lambda_entropy: float = 0.0
    entropy_start_frac: float = 0.5  # switch the entropy penalty on after this share of epochs
    log_every: int = 1  # epochs between diagnostic evaluations (0 = never)
    # T-like margin (single-output models only): an example whose vote sum is
    # already at least `margin` on the correct side (s - b >= margin for y = 1,
    # s - b <= -margin for y = 0) adds no loss, like the standard TM's target T.
    # None = plain cross-entropy.
    margin: Optional[float] = None
    # "bce": cross-entropy (default). "squared": squared error between the soft
    # output and the label (single-output: sigmoid(logits) vs 0/1; multi-class:
    # softmax probabilities vs one-hot), averaged over the batch, as DiffLogic
    # CA's loss (which is summed).
    loss: str = "bce"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss(model: Model, x: torch.Tensor, y: torch.Tensor, margin: Optional[float] = None,
          kind: str = "bce") -> torch.Tensor:
    if kind == "squared":
        if margin is not None:
            raise NotImplementedError("squared error is not implemented with a margin")
        if isinstance(model, MultiClassRelaxedTM):
            # squared error between the softmax class probabilities and the one-hot label
            probs = torch.softmax(model.logits(x), dim=-1)
            return ((probs - F.one_hot(y, probs.shape[-1]).float()) ** 2).sum(dim=-1).mean()
        return ((torch.sigmoid(model.logits(x)) - y.float()) ** 2).mean()
    if kind != "bce":
        raise ValueError(f"unknown loss {kind!r}")
    if isinstance(model, MultiClassRelaxedTM):
        if margin is not None:
            raise NotImplementedError("margin is only implemented for single-output models")
        return F.cross_entropy(model.logits(x), y)
    logits = model.logits(x)
    if margin is None:
        return F.binary_cross_entropy_with_logits(logits, y.float())
    per_example = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    s = model.vote_sum(x) - model.bias
    satisfied = ((y == 1) & (s >= margin)) | ((y == 0) & (s <= -margin))
    # satisfied examples give no gradient; average over the whole batch, so
    # fewer unsatisfied examples also means a smaller step (as in the TM)
    return (per_example * (~satisfied).float()).mean()


@torch.no_grad()
def soft_accuracy(model: Model, X: torch.Tensor, y: torch.Tensor) -> float:
    if isinstance(model, MultiClassRelaxedTM):
        pred = model.logits(X).argmax(dim=-1)
    else:
        pred = (model.logits(X) >= 0).long()
    return float((pred == y).float().mean())


@torch.no_grad()
def hard_accuracy(model: Model, X: torch.Tensor, y: torch.Tensor) -> float:
    return float((harden(model).predict(X) == y).float().mean())


def train(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    on_epoch_end: Optional[Callable[[int, Model], None]] = None,
) -> Dict[str, List[float]]:
    """
    Train ``model`` in place. Returns a per-epoch history of loss and
    (if ``log_every`` > 0) soft/hard accuracy on training and validation data.
    ``on_epoch_end(epoch, model)``, if given, is called after every epoch
    (e.g. to record include-probabilities); it must not use random numbers.
    """
    Xt = torch.as_tensor(X_train).long()
    yt = torch.as_tensor(y_train).long()
    Xv = torch.as_tensor(X_val).long() if X_val is not None else None
    yv = torch.as_tensor(y_val).long() if y_val is not None else None

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = len(Xt)
    entropy_start = int(config.epochs * config.entropy_start_frac)
    history: Dict[str, List[float]] = {
        "loss": [], "train_soft": [], "train_hard": [], "val_soft": [], "val_hard": [], "undecided": []
    }

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            loss = _loss(model, Xt[idx], yt[idx], config.margin, config.loss)
            if config.lambda_l1:
                loss = loss + config.lambda_l1 * model.l1_penalty()
            if config.lambda_entropy and epoch >= entropy_start:
                loss = loss + config.lambda_entropy * model.entropy_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["loss"].append(total / n)

        last = epoch == config.epochs - 1
        if config.log_every and (epoch % config.log_every == 0 or last):
            model.eval()
            history["train_soft"].append(soft_accuracy(model, Xt, yt))
            history["train_hard"].append(hard_accuracy(model, Xt, yt))
            if Xv is not None:
                history["val_soft"].append(soft_accuracy(model, Xv, yv))
                history["val_hard"].append(hard_accuracy(model, Xv, yv))
            history["undecided"].append(undecided_fraction(model))
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
    return history


def evaluate(model: Model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Soft and hard accuracy plus the soft-hard gap."""
    Xt = torch.as_tensor(X).long()
    yt = torch.as_tensor(y).long()
    soft = soft_accuracy(model, Xt, yt)
    hard = hard_accuracy(model, Xt, yt)
    return {"soft": soft, "hard": hard, "gap": soft - hard}


def summarize(values: List[float]) -> Dict[str, float]:
    """Granmo's reporting: mean ± 95% CI, 5th/95th percentile, min, max (in %)."""
    v = 100.0 * np.asarray(values, dtype=float)
    ci = 1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return {
        "mean": float(v.mean()),
        "ci95": float(ci),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n_runs": int(len(v)),
    }
