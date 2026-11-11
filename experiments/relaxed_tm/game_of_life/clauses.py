"""
Clause analysis for Game of Life models: distinct clauses, duplicates,
pure clauses and overlap with the reference set (PLAN.md, section 7).

Works on one hard clause bank (``HardTM``). For a bank, ``target`` is the
label its positive clauses vote for: 1 (alive) for a single-output TM; for
bank c of a two-class multi-class TM, c. Negative clauses vote for the other
label.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from cakit.tm import HardTM
from cakit.tm.rules import clause_to_text
from experiments.relaxed_tm.game_of_life.data import POSITIONS, gol_table
from experiments.relaxed_tm.game_of_life.reference import reference_tm


def clause_keys(include: torch.Tensor) -> List[int]:
    """Each clause's include row as one integer (bit k = literal k included)."""
    weights = 1 << torch.arange(include.shape[1])
    return [int(v) for v in (include.long() * weights).sum(dim=1)]


def _active_mask(bank: HardTM) -> torch.Tensor:
    n = bank.n_inputs
    empty = ~bank.include.any(dim=1)
    contradictory = (bank.include[:, :n] & bank.include[:, n:]).any(dim=1)
    return ~empty & ~contradictory


_REFERENCE = None


def _reference_keys() -> Dict[int, set]:
    global _REFERENCE
    if _REFERENCE is None:
        ref = reference_tm()
        keys = clause_keys(ref.include)
        _REFERENCE = {
            +1: {k for k, p in zip(keys, ref.polarity) if p > 0},
            -1: {k for k, p in zip(keys, ref.polarity) if p < 0},
        }
    return _REFERENCE


def analyze_bank(bank: HardTM, target: int = 1) -> Dict:
    """
    Clause statistics for one bank, over the 512-row table.

    - active: non-empty and not contradictory (contradictory clauses never fire)
    - distinct: different active clauses; duplicate_fraction = 1 - distinct / active
    - pure: an active clause that fires only on rows whose label is the one it
      votes for (positive: ``target``, negative: 1 - target)
    - in_reference: distinct active clauses that are in the reference set with
      the same vote direction (only meaningful for target = 1 / single-output)
    Counts are given per team (pos / neg) and in total.
    """
    X, y = gol_table()
    fires = bank.clause_outputs(torch.as_tensor(X), predict=True).numpy().astype(bool)  # (512, m)
    active = _active_mask(bank).numpy()
    sizes = bank.include.sum(dim=1).numpy()
    keys = clause_keys(bank.include)
    pol = bank.polarity.numpy()
    ref = _reference_keys()

    out: Dict = {}
    for team, sign in (("pos", 1.0), ("neg", -1.0)):
        sel = active & (pol == sign)
        votes_for = target if sign > 0 else 1 - target
        idx = np.flatnonzero(sel)
        team_keys = [keys[j] for j in idx]
        distinct = set(team_keys)
        pure = [bool(np.all(y[fires[:, j]] == votes_for)) for j in idx]
        ref_set = ref[+1 if votes_for == 1 else -1]
        out[team] = {
            "n_clauses": int((pol == sign).sum()),
            "active": int(len(idx)),
            "distinct": int(len(distinct)),
            "duplicate_fraction": float(1 - len(distinct) / len(idx)) if len(idx) else 0.0,
            "pure": int(sum(pure)),
            "pure_fraction": float(np.mean(pure)) if pure else 0.0,
            "distinct_in_reference": int(len(distinct & ref_set)),
            "mean_literals": float(sizes[idx].mean()) if len(idx) else 0.0,
        }
    a, b = out["pos"], out["neg"]
    n_active = a["active"] + b["active"]
    n_distinct = a["distinct"] + b["distinct"]
    n = bank.n_inputs
    out["total"] = {
        "n_clauses": int(bank.n_clauses),
        "empty": int((~bank.include.any(dim=1)).sum()),
        "contradictory": int((bank.include[:, :n] & bank.include[:, n:]).any(dim=1).sum()),
        "active": n_active,
        "distinct": n_distinct,
        "duplicate_fraction": float(1 - n_distinct / n_active) if n_active else 0.0,
        "pure": a["pure"] + b["pure"],
        "pure_fraction": float((a["pure"] + b["pure"]) / n_active) if n_active else 0.0,
        "distinct_in_reference": a["distinct_in_reference"] + b["distinct_in_reference"],
        "mean_literals": float(sizes[active].mean()) if active.any() else 0.0,
        "literals_hist": [int(v) for v in np.bincount(sizes[active], minlength=19)] if active.any() else [0] * 19,
        "included_literals": int(sizes[active].sum()),
    }
    return out


def merged_rules(bank: HardTM, feature_names: Sequence[str] = POSITIONS) -> List[str]:
    """Active clauses as text, identical ones merged with a count: ``+ : N ∧ ¬SE (×3)``."""
    active = _active_mask(bank)
    counts: Counter = Counter()
    order: List = []
    for j in range(bank.n_clauses):
        if not bool(active[j]):
            continue
        sign = "+" if float(bank.polarity[j]) > 0 else "−"
        line = f"{sign} : {clause_to_text(bank.include[j], feature_names)}"
        if line not in counts:
            order.append(line)
        counts[line] += 1
    order.sort(key=lambda l: (l[0] != "+", -counts[l], l))
    return [l + (f"  (×{counts[l]})" if counts[l] > 1 else "") for l in order]


def wrong_rows(predict, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> List[int]:
    """Table rows a hard model gets wrong (``predict`` maps (n, 9) -> (n,))."""
    if X is None:
        X, y = gol_table()
    pred = np.asarray(predict(torch.as_tensor(X)))
    return [int(i) for i in np.flatnonzero(pred != y)]


def pack_bank(bank: HardTM) -> Dict:
    """A hard bank as compact JSON: one integer per clause (see ``clause_keys``)."""
    return {"include": clause_keys(bank.include), "polarity": [int(p) for p in bank.polarity.tolist()],
            "bias": float(bank.bias), "n_literals": int(bank.include.shape[1])}


def unpack_bank(d: Dict) -> HardTM:
    keys = torch.tensor(d["include"], dtype=torch.long)
    include = ((keys[:, None] >> torch.arange(d["n_literals"])) & 1).bool()
    return HardTM(include=include, polarity=torch.tensor(d["polarity"], dtype=torch.float), bias=d["bias"])


def analyze_multiclass(banks: Sequence[HardTM]) -> Dict:
    """
    Clause statistics for a two-class model with banks [dead, alive].

    Per bank (``analyze_bank`` with target = the bank's class), and combined:
    every active clause is labelled with the class it votes for (a positive
    clause of bank c votes for c, a negative one for the other class); two
    clauses are duplicates if they are equal and vote for the same class,
    within a bank or across the two banks. ``shared_across`` counts distinct
    clauses that appear, voting for the same class, in both banks (e.g. the
    dead bank's + team and the alive bank's - team).
    """
    X, y = gol_table()
    per_bank = [analyze_bank(b, target=c) for c, b in enumerate(banks)]
    ref = _reference_keys()
    items = []          # (votes_for, key)
    by_bank = [set(), set()]
    sizes, pure = [], 0
    for c, bank in enumerate(banks):
        fires = bank.clause_outputs(torch.as_tensor(X), predict=True).numpy().astype(bool)
        active = _active_mask(bank).numpy()
        keys = clause_keys(bank.include)
        sz = bank.include.sum(dim=1).numpy()
        for j in np.flatnonzero(active):
            votes_for = c if float(bank.polarity[j]) > 0 else 1 - c
            items.append((votes_for, keys[j]))
            by_bank[c].add((votes_for, keys[j]))
            sizes.append(int(sz[j]))
            pure += bool(np.all(y[fires[:, j]] == votes_for))
    distinct = set(items)
    n_active = len(items)
    in_ref = sum(1 for v, k in distinct if k in ref[+1 if v == 1 else -1])
    sizes = np.asarray(sizes, dtype=int)
    total = {
        "n_clauses": int(sum(b.n_clauses for b in banks)),
        "empty": int(sum(p["total"]["empty"] for p in per_bank)),
        "contradictory": int(sum(p["total"]["contradictory"] for p in per_bank)),
        "active": n_active,
        "distinct": len(distinct),
        "duplicate_fraction": float(1 - len(distinct) / n_active) if n_active else 0.0,
        "pure": int(pure),
        "pure_fraction": float(pure / n_active) if n_active else 0.0,
        "distinct_in_reference": int(in_ref),
        "mean_literals": float(sizes.mean()) if n_active else 0.0,
        "literals_hist": [int(v) for v in np.bincount(sizes, minlength=19)] if n_active else [0] * 19,
        "included_literals": int(sizes.sum()),
        "shared_across": len(by_bank[0] & by_bank[1]),
    }
    return {"dead": per_bank[0], "alive": per_bank[1], "total": total}
