"""
A hand-built clause set that implements Game of Life exactly (PLAN.md,
section 3), as a hard TM. Every positive clause fires only on rows that lead
to alive, every negative clause only on rows that lead to dead, and every
row is covered by a clause of the right sign: the "any clause fires" style.

    + exactly 3 live neighbours, any centre        56 clauses, 8 literals
    + centre alive and exactly 2 live neighbours   28 clauses, 9 literals
    - 4 or more live neighbours                    70 clauses, 4 literals
    - at most 1 live neighbour                      8 clauses, 7 literals
    - centre dead and exactly 2 live neighbours    28 clauses, 9 literals

190 clauses in total (84 positive, 106 negative). Not proven minimal.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import torch

from cakit.tm import HardTM
from experiments.relaxed_tm.game_of_life.data import CENTRE, N_INPUTS

NEIGHBOURS = [k for k in range(N_INPUTS) if k != CENTRE]

# (sign, case, list of {bit: value}) ; value 1 = literal x, 0 = literal ¬x
Clause = dict


def _cases() -> List[Tuple[int, str, List[Clause]]]:
    exactly3 = [{k: int(k in S) for k in NEIGHBOURS} for S in combinations(NEIGHBOURS, 3)]
    alive_exactly2 = [{CENTRE: 1, **{k: int(k in S) for k in NEIGHBOURS}} for S in combinations(NEIGHBOURS, 2)]
    four_or_more = [{k: 1 for k in S} for S in combinations(NEIGHBOURS, 4)]
    at_most1 = [{k: 0 for k in S} for S in combinations(NEIGHBOURS, 7)]
    dead_exactly2 = [{CENTRE: 0, **{k: int(k in S) for k in NEIGHBOURS}} for S in combinations(NEIGHBOURS, 2)]
    return [
        (+1, "exactly 3 live neighbours", exactly3),
        (+1, "centre alive, exactly 2 live neighbours", alive_exactly2),
        (-1, "4 or more live neighbours", four_or_more),
        (-1, "at most 1 live neighbour", at_most1),
        (-1, "centre dead, exactly 2 live neighbours", dead_exactly2),
    ]


def _include_row(clause: Clause) -> torch.Tensor:
    row = torch.zeros(2 * N_INPUTS, dtype=torch.bool)
    for k, v in clause.items():
        row[k if v == 1 else N_INPUTS + k] = True
    return row


def reference_tm() -> HardTM:
    """The 190-clause reference set (positive clauses first), threshold 0."""
    rows, pol = [], []
    for sign, _, clauses in _cases():
        for cl in clauses:
            rows.append(_include_row(cl))
            pol.append(float(sign))
    return HardTM(include=torch.stack(rows), polarity=torch.tensor(pol), bias=0.0)


def reference_cases() -> List[Tuple[str, str, int, int]]:
    """(sign, case, clauses, literals each) for printing."""
    return [("+" if s > 0 else "−", name, len(cl), len(cl[0])) for s, name, cl in _cases()]
