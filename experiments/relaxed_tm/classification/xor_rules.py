"""
Checks of learned clauses against the known XOR rule (noisy XOR, E1-E3).

The rule for "y = 1" is: positive clauses x1 ∧ ¬x2 and ¬x1 ∧ x2, negative
clauses x1 ∧ x2 and ¬x1 ∧ ¬x2. A Multi-Class TM's "y = 0" bank learns the
mirror image (same clauses, opposite polarity). Bits x3..x12 are noise.

Three measures, from lenient to strict:

- ``core_present``: all four XOR clauses appear with the right polarity.
- ``n_extra``: non-empty clauses that are not one of the four XOR clauses
  (e.g. a clause on noise bits, or an XOR clause with a noise bit added).
- ``exact``: ``core_present`` and ``n_extra == 0``. Duplicates are allowed.
"""

from __future__ import annotations

from typing import Dict

from cakit.tm import HardTM, format_rules

XOR_POSITIVE = {"x1 ∧ ¬x2", "¬x1 ∧ x2"}
XOR_NEGATIVE = {"x1 ∧ x2", "¬x1 ∧ ¬x2"}


def expected_rule(target: int = 1) -> set:
    """Expected rule lines for the bank voting for ``target`` (1 or 0)."""
    pos, neg = (XOR_POSITIVE, XOR_NEGATIVE) if target == 1 else (XOR_NEGATIVE, XOR_POSITIVE)
    return {f"+ : {c}" for c in pos} | {f"− : {c}" for c in neg}


def check_xor_rule(bank: HardTM, target: int = 1) -> Dict:
    expected = expected_rule(target)
    lines = format_rules(bank)  # empty clauses are skipped (they do not vote)
    present = set(lines)
    n_extra = sum(line not in expected for line in lines)
    core = expected <= present
    return {
        "core_present": core,
        "n_extra": n_extra,
        "exact": core and n_extra == 0,
        "rules": lines,
    }
