"""
Wolfram elementary CA rule classifications.

The 88 behaviorally unique elementary CA rules (under symmetry transformations)
organized by Wolfram's four behavioral classes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ---- Class metadata ----------------------------------------------------------

CLASS_LABELS: Dict[int, str] = {
    1: "Class I (fixed-point)",
    2: "Class II (periodic)",
    3: "Class III (chaotic)",
    4: "Class IV (complex)",
}

CLASS_SHORT_LABELS: Dict[int, str] = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
}

# ---- Rule membership per class -----------------------------------------------

_CLASS_RULES: Dict[int, List[int]] = {
    1: [0, 8, 32, 40, 128, 136, 160, 168],
    2: [
        1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26,
        27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57,
        58, 62, 72, 73, 74, 76, 77, 78, 94, 104, 108, 130, 132, 134, 138,
        140, 142, 152, 154, 156, 162, 164, 170, 172, 178, 184, 200, 204, 232,
    ],
    3: [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150],
    4: [41, 54, 106, 110],
}

# ---- Derived flat lookup -----------------------------------------------------

WOLFRAM_CLASSES: Dict[int, int] = {
    rule: cls
    for cls, rules in _CLASS_RULES.items()
    for rule in rules
}

UNIQUE_RULES: List[int] = sorted(WOLFRAM_CLASSES.keys())


# ---- Helper functions --------------------------------------------------------

def get_class(rule: int) -> int:
    """Return Wolfram class (1–4) for a given rule number."""
    try:
        return WOLFRAM_CLASSES[rule]
    except KeyError:
        raise ValueError(
            f"Rule {rule} is not among the 88 unique elementary CA rules."
        )


def get_label(rule: int) -> str:
    """Return full label string, e.g. 'Rule 30 · Class III (chaotic)'."""
    cls = get_class(rule)
    return f"Rule {rule} · {CLASS_LABELS[cls]}"


def get_rules_for_class(cls: int) -> List[int]:
    """Return sorted list of rule numbers for a given Wolfram class (1–4)."""
    if cls not in _CLASS_RULES:
        raise ValueError(f"Class must be 1–4, got {cls}.")
    return list(_CLASS_RULES[cls])


def get_cross_class_pairs() -> List[Tuple[int, int]]:
    """
    Return all unique unordered cross-class rule pairs (1,659 total).

    Within-class pairs are excluded. Pair ordering is (lower-class rule,
    higher-class rule) so class(a) < class(b) always holds.
    """
    pairs: List[Tuple[int, int]] = []
    classes = sorted(_CLASS_RULES.keys())
    for i, cls_a in enumerate(classes):
        for cls_b in classes[i + 1:]:
            for rule_a in _CLASS_RULES[cls_a]:
                for rule_b in _CLASS_RULES[cls_b]:
                    pairs.append((rule_a, rule_b))
    return pairs


__all__ = [
    "CLASS_LABELS",
    "CLASS_SHORT_LABELS",
    "UNIQUE_RULES",
    "WOLFRAM_CLASSES",
    "get_class",
    "get_cross_class_pairs",
    "get_label",
    "get_rules_for_class",
]
