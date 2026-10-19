"""
Printing hardened clauses as readable rules, e.g. ``+ : x1 ∧ ¬x2``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .harden import HardMultiClassTM, HardTM


def clause_to_text(include_row, feature_names: Sequence[str]) -> str:
    n = len(feature_names)
    parts = []
    for k in range(n):
        if bool(include_row[k]):
            parts.append(feature_names[k])
        if bool(include_row[n + k]):
            parts.append(f"¬{feature_names[k]}")
    return " ∧ ".join(parts) if parts else "(empty)"


def format_rules(
    hard: HardTM,
    feature_names: Optional[Sequence[str]] = None,
    skip_empty: bool = True,
) -> List[str]:
    """
    One line per clause: ``"+ : x1 ∧ ¬x2"``. Feature names default to
    ``x1 .. xn``. Empty clauses (which do not vote at prediction) are
    skipped unless ``skip_empty`` is False.
    """
    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(hard.n_inputs)]
    if len(feature_names) != hard.n_inputs:
        raise ValueError("feature_names must have one name per input bit")

    lines = []
    for j in range(hard.n_clauses):
        row = hard.include[j]
        if skip_empty and not bool(row.any()):
            continue
        sign = "+" if float(hard.polarity[j]) > 0 else "−"
        lines.append(f"{sign} : {clause_to_text(row, feature_names)}")
    return lines


def format_multiclass_rules(
    hard: HardMultiClassTM,
    class_names: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> List[str]:
    if class_names is None:
        class_names = [f"class {c}" for c in range(hard.n_classes)]
    lines = []
    for name, bank in zip(class_names, hard.banks):
        lines.append(f"[{name}]")
        lines.extend("  " + line for line in format_rules(bank, feature_names))
    return lines
