"""
Tsetlin Machine sub-package: relaxed (differentiable) Tsetlin Machines.

- :class:`RelaxedTM`: single-output relaxed TM (soft AND clauses, signed vote).
- :class:`MultiClassRelaxedTM`: one clause bank per class, argmax prediction.
- :func:`harden`: round a relaxed model into an ordinary hard TM.
- :func:`format_rules`: print hard clauses as readable rules.
- :class:`ThermometerEncoder`: continuous features to bits.
"""

from .encoding import ThermometerEncoder
from .harden import HardMultiClassTM, HardTM, harden, undecided_fraction
from .relaxed import MultiClassRelaxedTM, RelaxedTM, make_literals
from .rules import format_multiclass_rules, format_rules

__all__ = [
    "RelaxedTM",
    "MultiClassRelaxedTM",
    "make_literals",
    "HardTM",
    "HardMultiClassTM",
    "harden",
    "undecided_fraction",
    "format_rules",
    "format_multiclass_rules",
    "ThermometerEncoder",
]
