"""
System sub-package.

Provides abstract base classes and concrete implementations for automata systems.
"""

from .base import AutomataSystem
from .state_system import StateSystem
from .rule_system import RuleSystem

__all__ = [
    'AutomataSystem',
    'StateSystem',
    'RuleSystem',
]
