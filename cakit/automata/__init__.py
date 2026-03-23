"""
Automata sub-package.

Provides abstract base classes and concrete implementations of automata.
"""

from .base import Automaton
from .learning import LearningAutomaton
from .tsetlin import TsetlinAutomaton
from .wolfram import WolframAutomaton

__all__ = [
    'Automaton',
    'LearningAutomaton',
    'TsetlinAutomaton',
    'WolframAutomaton',
]
