"""
StateSystem implementation.

A system where each cell's automaton output is interpreted directly as the next cell state.
"""

from typing import Any
import numpy as np

from .base import AutomataSystem


class StateSystem(AutomataSystem):
    """
    State-based automata system.
    
    In a StateSystem, the automaton's output is interpreted directly as the next cell
    state. The neighborhood is not used in the interpretation - only in the automaton's
    get_output() call.
    
    Used as a pure CA when feedback_fn=None, or as a binary-output CLA when feedback_fn
    is provided and the automaton is a LearningAutomaton.
    
    All parameters are inherited from AutomataSystem.
    """
    
    def _interpret(self, output: Any, neighborhood: np.ndarray) -> int:
        """
        Interprets automaton output directly as the next cell state.
        
        Args:
            output: Automaton output (expected to be an integer state).
            neighborhood: Ignored - not used in state-based interpretation.
        
        Returns:
            The output value cast to int.
        """
        return int(output)
