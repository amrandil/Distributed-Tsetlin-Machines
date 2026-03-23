"""
Base automaton abstract class.

An Automaton has an internal state and produces an output given its current state.
"""

from abc import ABC, abstractmethod
from typing import Any


class Automaton(ABC):
    """
    Abstract base class for all automata.
    
    An Automaton has an internal state and produces an output based on that state
    and an external input. The meaning of the output is determined by the system
    that uses the automaton, not by the automaton itself.
    """
    
    @abstractmethod
    def get_output(self, input: Any = None) -> Any:
        """
        Returns the automaton's output based on its current internal state.
        
        For a plain automaton, this typically processes the input and returns
        a next state. For a LearningAutomaton, this returns an action index
        and may ignore the input.
        
        Args:
            input: External input to the automaton. Type depends on the concrete
                   implementation. For CA cells this is typically a neighborhood array.
        
        Returns:
            The automaton's output. Type depends on the concrete implementation.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Any:
        """
        Returns the current internal state of the automaton.
        
        Returns:
            The current internal state. Type depends on the concrete implementation.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the automaton to its initial state.
        """
        pass
