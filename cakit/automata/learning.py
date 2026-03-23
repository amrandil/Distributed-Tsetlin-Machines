"""
Learning automaton abstract class.

A LearningAutomaton extends Automaton with reinforcement learning capabilities.
"""

from abc import abstractmethod
from typing import Any

from .base import Automaton


class LearningAutomaton(Automaton):
    """
    Abstract base class for all learning automata.
    
    A LearningAutomaton extends Automaton with reward and penalty mechanisms.
    It learns over time which action to prefer based on feedback from its environment.
    
    The get_output() method for a LearningAutomaton returns an action index from
    the action set {0, 1, ..., action_set_size - 1}.
    """
    
    def __init__(self, action_set_size: int = 2):
        """
        Initialize a learning automaton.
        
        Args:
            action_set_size: Number of actions available. Must be >= 2.
                             Action indices are {0, 1, ..., action_set_size - 1}.
        """
        if action_set_size < 2:
            raise ValueError("action_set_size must be at least 2")
        self.action_set_size = action_set_size
    
    @abstractmethod
    def get_output(self, input: Any = None) -> int:
        """
        Returns the current action index based on the internal state.
        
        For learning automata, the input parameter is typically ignored since
        the action is determined by the automaton's internal learning state,
        not by external input.
        
        Args:
            input: Ignored for LearningAutomaton. Accepted for interface consistency.
        
        Returns:
            Action index in {0, 1, ..., action_set_size - 1}.
        """
        pass
    
    @abstractmethod
    def reward(self) -> None:
        """
        Apply the reward transition to the internal structure.
        
        Moves the automaton toward committing to the current action.
        """
        pass
    
    @abstractmethod
    def penalize(self) -> None:
        """
        Apply the penalty transition to the internal structure.
        
        Moves the automaton away from the current action.
        """
        pass
