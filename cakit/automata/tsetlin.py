"""
Tsetlin Automaton implementation.

A 2-armed Fixed Structure Stochastic Automaton (FSSA) with deterministic transitions.
"""

import random
from typing import Any, Union

from .learning import LearningAutomaton


class TsetlinAutomaton(LearningAutomaton):
    """
    Tsetlin Automaton - a 2-armed FSSA with deterministic state transitions.
    
    The automaton has 2N states divided into two arms of N states each.
    States {1, ..., N} correspond to action 0, states {N+1, ..., 2N} correspond to action 1.
    
    On reward: moves one step deeper into the current arm (toward the extreme end).
    On penalty: moves one step back toward the boundary (transition point between arms).
    
    Parameters:
        n_states: Number of states per arm. Total states = 2 * n_states.
        initial_state: Starting state in {1, ..., 2N}, or 'random' for uniform sampling.
    """
    
    def __init__(self, n_states: int = 5, initial_state: Union[int, str] = 'random'):
        """
        Initialize a Tsetlin Automaton.
        
        Args:
            n_states: Number of states per arm (N). Must be >= 1. Default: 5.
            initial_state: Starting state. Either an integer in {1, ..., 2N},
                          or 'random' for uniform random initialization.
        
        Raises:
            ValueError: If n_states < 1 or initial_state is out of valid range.
        """
        super().__init__(action_set_size=2)
        
        if n_states < 1:
            raise ValueError("n_states must be at least 1")
        
        self.n_states = n_states
        self.total_states = 2 * n_states
        
        if initial_state == 'random':
            self._state = random.randint(1, self.total_states)
            self._initial_state = self._state
        elif isinstance(initial_state, int):
            if not (1 <= initial_state <= self.total_states):
                raise ValueError(
                    f"initial_state must be in {{1, ..., {self.total_states}}} "
                    f"or 'random', got {initial_state}"
                )
            self._state = initial_state
            self._initial_state = initial_state
        else:
            raise ValueError(
                f"initial_state must be an integer or 'random', got {initial_state}"
            )
    
    def get_output(self, input: Any = None) -> int:
        """
        Returns the current action index (arm) based on the current state.
        
        Args:
            input: Ignored. Accepted for interface consistency.
        
        Returns:
            Action index: 0 if in arm 0 (states 1..N), 1 if in arm 1 (states N+1..2N).
        """
        return self.get_arm()
    
    def get_state(self) -> int:
        """
        Returns the current internal state.
        
        Returns:
            Current state in {1, ..., 2N}.
        """
        return self._state
    
    def get_arm(self) -> int:
        """
        Returns the current arm (action) index.
        
        Returns:
            0 if state is in {1, ..., N}, 1 if state is in {N+1, ..., 2N}.
        """
        return (self._state - 1) // self.n_states
    
    def get_position(self) -> int:
        """
        Returns the position within the current arm.
        
        Position 0 is the boundary end (least committed to the current action).
        Position N-1 is the extreme end (most committed to the current action).
        
        Returns:
            Position in {0, ..., N-1}.
        """
        return (self._state - 1) % self.n_states
    
    def reward(self) -> None:
        """
        Apply the reward transition.
        
        Moves one step deeper into the current arm (toward the extreme end),
        unless already at the extreme end.
        """
        position = self.get_position()
        if position < self.n_states - 1:
            self._state += 1
    
    def penalize(self) -> None:
        """
        Apply the penalty transition.
        
        Moves one step toward the boundary (transition point between arms).
        If already at the boundary, switches to the boundary of the opposite arm.
        """
        position = self.get_position()
        if position > 0:
            # Not at boundary - move one step toward boundary
            self._state -= 1
        else:
            # At boundary - switch to opposite arm's boundary
            current_arm = self.get_arm()
            if current_arm == 0:
                # Switch from arm 0 boundary (state 1) to arm 1 boundary (state N+1)
                self._state = self.n_states + 1
            else:
                # Switch from arm 1 boundary (state N+1) to arm 0 boundary (state 1)
                self._state = 1
    
    def reset(self) -> None:
        """
        Resets the automaton to its initial state.
        """
        self._state = self._initial_state
