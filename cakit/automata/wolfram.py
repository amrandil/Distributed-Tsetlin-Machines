"""
Wolfram Automaton implementation.

A plain (non-learning) automaton that applies a fixed Wolfram rule to a neighborhood.
"""

from typing import Any
import numpy as np

from .base import Automaton


class WolframAutomaton(Automaton):
    """
    Wolfram Automaton - applies a fixed Wolfram rule to a neighborhood.
    
    This is a plain automaton with no learning mechanism. The rule is fixed at
    construction time. get_output(neighborhood) applies the rule's lookup table
    to the neighborhood and returns the next cell state.
    
    Wolfram rules are numbered 0-255 for elementary CA (3-cell neighborhoods with
    binary states). The rule number is interpreted as an 8-bit lookup table.
    
    Parameters:
        rule: Wolfram rule number (0-255).
    """
    
    def __init__(self, rule: int):
        """
        Initialize a Wolfram Automaton with a fixed rule.
        
        Args:
            rule: Wolfram rule number. Must be in {0, ..., 255}.
        
        Raises:
            ValueError: If rule is not in valid range.
        """
        if not (0 <= rule <= 255):
            raise ValueError(f"rule must be in {{0, ..., 255}}, got {rule}")
        
        self.rule = rule
        self._lookup_table = self._build_lookup_table(rule)
    
    def _build_lookup_table(self, rule: int) -> dict:
        """
        Build the lookup table for the given Wolfram rule.
        
        The rule number is interpreted as an 8-bit binary number. Each bit
        corresponds to the output for one of the 8 possible 3-cell neighborhood
        configurations (111, 110, 101, 100, 011, 010, 001, 000).
        
        Args:
            rule: Wolfram rule number (0-255).
        
        Returns:
            Dictionary mapping neighborhood tuples to output states.
        """
        binary = format(rule, '08b')
        neighborhoods = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        return {nbhd: int(binary[i]) for i, nbhd in enumerate(neighborhoods)}
    
    def get_output(self, input: Any = None) -> int:
        """
        Applies the Wolfram rule to the neighborhood and returns the next state.
        
        Args:
            input: Neighborhood array. Expected to be a sequence of 3 binary values
                   for elementary CA. Can be a list, tuple, or numpy array.
        
        Returns:
            Next cell state (0 or 1) according to the rule's lookup table.
        
        Raises:
            ValueError: If input is None or has wrong length.
        """
        if input is None:
            raise ValueError("WolframAutomaton.get_output() requires a neighborhood input")
        
        # Convert to tuple for lookup
        if isinstance(input, np.ndarray):
            neighborhood = tuple(input.tolist())
        else:
            neighborhood = tuple(input)
        
        if len(neighborhood) != 3:
            raise ValueError(
                f"WolframAutomaton expects 3-cell neighborhood, got {len(neighborhood)}"
            )
        
        return self._lookup_table[neighborhood]
    
    def get_state(self) -> int:
        """
        Returns the rule number (the automaton's "state").
        
        Since this is a stateless automaton (the rule doesn't change), the state
        is just the fixed rule number.
        
        Returns:
            The rule number.
        """
        return self.rule
    
    def reset(self) -> None:
        """
        Reset operation for WolframAutomaton is a no-op.
        
        Since the rule is fixed and there is no internal evolving state,
        there is nothing to reset.
        """
        pass
