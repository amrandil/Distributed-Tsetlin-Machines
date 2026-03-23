"""
RuleSystem implementation.

A system where each cell's automaton output selects a Wolfram rule to apply to the neighborhood.
"""

from typing import Any, List
import numpy as np

from .base import AutomataSystem


class RuleSystem(AutomataSystem):
    """
    Rule-selection automata system.
    
    In a RuleSystem, the automaton's output is interpreted as an index into a list of
    Wolfram rules. The selected rule is applied to the cell's neighborhood to produce
    the next cell state.
    
    Used as a pure rule-selection CA when feedback_fn=None, or as a rule-selection CLA
    when feedback_fn is provided and the automaton is a LearningAutomaton.
    
    Parameters:
        rules: List of Wolfram rule numbers. rules[i] is the rule selected by output i.
               Length must equal the automaton's action_set_size.
        ... (all other parameters inherited from AutomataSystem)
    """
    
    def __init__(self, rules: List[int], *args, **kwargs):
        """
        Initialize a RuleSystem.
        
        Args:
            rules: List of Wolfram rule numbers (0-255).
            *args, **kwargs: Passed to AutomataSystem.__init__().
        
        Raises:
            ValueError: If any rule is out of valid range.
        """
        for rule in rules:
            if not (0 <= rule <= 255):
                raise ValueError(f"All rules must be in {{0, ..., 255}}, got {rule}")
        
        self.rules = rules
        self._rule_lookup_tables = [self._build_lookup_table(r) for r in rules]
        
        super().__init__(*args, **kwargs)
    
    def _build_lookup_table(self, rule: int) -> dict:
        """
        Build the lookup table for a Wolfram rule.
        
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
    
    def _interpret(self, output: Any, neighborhood: np.ndarray) -> int:
        """
        Interprets automaton output as a rule index and applies that rule to the neighborhood.
        
        Args:
            output: Automaton output (expected to be an integer index into self.rules).
            neighborhood: Neighborhood array (expected to be length 3 for elementary CA).
        
        Returns:
            Next cell state according to the selected rule's lookup table.
        """
        rule_index = int(output)
        lookup_table = self._rule_lookup_tables[rule_index]
        
        # Convert neighborhood to tuple for lookup
        neighborhood_tuple = tuple(neighborhood.tolist())
        
        return lookup_table[neighborhood_tuple]
