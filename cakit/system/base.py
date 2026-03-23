"""
Base automata system abstract class.

An AutomataSystem attaches automata to a grid and evolves them over time.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import random

from ..grid.base import Grid
from ..automata.base import Automaton


class AutomataSystem(ABC):
    """
    Abstract base class for all automata systems.
    
    An AutomataSystem owns the step loop, history recording, and automaton grid
    construction. It is automaton-agnostic and dimension-agnostic, delegating all
    spatial operations to its Grid instance.
    
    Parameters:
        grid: A configured Grid instance (Grid1D, Grid2D, etc.).
        automaton_type: The Automaton subclass to instantiate at each cell.
        automaton_params: Keyword arguments forwarded to each automaton constructor.
        feedback_fn: Optional feedback function. If provided, called after each step
                    to reward/penalize learning automata. Signature:
                    (position, grid_before, grid_after) -> bool (True=reward, False=penalty).
        feedback_radius: Neighborhood radius used by feedback_fn. Only relevant when
                        feedback_fn is not None.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        grid: Grid,
        automaton_type: type,
        automaton_params: Optional[Dict[str, Any]] = None,
        feedback_fn: Optional[Callable[[Any, np.ndarray, np.ndarray], bool]] = None,
        feedback_radius: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize an AutomataSystem.
        
        Args:
            grid: Grid instance providing spatial structure.
            automaton_type: Class of automaton to instantiate at each cell.
            automaton_params: Parameters for automaton constructor. Default: {}.
            feedback_fn: Optional feedback function for learning automata.
            feedback_radius: Radius for feedback neighborhood.
            seed: Random seed.
        """
        self.grid = grid
        self.automaton_type = automaton_type
        self.automaton_params = automaton_params or {}
        self.feedback_fn = feedback_fn
        self.feedback_radius = feedback_radius
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Build automaton grid
        self.automata = self._build_automata()

        # History tracking — record generation 0 immediately
        self.generation = 0
        self.grid_history = []
        self.automaton_state_history = []
        self._record_state()
    
    def _record_state(self) -> None:
        """
        Appends the current grid and automaton states to the history.

        Called once at initialisation (generation 0) and once at the end of
        every step, so history is always consistent regardless of whether the
        caller uses step() or run().
        """
        self.grid_history.append(self.grid.get_all_states().copy())
        automaton_states = np.array([
            self.automata[pos].get_state()
            for pos in self.grid.get_positions()
        ])
        self.automaton_state_history.append(automaton_states)

    def _build_automata(self) -> Dict[Any, Automaton]:
        """
        Instantiate one automaton per grid position.
        
        Returns:
            Dictionary mapping position to automaton instance.
        """
        automata = {}
        for pos in self.grid.get_positions():
            automata[pos] = self.automaton_type(**self.automaton_params)
        return automata
    
    @abstractmethod
    def _interpret(self, output: Any, neighborhood: np.ndarray) -> int:
        """
        Translates automaton output into a concrete next cell state.
        
        This is the extension point that distinguishes different system types.
        StateSystem returns output directly. RuleSystem applies a rule to the neighborhood.
        
        Args:
            output: Output from an automaton's get_output() method.
            neighborhood: Neighborhood array for the cell.
        
        Returns:
            Next state for the cell (integer).
        """
        pass
    
    def step(self) -> None:
        """
        Advances the system by one generation.
        
        Computes next states for all cells, updates the grid, applies feedback
        (if feedback_fn is not None), and records history.
        """
        # Save current grid state for feedback comparison
        grid_before = self.grid.get_all_states()
        
        # Compute next states
        next_states = np.zeros(len(list(self.grid.get_positions())), dtype=int)
        
        for i, pos in enumerate(self.grid.get_positions()):
            neighborhood = self.grid.get_neighborhood(pos)
            output = self.automata[pos].get_output(neighborhood)
            next_states[i] = self._interpret(output, neighborhood)
        
        # Update grid
        self.grid.set_all_states(next_states)
        grid_after = self.grid.get_all_states()
        
        # Apply feedback if provided
        if self.feedback_fn is not None:
            for pos in self.grid.get_positions():
                reward = self.feedback_fn(pos, grid_before, grid_after)
                automaton = self.automata[pos]
                
                # Call reward() or penalize() - assumes automaton is a LearningAutomaton
                if reward:
                    automaton.reward()
                else:
                    automaton.penalize()
        
        self._record_state()
        self.generation += 1
    
    def run(self, generations: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the system for a specified number of generations.
        
        Args:
            generations: Number of generations to run.
        
        Returns:
            Tuple of (grid_history, automaton_state_history).
            - grid_history: shape (generations, grid_size)
            - automaton_state_history: shape (generations, grid_size)
        """
        # Reset history to just generation 0 (current state)
        self.grid_history = []
        self.automaton_state_history = []
        self.generation = 0
        self._record_state()

        for _ in range(generations):
            self.step()
        
        return np.array(self.grid_history), np.array(self.automaton_state_history)
    
    def reset(self) -> None:
        """
        Resets all automata and the grid to their initial states.

        After reset, history contains exactly one entry: generation 0.
        """
        self.grid.reset()
        for automaton in self.automata.values():
            automaton.reset()
        self.generation = 0
        self.grid_history = []
        self.automaton_state_history = []
        self._record_state()
