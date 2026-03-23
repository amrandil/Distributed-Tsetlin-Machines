"""
Base grid abstract class.

A Grid owns all spatial mechanics: cell state storage, neighborhood extraction,
and boundary condition handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Union
import numpy as np


class Grid(ABC):
    """
    Abstract base class for all grids.
    
    A Grid provides spatial structure for cellular automata. It handles cell state
    storage, neighborhood extraction, boundary conditions, and position iteration.
    The Grid is dimension-agnostic - subclasses implement 1D, 2D, or higher dimensions.
    
    Parameters:
        boundary: Boundary condition type. One of 'periodic', 'zero', or 'fixed'.
        initial_states: Initial cell states. Can be a numpy array of the correct shape,
                       or a string key: 'random', 'single_center', 'half'.
        k_states: Number of possible cell states. Default: 2 (binary).
    """
    
    def __init__(
        self,
        boundary: str = 'periodic',
        initial_states: Union[np.ndarray, str] = 'random',
        k_states: int = 2,
        boundary_value: int = 0
    ):
        """
        Initialize a Grid.

        Args:
            boundary: Boundary condition. One of 'periodic', 'zero', 'fixed'.
            initial_states: Initial states array or initialization key string.
            k_states: Number of possible cell states (>= 2).
            boundary_value: Constant cell state used when boundary='fixed'. Default: 0.

        Raises:
            ValueError: If boundary type is invalid or k_states < 2.
        """
        if boundary not in ['periodic', 'zero', 'fixed']:
            raise ValueError(
                f"boundary must be 'periodic', 'zero', or 'fixed', got '{boundary}'"
            )

        if k_states < 2:
            raise ValueError(f"k_states must be at least 2, got {k_states}")

        self.boundary = boundary
        self.k_states = k_states
        self.boundary_value = boundary_value
        self._initial_states_spec = initial_states
    
    @abstractmethod
    def get_neighborhood(self, position: Any) -> np.ndarray:
        """
        Returns the neighborhood array for the cell at the given position.
        
        Args:
            position: Cell position. Type depends on grid dimensionality.
                     int for 1D, (row, col) tuple for 2D.
        
        Returns:
            Numpy array of cell states in the neighborhood.
        """
        pass
    
    @abstractmethod
    def get_state(self, position: Any) -> int:
        """
        Returns the current state of the cell at the given position.
        
        Args:
            position: Cell position.
        
        Returns:
            Current state of the cell (integer in {0, ..., k_states-1}).
        """
        pass
    
    @abstractmethod
    def set_state(self, position: Any, state: int) -> None:
        """
        Sets the state of the cell at the given position.
        
        Args:
            position: Cell position.
            state: New state for the cell.
        """
        pass
    
    @abstractmethod
    def get_all_states(self) -> np.ndarray:
        """
        Returns the full grid state array.
        
        Returns:
            Numpy array containing all cell states.
        """
        pass
    
    @abstractmethod
    def set_all_states(self, states: np.ndarray) -> None:
        """
        Sets the full grid state array.
        
        Args:
            states: New state array. Must match the grid's shape.
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Iterable[Any]:
        """
        Returns an iterable of all valid cell positions.
        
        Returns:
            Iterable yielding positions. int for 1D, (row, col) tuples for 2D.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the grid to its initial state.
        """
        pass
