"""
1D grid implementation.

A 1D grid where cells are indexed by a single integer and neighborhoods are
contiguous slices of the array.
"""

from typing import Iterable, Union
import numpy as np

from .base import Grid


class Grid1D(Grid):
    """
    1D grid implementation for cellular automata.
    
    Cells are indexed by integers from 0 to size-1. The neighborhood of a cell
    is a contiguous slice of radius r: {i-r, ..., i, ..., i+r} with size 2r+1.
    
    Parameters:
        size: Number of cells in the grid (>= 3).
        radius: Neighborhood radius (>= 1). Neighborhood size = 2*radius + 1.
        boundary: Boundary condition ('periodic', 'zero', 'fixed').
        initial_states: Initial states array or key string.
        k_states: Number of possible cell states.
    """
    
    def __init__(
        self,
        size: int,
        radius: int = 1,
        boundary: str = 'periodic',
        initial_states: Union[np.ndarray, str] = 'random',
        k_states: int = 2,
        boundary_value: int = 0
    ):
        """
        Initialize a 1D grid.

        Args:
            size: Number of cells. Must be >= 3.
            radius: Neighborhood radius. Must be >= 1.
            boundary: Boundary condition type.
            initial_states: Initial states or initialization key.
            k_states: Number of possible cell states.
            boundary_value: Constant used when boundary='fixed'. Default: 0.

        Raises:
            ValueError: If size < 3 or radius < 1.
        """
        if size < 3:
            raise ValueError(f"size must be at least 3, got {size}")

        if radius < 1:
            raise ValueError(f"radius must be at least 1, got {radius}")

        super().__init__(boundary=boundary, initial_states=initial_states, k_states=k_states, boundary_value=boundary_value)
        
        self.size = size
        self.radius = radius
        self.neighborhood_size = 2 * radius + 1
        
        # Initialize grid states
        self._states = self._initialize_states(initial_states)
        self._initial_states = self._states.copy()
    
    def _initialize_states(self, initial_states: Union[np.ndarray, str]) -> np.ndarray:
        """
        Initialize the grid state array.
        
        Args:
            initial_states: Either a numpy array or a key string.
        
        Returns:
            Initialized state array.
        
        Raises:
            ValueError: If initial_states is invalid.
        """
        if isinstance(initial_states, np.ndarray):
            if initial_states.shape != (self.size,):
                raise ValueError(
                    f"initial_states array must have shape ({self.size},), "
                    f"got {initial_states.shape}"
                )
            return initial_states.copy()
        
        elif isinstance(initial_states, str):
            if initial_states == 'random':
                return np.random.randint(0, self.k_states, size=self.size)
            
            elif initial_states == 'single_center':
                states = np.zeros(self.size, dtype=int)
                states[self.size // 2] = 1
                return states
            
            elif initial_states == 'half':
                states = np.zeros(self.size, dtype=int)
                states[:self.size // 2] = 1
                return states
            
            else:
                raise ValueError(
                    f"initial_states string must be 'random', 'single_center', or 'half', "
                    f"got '{initial_states}'"
                )
        
        else:
            raise ValueError(
                f"initial_states must be numpy array or string, got {type(initial_states)}"
            )
    
    def get_neighborhood(self, position: int) -> np.ndarray:
        """
        Returns the neighborhood array for the cell at the given position.
        
        The neighborhood consists of 2*radius + 1 cells centered on position,
        with boundary conditions applied at the edges.
        
        Args:
            position: Cell index (0 to size-1).
        
        Returns:
            Numpy array of neighborhood states, length 2*radius + 1.
        
        Raises:
            ValueError: If position is out of bounds.
        """
        if not (0 <= position < self.size):
            raise ValueError(
                f"position must be in {{0, ..., {self.size-1}}}, got {position}"
            )
        
        neighborhood = np.zeros(self.neighborhood_size, dtype=int)
        
        for i, offset in enumerate(range(-self.radius, self.radius + 1)):
            neighbor_pos = position + offset
            
            if 0 <= neighbor_pos < self.size:
                # Within bounds
                neighborhood[i] = self._states[neighbor_pos]
            else:
                # Out of bounds - apply boundary condition
                neighborhood[i] = self._get_boundary_value(neighbor_pos)
        
        return neighborhood
    
    def _get_boundary_value(self, position: int) -> int:
        """
        Get the value at a position outside the grid bounds, applying boundary conditions.
        
        Args:
            position: Position index (may be negative or >= size).
        
        Returns:
            Cell state according to boundary condition.
        """
        if self.boundary == 'periodic':
            # Wrap around
            return self._states[position % self.size]
        
        elif self.boundary == 'zero':
            # Pad with zeros
            return 0
        
        elif self.boundary == 'fixed':
            return self.boundary_value
        
        else:
            raise ValueError(f"Unknown boundary type: {self.boundary}")
    
    def get_state(self, position: int) -> int:
        """
        Returns the current state of the cell at the given position.
        
        Args:
            position: Cell index (0 to size-1).
        
        Returns:
            Current state of the cell.
        
        Raises:
            ValueError: If position is out of bounds.
        """
        if not (0 <= position < self.size):
            raise ValueError(
                f"position must be in {{0, ..., {self.size-1}}}, got {position}"
            )
        return int(self._states[position])
    
    def set_state(self, position: int, state: int) -> None:
        """
        Sets the state of the cell at the given position.
        
        Args:
            position: Cell index (0 to size-1).
            state: New state for the cell.
        
        Raises:
            ValueError: If position is out of bounds.
        """
        if not (0 <= position < self.size):
            raise ValueError(
                f"position must be in {{0, ..., {self.size-1}}}, got {position}"
            )
        self._states[position] = state
    
    def get_all_states(self) -> np.ndarray:
        """
        Returns a copy of the full grid state array.
        
        Returns:
            Numpy array of length size containing all cell states.
        """
        return self._states.copy()
    
    def set_all_states(self, states: np.ndarray) -> None:
        """
        Sets the full grid state array.
        
        Args:
            states: New state array. Must have shape (size,).
        
        Raises:
            ValueError: If states has wrong shape.
        """
        if states.shape != (self.size,):
            raise ValueError(
                f"states must have shape ({self.size},), got {states.shape}"
            )
        self._states = states.copy()
    
    def get_positions(self) -> Iterable[int]:
        """
        Returns an iterable of all valid cell positions.
        
        Returns:
            Range object yielding integers from 0 to size-1.
        """
        return range(self.size)
    
    def reset(self) -> None:
        """
        Resets the grid to its initial state.
        """
        self._states = self._initial_states.copy()
