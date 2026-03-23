"""
Base feedback function abstract class.

A FeedbackFunction evaluates cell performance and returns reward/penalty signals.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class FeedbackFunction(ABC):
    """
    Abstract base class for feedback functions.

    A FeedbackFunction is called after each step to evaluate each cell's performance.
    It receives the cell position and the grid states before and after the step,
    and returns True for reward or False for penalty.

    Subclasses that operate on a local neighborhood should use the shared helpers
    _get_feedback_neighborhood, _get_boundary_value, and _get_majority_state rather
    than re-implementing them.

    Parameters:
        radius: Neighborhood radius for feedback evaluation. Neighborhood size = 2*radius + 1.
        boundary: Boundary condition for neighborhood extraction. Default: 'periodic'.
    """

    def __init__(self, radius: int = 1, boundary: str = 'periodic'):
        """
        Initialize FeedbackFunction.

        Args:
            radius: Neighborhood radius (>= 1).
            boundary: Boundary condition ('periodic', 'zero', 'fixed').

        Raises:
            ValueError: If radius < 1 or boundary is invalid.
        """
        if radius < 1:
            raise ValueError(f"radius must be at least 1, got {radius}")

        if boundary not in ['periodic', 'zero', 'fixed']:
            raise ValueError(
                f"boundary must be 'periodic', 'zero', or 'fixed', got '{boundary}'"
            )

        self.radius = radius
        self.boundary = boundary

    @abstractmethod
    def __call__(
        self,
        position: Any,
        grid_before: np.ndarray,
        grid_after: np.ndarray
    ) -> bool:
        """
        Evaluates the cell and returns feedback signal.

        Args:
            position: Position of the cell being evaluated. Type depends on grid
                     dimensionality (int for 1D, (row, col) tuple for 2D).
            grid_before: Grid state array before the step.
            grid_after: Grid state array after the step.

        Returns:
            True for reward, False for penalty.
        """
        pass

    def _get_feedback_neighborhood(
        self,
        position: int,
        grid: np.ndarray
    ) -> np.ndarray:
        """
        Extract the feedback neighborhood for a cell.

        Args:
            position: Cell position (int for 1D).
            grid: Grid state array.

        Returns:
            Neighborhood array of length 2*radius + 1.
        """
        size = len(grid)
        neighborhood = np.zeros(2 * self.radius + 1, dtype=int)

        for i, offset in enumerate(range(-self.radius, self.radius + 1)):
            neighbor_pos = position + offset

            if 0 <= neighbor_pos < size:
                neighborhood[i] = grid[neighbor_pos]
            else:
                neighborhood[i] = self._get_boundary_value(neighbor_pos, grid)

        return neighborhood

    def _get_boundary_value(self, position: int, grid: np.ndarray) -> int:
        """
        Get boundary value for out-of-bounds positions.

        Args:
            position: Position (may be < 0 or >= size).
            grid: Grid state array.

        Returns:
            Boundary value according to boundary condition.
        """
        size = len(grid)

        if self.boundary == 'periodic':
            return grid[position % size]
        elif self.boundary in ['zero', 'fixed']:
            return 0
        else:
            raise ValueError(f"Unknown boundary type: {self.boundary}")

    def _get_majority_state(self, neighborhood: np.ndarray) -> int:
        """
        Calculate the majority state in a neighborhood.

        For binary states, returns 1 if count of 1s >= count of 0s, else 0.

        Args:
            neighborhood: Neighborhood array.

        Returns:
            Majority state (0 or 1 for binary).
        """
        count_ones = np.sum(neighborhood == 1)
        count_zeros = np.sum(neighborhood == 0)

        # Tie-breaking: favor 1 if equal
        return 1 if count_ones >= count_zeros else 0
