"""
Neighbourhood agreement feedback function.

Rewards cells whose new state matches the majority state in their feedback neighborhood.
"""

from typing import Any
import numpy as np

from .base import FeedbackFunction


class NeighbourhoodAgreementFeedback(FeedbackFunction):
    """
    Neighbourhood agreement feedback function.

    Rewards a cell if its new state matches the majority state in its feedback
    neighborhood after the step. This encourages local agreement/synchronization.

    Parameters:
        radius: Neighborhood radius for feedback evaluation. Neighborhood size = 2*radius + 1.
        boundary: Boundary condition for neighborhood extraction. Default: 'periodic'.
    """

    def __call__(
        self,
        position: Any,
        grid_before: np.ndarray,
        grid_after: np.ndarray
    ) -> bool:
        """
        Evaluates whether the cell's new state matches the neighborhood majority.

        Args:
            position: Cell position (int for 1D grids).
            grid_before: Grid state before the step (not used in this implementation).
            grid_after: Grid state after the step.

        Returns:
            True if cell's new state matches majority in feedback neighborhood, False otherwise.
        """
        neighborhood = self._get_feedback_neighborhood(position, grid_after)
        majority_state = self._get_majority_state(neighborhood)
        return grid_after[position] == majority_state
