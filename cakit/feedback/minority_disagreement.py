"""
Minority disagreement feedback function.

Rewards cells whose state differs from the majority state in their feedback neighborhood.
This encourages diversity and prevents uniform convergence.
"""

from typing import Any
import numpy as np

from .base import FeedbackFunction


class MinorityDisagreementFeedback(FeedbackFunction):
    """
    Minority disagreement feedback function.

    Rewards a cell if its new state DIFFERS from the majority state in its feedback
    neighborhood after the step. This encourages diversity and local heterogeneity,
    preventing the system from converging to uniform states.

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
        Evaluates whether the cell's new state differs from the neighborhood majority.

        Args:
            position: Cell position (int for 1D grids).
            grid_before: Grid state before the step (not used in this implementation).
            grid_after: Grid state after the step.

        Returns:
            True if cell's new state differs from majority in feedback neighborhood, False otherwise.
        """
        neighborhood = self._get_feedback_neighborhood(position, grid_after)
        majority_state = self._get_majority_state(neighborhood)
        return grid_after[position] != majority_state
