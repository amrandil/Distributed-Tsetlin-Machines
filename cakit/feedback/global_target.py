"""
Global target feedback function.

Rewards cells based on whether they match a global target state, independent of
their neighbors. This creates directed learning toward a specific configuration.
"""

from typing import Any
import numpy as np

from .base import FeedbackFunction


class GlobalTargetFeedback(FeedbackFunction):
    """
    Global target feedback function.
    
    Rewards a cell if its state matches a specified target state. This encourages
    the entire system to converge toward a uniform global configuration.
    
    This feedback is useful for:
    - Demonstrating collective learning
    - Studying convergence dynamics
    - Testing how quickly TAs can learn a simple goal
    
    Parameters:
        target_state: The desired state for all cells (e.g., 0 or 1).
    """
    
    def __init__(self, target_state: int = 1):
        """
        Initialize GlobalTargetFeedback.
        
        Args:
            target_state: Target state for all cells. Default: 1 (all "on").
        """
        self.target_state = target_state
    
    def __call__(
        self,
        position: Any,
        grid_before: np.ndarray,
        grid_after: np.ndarray
    ) -> bool:
        """
        Evaluates whether the cell's new state matches the global target.
        
        Args:
            position: Cell position (int for 1D grids).
            grid_before: Grid state before the step (not used).
            grid_after: Grid state after the step.
        
        Returns:
            True if cell's new state matches the target, False otherwise.
        """
        cell_state = grid_after[position]
        return cell_state == self.target_state
