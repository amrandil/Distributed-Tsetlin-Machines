"""
Unit tests for NeighbourhoodAgreementFeedback.
"""

import pytest
import numpy as np
from cakit.feedback import NeighbourhoodAgreementFeedback


class TestNeighbourhoodAgreementInitialization:
    """Tests for NeighbourhoodAgreementFeedback initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        feedback = NeighbourhoodAgreementFeedback()
        assert feedback.radius == 1
        assert feedback.boundary == 'periodic'
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        feedback = NeighbourhoodAgreementFeedback(radius=2, boundary='zero')
        assert feedback.radius == 2
        assert feedback.boundary == 'zero'
    
    def test_invalid_radius(self):
        """Test that radius < 1 raises ValueError."""
        with pytest.raises(ValueError, match="radius must be at least 1"):
            NeighbourhoodAgreementFeedback(radius=0)
    
    def test_invalid_boundary(self):
        """Test that invalid boundary raises ValueError."""
        with pytest.raises(ValueError, match="boundary must be"):
            NeighbourhoodAgreementFeedback(boundary='invalid')


class TestNeighbourhoodAgreementMajority:
    """Tests for majority calculation."""
    
    def test_majority_all_ones(self):
        """Test majority with all ones."""
        feedback = NeighbourhoodAgreementFeedback()
        neighborhood = np.array([1, 1, 1])
        assert feedback._get_majority_state(neighborhood) == 1
    
    def test_majority_all_zeros(self):
        """Test majority with all zeros."""
        feedback = NeighbourhoodAgreementFeedback()
        neighborhood = np.array([0, 0, 0])
        assert feedback._get_majority_state(neighborhood) == 0
    
    def test_majority_more_ones(self):
        """Test majority with more ones than zeros."""
        feedback = NeighbourhoodAgreementFeedback()
        neighborhood = np.array([1, 1, 0])
        assert feedback._get_majority_state(neighborhood) == 1
    
    def test_majority_more_zeros(self):
        """Test majority with more zeros than ones."""
        feedback = NeighbourhoodAgreementFeedback()
        neighborhood = np.array([0, 0, 1])
        assert feedback._get_majority_state(neighborhood) == 0
    
    def test_majority_tie_favors_one(self):
        """Test that ties favor 1."""
        feedback = NeighbourhoodAgreementFeedback()
        neighborhood = np.array([0, 1])
        assert feedback._get_majority_state(neighborhood) == 1
        
        neighborhood = np.array([0, 0, 1, 1])
        assert feedback._get_majority_state(neighborhood) == 1


class TestNeighbourhoodAgreementFeedbackCenter:
    """Tests for feedback evaluation at center positions."""
    
    def test_reward_when_matches_majority(self):
        """Test that cell is rewarded when it matches majority."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # Grid: [1, 1, 1, 0, 0]
        # Position 1: neighborhood is [1, 1, 1], majority is 1, cell is 1 -> reward
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 0, 0])
        
        assert feedback(1, grid_before, grid_after) == True
    
    def test_penalty_when_differs_from_majority(self):
        """Test that cell is penalized when it differs from majority."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # Grid: [1, 0, 1, 0, 0]
        # Position 1: neighborhood is [1, 0, 1], majority is 1, cell is 0 -> penalty
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 0, 1, 0, 0])
        
        assert feedback(1, grid_before, grid_after) == False
    
    def test_reward_with_all_same(self):
        """Test reward when all neighbors are same."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # Grid: all 1s, position 2: neighborhood [1, 1, 1], cell is 1 -> reward
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 1, 1])
        
        assert feedback(2, grid_before, grid_after) == True


class TestNeighbourhoodAgreementFeedbackEdges:
    """Tests for feedback evaluation at edge positions."""
    
    def test_feedback_at_left_edge_periodic(self):
        """Test feedback at left edge with periodic boundary."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # Grid: [1, 0, 0, 0, 1]
        # Position 0: neighborhood is [1, 1, 0] (wraps to position 4), majority is 1, cell is 1
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 0, 0, 0, 1])
        
        assert feedback(0, grid_before, grid_after) == True
    
    def test_feedback_at_right_edge_periodic(self):
        """Test feedback at right edge with periodic boundary."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # Grid: [1, 0, 0, 0, 1]
        # Position 4: neighborhood is [0, 1, 1] (wraps to position 0), majority is 1, cell is 1
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 0, 0, 0, 1])
        
        assert feedback(4, grid_before, grid_after) == True
    
    def test_feedback_at_edge_zero_boundary(self):
        """Test feedback at edge with zero boundary."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='zero')
        
        # Grid: [1, 1, 1, 1, 1]
        # Position 0: neighborhood is [0, 1, 1] (left is padded with 0), majority is 1, cell is 1
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 1, 1])
        
        assert feedback(0, grid_before, grid_after) == True


class TestNeighbourhoodAgreementFeedbackRadius:
    """Tests for different radius values."""
    
    def test_radius_2_center(self):
        """Test feedback with radius 2 at center."""
        feedback = NeighbourhoodAgreementFeedback(radius=2, boundary='periodic')
        
        # Grid: [1, 1, 1, 1, 1, 0, 0]
        # Position 2: neighborhood is [1, 1, 1, 1, 1] (positions 0-4), majority is 1, cell is 1
        grid_before = np.array([0, 0, 0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 1, 1, 0, 0])
        
        assert feedback(2, grid_before, grid_after) == True
    
    def test_radius_2_edge(self):
        """Test feedback with radius 2 at edge."""
        feedback = NeighbourhoodAgreementFeedback(radius=2, boundary='zero')
        
        # Grid: [1, 1, 1, 0, 0]
        # Position 1: neighborhood is [0, 0, 1, 1, 1] (positions -1, 0, 1, 2, 3)
        # Left padding with zeros, majority is 1, cell is 1 -> reward
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 0, 0])
        
        assert feedback(1, grid_before, grid_after) == True


class TestNeighbourhoodAgreementIntegration:
    """Integration tests with actual grid configurations."""
    
    def test_alternating_pattern(self):
        """Test feedback on alternating pattern."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # Alternating: [1, 0, 1, 0, 1]
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 0, 1, 0, 1])
        
        # Position 0: neighborhood [1, 1, 0] (pos 4, 0, 1) -> majority 1, cell 1 -> reward
        assert feedback(0, grid_before, grid_after) == True
        # Position 1: neighborhood [1, 0, 1] -> majority 1, cell 0 -> penalty
        assert feedback(1, grid_before, grid_after) == False
        # Position 2: neighborhood [0, 1, 0] -> majority 0, cell 1 -> penalty
        assert feedback(2, grid_before, grid_after) == False
        # Position 3: neighborhood [1, 0, 1] -> majority 1, cell 0 -> penalty
        assert feedback(3, grid_before, grid_after) == False
        # Position 4: neighborhood [0, 1, 1] (pos 3, 4, 0) -> majority 1, cell 1 -> reward
        assert feedback(4, grid_before, grid_after) == True
    
    def test_uniform_pattern(self):
        """Test feedback on uniform pattern."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='periodic')
        
        # All ones
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 1, 1])
        
        # All cells should be rewarded (all match majority)
        for pos in range(5):
            assert feedback(pos, grid_before, grid_after) == True
    
    def test_block_pattern(self):
        """Test feedback on block pattern."""
        feedback = NeighbourhoodAgreementFeedback(radius=1, boundary='zero')
        
        # Blocks: [1, 1, 1, 0, 0]
        grid_before = np.array([0, 0, 0, 0, 0])
        grid_after = np.array([1, 1, 1, 0, 0])
        
        # Position 0: [0, 1, 1] -> majority 1, cell 1 -> reward
        assert feedback(0, grid_before, grid_after) == True
        
        # Position 1: [1, 1, 1] -> majority 1, cell 1 -> reward
        assert feedback(1, grid_before, grid_after) == True
        
        # Position 2: [1, 1, 0] -> majority 1, cell 1 -> reward
        assert feedback(2, grid_before, grid_after) == True
        
        # Position 3: [1, 0, 0] -> majority 0, cell 0 -> reward
        assert feedback(3, grid_before, grid_after) == True
        
        # Position 4: [0, 0, 0] -> majority 0, cell 0 -> reward
        assert feedback(4, grid_before, grid_after) == True
