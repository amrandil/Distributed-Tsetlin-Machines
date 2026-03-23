"""
Unit tests for Grid1D.
"""

import pytest
import numpy as np
from cakit.grid import Grid1D


class TestGrid1DInitialization:
    """Tests for Grid1D initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        grid = Grid1D(size=10)
        assert grid.size == 10
        assert grid.radius == 1
        assert grid.boundary == 'periodic'
        assert grid.k_states == 2
        assert grid.neighborhood_size == 3
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        grid = Grid1D(size=20, radius=2, boundary='zero', k_states=3)
        assert grid.size == 20
        assert grid.radius == 2
        assert grid.boundary == 'zero'
        assert grid.k_states == 3
        assert grid.neighborhood_size == 5
    
    def test_invalid_size(self):
        """Test that size < 3 raises ValueError."""
        with pytest.raises(ValueError, match="size must be at least 3"):
            Grid1D(size=2)
    
    def test_invalid_radius(self):
        """Test that radius < 1 raises ValueError."""
        with pytest.raises(ValueError, match="radius must be at least 1"):
            Grid1D(size=10, radius=0)
    
    def test_invalid_boundary(self):
        """Test that invalid boundary type raises ValueError."""
        with pytest.raises(ValueError, match="boundary must be"):
            Grid1D(size=10, boundary='invalid')
    
    def test_invalid_k_states(self):
        """Test that k_states < 2 raises ValueError."""
        with pytest.raises(ValueError, match="k_states must be at least 2"):
            Grid1D(size=10, k_states=1)


class TestGrid1DInitialStates:
    """Tests for different initial state configurations."""
    
    def test_random_initialization(self):
        """Test random initialization."""
        grid = Grid1D(size=10, initial_states='random')
        states = grid.get_all_states()
        assert states.shape == (10,)
        assert all(0 <= s < 2 for s in states)
    
    def test_single_center_initialization(self):
        """Test single_center initialization."""
        grid = Grid1D(size=11, initial_states='single_center')
        states = grid.get_all_states()
        
        # Only the center cell should be 1
        assert states[5] == 1
        assert np.sum(states) == 1
        assert all(s in [0, 1] for s in states)
    
    def test_half_initialization(self):
        """Test half initialization."""
        grid = Grid1D(size=10, initial_states='half')
        states = grid.get_all_states()
        
        # First half should be 1, second half should be 0
        assert all(states[:5] == 1)
        assert all(states[5:] == 0)
    
    def test_array_initialization(self):
        """Test initialization with explicit array."""
        initial = np.array([1, 0, 1, 0, 1])
        grid = Grid1D(size=5, initial_states=initial)
        states = grid.get_all_states()
        np.testing.assert_array_equal(states, initial)
    
    def test_array_wrong_shape_raises(self):
        """Test that wrong shape array raises ValueError."""
        initial = np.array([1, 0, 1])
        with pytest.raises(ValueError, match="must have shape"):
            Grid1D(size=5, initial_states=initial)
    
    def test_invalid_initial_states_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="must be 'random', 'single_center', or 'half'"):
            Grid1D(size=10, initial_states='invalid')
    
    def test_invalid_initial_states_type(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="must be numpy array or string"):
            Grid1D(size=10, initial_states=[1, 0, 1])


class TestGrid1DGetSetState:
    """Tests for get_state and set_state."""
    
    def test_get_state(self):
        """Test getting individual cell states."""
        initial = np.array([1, 0, 1, 0, 1])
        grid = Grid1D(size=5, initial_states=initial)
        
        assert grid.get_state(0) == 1
        assert grid.get_state(1) == 0
        assert grid.get_state(2) == 1
        assert grid.get_state(3) == 0
        assert grid.get_state(4) == 1
    
    def test_set_state(self):
        """Test setting individual cell states."""
        grid = Grid1D(size=5, initial_states='single_center')
        
        grid.set_state(0, 1)
        assert grid.get_state(0) == 1
        
        grid.set_state(2, 0)
        assert grid.get_state(2) == 0
    
    def test_get_state_out_of_bounds(self):
        """Test that out of bounds position raises ValueError."""
        grid = Grid1D(size=5)
        
        with pytest.raises(ValueError, match="position must be in"):
            grid.get_state(-1)
        
        with pytest.raises(ValueError, match="position must be in"):
            grid.get_state(5)
    
    def test_set_state_out_of_bounds(self):
        """Test that out of bounds position raises ValueError."""
        grid = Grid1D(size=5)
        
        with pytest.raises(ValueError, match="position must be in"):
            grid.set_state(-1, 1)
        
        with pytest.raises(ValueError, match="position must be in"):
            grid.set_state(5, 1)


class TestGrid1DGetSetAllStates:
    """Tests for get_all_states and set_all_states."""
    
    def test_get_all_states(self):
        """Test getting all states."""
        initial = np.array([1, 0, 1, 0, 1])
        grid = Grid1D(size=5, initial_states=initial)
        
        states = grid.get_all_states()
        np.testing.assert_array_equal(states, initial)
    
    def test_get_all_states_returns_copy(self):
        """Test that get_all_states returns a copy."""
        grid = Grid1D(size=5, initial_states='single_center')
        states1 = grid.get_all_states()
        states1[0] = 1
        
        states2 = grid.get_all_states()
        assert states2[0] == 0  # Original unchanged
    
    def test_set_all_states(self):
        """Test setting all states."""
        grid = Grid1D(size=5, initial_states='single_center')
        
        new_states = np.array([1, 1, 0, 0, 1])
        grid.set_all_states(new_states)
        
        np.testing.assert_array_equal(grid.get_all_states(), new_states)
    
    def test_set_all_states_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        grid = Grid1D(size=5)
        
        with pytest.raises(ValueError, match="must have shape"):
            grid.set_all_states(np.array([1, 0, 1]))


class TestGrid1DGetPositions:
    """Tests for get_positions."""
    
    def test_get_positions(self):
        """Test that get_positions returns all indices."""
        grid = Grid1D(size=5)
        positions = list(grid.get_positions())
        assert positions == [0, 1, 2, 3, 4]
    
    def test_get_positions_iteration(self):
        """Test iterating over positions."""
        grid = Grid1D(size=3)
        count = 0
        for pos in grid.get_positions():
            assert 0 <= pos < 3
            count += 1
        assert count == 3


class TestGrid1DReset:
    """Tests for reset functionality."""
    
    def test_reset_restores_initial_state(self):
        """Test that reset restores initial state."""
        initial = np.array([1, 0, 1, 0, 1])
        grid = Grid1D(size=5, initial_states=initial)
        
        # Modify the grid
        grid.set_all_states(np.zeros(5, dtype=int))
        assert np.sum(grid.get_all_states()) == 0
        
        # Reset should restore initial state
        grid.reset()
        np.testing.assert_array_equal(grid.get_all_states(), initial)
    
    def test_reset_after_random_initialization(self):
        """Test reset with random initialization."""
        grid = Grid1D(size=5, initial_states='random')
        initial = grid.get_all_states()
        
        # Modify the grid
        grid.set_all_states(np.ones(5, dtype=int))
        
        # Reset should restore initial random state
        grid.reset()
        np.testing.assert_array_equal(grid.get_all_states(), initial)


class TestGrid1DNeighborhoodPeriodic:
    """Tests for neighborhood extraction with periodic boundary."""
    
    def test_neighborhood_center(self):
        """Test neighborhood in the center of the grid."""
        grid = Grid1D(size=5, radius=1, boundary='periodic',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        nbhd = grid.get_neighborhood(2)
        np.testing.assert_array_equal(nbhd, [1, 0, 1])
    
    def test_neighborhood_at_left_edge(self):
        """Test neighborhood at left edge with periodic wrapping."""
        grid = Grid1D(size=5, radius=1, boundary='periodic',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        # Position 0: neighbors are [4, 0, 1]
        nbhd = grid.get_neighborhood(0)
        np.testing.assert_array_equal(nbhd, [0, 0, 1])
    
    def test_neighborhood_at_right_edge(self):
        """Test neighborhood at right edge with periodic wrapping."""
        grid = Grid1D(size=5, radius=1, boundary='periodic',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        # Position 4: neighbors are [3, 4, 0]
        nbhd = grid.get_neighborhood(4)
        np.testing.assert_array_equal(nbhd, [1, 0, 0])
    
    def test_neighborhood_radius_2(self):
        """Test neighborhood with radius 2."""
        grid = Grid1D(size=7, radius=2, boundary='periodic',
                     initial_states=np.array([1, 0, 1, 0, 1, 0, 1]))
        
        # Position 3: neighbors are [1, 2, 3, 4, 5]
        nbhd = grid.get_neighborhood(3)
        np.testing.assert_array_equal(nbhd, [0, 1, 0, 1, 0])


class TestGrid1DNeighborhoodZero:
    """Tests for neighborhood extraction with zero boundary."""
    
    def test_neighborhood_center(self):
        """Test neighborhood in the center."""
        grid = Grid1D(size=5, radius=1, boundary='zero',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        nbhd = grid.get_neighborhood(2)
        np.testing.assert_array_equal(nbhd, [1, 0, 1])
    
    def test_neighborhood_at_left_edge(self):
        """Test neighborhood at left edge with zero padding."""
        grid = Grid1D(size=5, radius=1, boundary='zero',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        # Position 0: left neighbor is out of bounds → 0
        nbhd = grid.get_neighborhood(0)
        np.testing.assert_array_equal(nbhd, [0, 0, 1])
    
    def test_neighborhood_at_right_edge(self):
        """Test neighborhood at right edge with zero padding."""
        grid = Grid1D(size=5, radius=1, boundary='zero',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        # Position 4: right neighbor is out of bounds → 0
        nbhd = grid.get_neighborhood(4)
        np.testing.assert_array_equal(nbhd, [1, 0, 0])
    
    def test_neighborhood_radius_2_at_edge(self):
        """Test radius 2 neighborhood at edge."""
        grid = Grid1D(size=5, radius=2, boundary='zero',
                     initial_states=np.array([1, 1, 1, 1, 1]))
        
        # Position 1: neighbors at positions [-1, 0, 1, 2, 3]
        # Position -1 is out of bounds → 0, rest are 1
        nbhd = grid.get_neighborhood(1)
        np.testing.assert_array_equal(nbhd, [0, 1, 1, 1, 1])


class TestGrid1DNeighborhoodFixed:
    """Tests for neighborhood extraction with fixed boundary."""
    
    def test_neighborhood_center(self):
        """Test neighborhood in the center."""
        grid = Grid1D(size=5, radius=1, boundary='fixed',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        nbhd = grid.get_neighborhood(2)
        np.testing.assert_array_equal(nbhd, [1, 0, 1])
    
    def test_neighborhood_at_edges(self):
        """Test neighborhood at edges with fixed padding."""
        grid = Grid1D(size=5, radius=1, boundary='fixed',
                     initial_states=np.array([0, 1, 0, 1, 0]))
        
        # Fixed boundary uses 0 as the fixed value (same as zero boundary for now)
        nbhd_left = grid.get_neighborhood(0)
        np.testing.assert_array_equal(nbhd_left, [0, 0, 1])
        
        nbhd_right = grid.get_neighborhood(4)
        np.testing.assert_array_equal(nbhd_right, [1, 0, 0])


class TestGrid1DNeighborhoodInvalid:
    """Tests for invalid neighborhood queries."""
    
    def test_get_neighborhood_out_of_bounds(self):
        """Test that out of bounds position raises ValueError."""
        grid = Grid1D(size=5, radius=1)
        
        with pytest.raises(ValueError, match="position must be in"):
            grid.get_neighborhood(-1)
        
        with pytest.raises(ValueError, match="position must be in"):
            grid.get_neighborhood(5)
