"""
Unit tests for TsetlinAutomaton.
"""

import pytest
from cakit.automata import TsetlinAutomaton


class TestTsetlinAutomatonInitialization:
    """Tests for TsetlinAutomaton initialization."""
    
    def test_default_initialization(self):
        """Test default initialization with n_states=5 and random initial state."""
        ta = TsetlinAutomaton()
        assert ta.n_states == 5
        assert ta.total_states == 10
        assert 1 <= ta.get_state() <= 10
    
    def test_custom_n_states(self):
        """Test initialization with custom n_states."""
        ta = TsetlinAutomaton(n_states=3)
        assert ta.n_states == 3
        assert ta.total_states == 6
    
    def test_specific_initial_state(self):
        """Test initialization with a specific initial state."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        assert ta.get_state() == 3
    
    def test_invalid_n_states(self):
        """Test that n_states < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_states must be at least 1"):
            TsetlinAutomaton(n_states=0)
    
    def test_invalid_initial_state(self):
        """Test that out-of-range initial_state raises ValueError."""
        with pytest.raises(ValueError, match="initial_state must be in"):
            TsetlinAutomaton(n_states=5, initial_state=11)
        
        with pytest.raises(ValueError, match="initial_state must be in"):
            TsetlinAutomaton(n_states=5, initial_state=0)
    
    def test_invalid_initial_state_type(self):
        """Test that invalid type for initial_state raises ValueError."""
        with pytest.raises(ValueError, match="initial_state must be an integer"):
            TsetlinAutomaton(initial_state="invalid")


class TestTsetlinAutomatonArmAndPosition:
    """Tests for arm and position calculation."""
    
    def test_arm_0_states(self):
        """Test that states 1..N are in arm 0."""
        ta = TsetlinAutomaton(n_states=5, initial_state=1)
        assert ta.get_arm() == 0
        
        ta._state = 3
        assert ta.get_arm() == 0
        
        ta._state = 5
        assert ta.get_arm() == 0
    
    def test_arm_1_states(self):
        """Test that states N+1..2N are in arm 1."""
        ta = TsetlinAutomaton(n_states=5, initial_state=6)
        assert ta.get_arm() == 1
        
        ta._state = 8
        assert ta.get_arm() == 1
        
        ta._state = 10
        assert ta.get_arm() == 1
    
    def test_position_in_arm_0(self):
        """Test position calculation for arm 0."""
        ta = TsetlinAutomaton(n_states=5, initial_state=1)
        assert ta.get_position() == 0  # Boundary
        
        ta._state = 3
        assert ta.get_position() == 2  # Mid-arm
        
        ta._state = 5
        assert ta.get_position() == 4  # Extreme (N-1)
    
    def test_position_in_arm_1(self):
        """Test position calculation for arm 1."""
        ta = TsetlinAutomaton(n_states=5, initial_state=6)
        assert ta.get_position() == 0  # Boundary
        
        ta._state = 8
        assert ta.get_position() == 2  # Mid-arm
        
        ta._state = 10
        assert ta.get_position() == 4  # Extreme (N-1)


class TestTsetlinAutomatonOutput:
    """Tests for get_output method."""
    
    def test_get_output_arm_0(self):
        """Test that get_output returns 0 for arm 0."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        assert ta.get_output() == 0
    
    def test_get_output_arm_1(self):
        """Test that get_output returns 1 for arm 1."""
        ta = TsetlinAutomaton(n_states=5, initial_state=8)
        assert ta.get_output() == 1
    
    def test_get_output_ignores_input(self):
        """Test that get_output ignores its input parameter."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        assert ta.get_output() == ta.get_output([1, 0, 1])


class TestTsetlinAutomatonReward:
    """Tests for reward transition."""
    
    def test_reward_moves_deeper_in_arm_0(self):
        """Test that reward moves deeper into arm 0."""
        ta = TsetlinAutomaton(n_states=5, initial_state=2)
        initial_state = ta.get_state()
        ta.reward()
        assert ta.get_state() == initial_state + 1
        assert ta.get_arm() == 0
    
    def test_reward_moves_deeper_in_arm_1(self):
        """Test that reward moves deeper into arm 1."""
        ta = TsetlinAutomaton(n_states=5, initial_state=7)
        initial_state = ta.get_state()
        ta.reward()
        assert ta.get_state() == initial_state + 1
        assert ta.get_arm() == 1
    
    def test_reward_at_extreme_stays(self):
        """Test that reward at extreme end doesn't move further."""
        ta = TsetlinAutomaton(n_states=5, initial_state=5)
        ta.reward()
        assert ta.get_state() == 5
        
        ta._state = 10
        ta.reward()
        assert ta.get_state() == 10
    
    def test_multiple_rewards(self):
        """Test multiple consecutive rewards."""
        ta = TsetlinAutomaton(n_states=5, initial_state=1)
        for i in range(4):
            ta.reward()
        assert ta.get_state() == 5
        assert ta.get_position() == 4


class TestTsetlinAutomatonPenalty:
    """Tests for penalty transition."""
    
    def test_penalty_moves_toward_boundary_in_arm_0(self):
        """Test that penalty moves toward boundary in arm 0."""
        ta = TsetlinAutomaton(n_states=5, initial_state=4)
        initial_state = ta.get_state()
        ta.penalize()
        assert ta.get_state() == initial_state - 1
        assert ta.get_arm() == 0
    
    def test_penalty_moves_toward_boundary_in_arm_1(self):
        """Test that penalty moves toward boundary in arm 1."""
        ta = TsetlinAutomaton(n_states=5, initial_state=9)
        initial_state = ta.get_state()
        ta.penalize()
        assert ta.get_state() == initial_state - 1
        assert ta.get_arm() == 1
    
    def test_penalty_at_boundary_switches_arms(self):
        """Test that penalty at boundary switches to the opposite arm's boundary."""
        # Arm 0 boundary (state 1) → arm 1 boundary (state N+1 = 6)
        ta = TsetlinAutomaton(n_states=5, initial_state=1)
        assert ta.get_arm() == 0
        ta.penalize()
        assert ta.get_state() == 6
        assert ta.get_arm() == 1
        assert ta.get_position() == 0

        # Arm 1 boundary (state 6) → arm 0 boundary (state 1)
        ta.penalize()
        assert ta.get_state() == 1
        assert ta.get_arm() == 0
        assert ta.get_position() == 0
    
    def test_multiple_penalties(self):
        """Test multiple consecutive penalties."""
        ta = TsetlinAutomaton(n_states=5, initial_state=5)
        for i in range(4):
            ta.penalize()
        assert ta.get_state() == 1
        assert ta.get_position() == 0


class TestTsetlinAutomatonRewardPenaltySequence:
    """Tests for sequences of rewards and penalties."""
    
    def test_reward_then_penalty(self):
        """Test reward followed by penalty."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        ta.reward()
        assert ta.get_state() == 4
        ta.penalize()
        assert ta.get_state() == 3
    
    def test_penalty_then_reward(self):
        """Test penalty followed by reward."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        ta.penalize()
        assert ta.get_state() == 2
        ta.reward()
        assert ta.get_state() == 3
    
    def test_alternating_reward_penalty(self):
        """Test alternating rewards and penalties."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        initial_state = ta.get_state()
        ta.reward()
        ta.penalize()
        ta.reward()
        ta.penalize()
        assert ta.get_state() == initial_state


class TestTsetlinAutomatonReset:
    """Tests for reset functionality."""
    
    def test_reset_to_initial_state(self):
        """Test that reset restores the initial state."""
        ta = TsetlinAutomaton(n_states=5, initial_state=3)
        initial = ta.get_state()
        
        ta.reward()
        ta.reward()
        assert ta.get_state() != initial
        
        ta.reset()
        assert ta.get_state() == initial
    
    def test_reset_after_random_initialization(self):
        """Test that reset works correctly with random initialization."""
        ta = TsetlinAutomaton(n_states=5, initial_state='random')
        initial = ta.get_state()
        
        for _ in range(10):
            ta.reward()
        
        ta.reset()
        assert ta.get_state() == initial
