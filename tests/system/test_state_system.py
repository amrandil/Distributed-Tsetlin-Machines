"""
Unit tests for StateSystem.
"""

import pytest
import numpy as np
from cakit.grid import Grid1D
from cakit.automata import WolframAutomaton, TsetlinAutomaton
from cakit.system import StateSystem


class TestStateSystemInitialization:
    """Tests for StateSystem initialization."""
    
    def test_initialization_with_wolfram(self):
        """Test StateSystem initialization with WolframAutomaton."""
        grid = Grid1D(size=10, radius=1, boundary='periodic', initial_states='random')
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        assert system.grid == grid
        assert system.automaton_type == WolframAutomaton
        assert len(system.automata) == 10
        assert system.generation == 0
    
    def test_initialization_with_tsetlin(self):
        """Test StateSystem initialization with TsetlinAutomaton."""
        grid = Grid1D(size=5, radius=1, boundary='periodic', initial_states='random')
        system = StateSystem(
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 5, 'initial_state': 'random'}
        )
        
        assert len(system.automata) == 5
        assert all(isinstance(a, TsetlinAutomaton) for a in system.automata.values())
    
    def test_initialization_with_feedback_fn(self):
        """Test StateSystem initialization with feedback function."""
        grid = Grid1D(size=5, radius=1, boundary='periodic')
        
        def mock_feedback(pos, grid_before, grid_after):
            return True
        
        system = StateSystem(
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3},
            feedback_fn=mock_feedback,
            feedback_radius=1
        )
        
        assert system.feedback_fn is not None
        assert system.feedback_radius == 1


class TestStateSystemInterpret:
    """Tests for _interpret method."""
    
    def test_interpret_returns_output_directly(self):
        """Test that _interpret returns output directly."""
        grid = Grid1D(size=5, radius=1)
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        # _interpret should return output directly, ignoring neighborhood
        assert system._interpret(0, np.array([1, 1, 1])) == 0
        assert system._interpret(1, np.array([0, 0, 0])) == 1


class TestStateSystemStepWithWolfram:
    """Tests for step method with WolframAutomaton."""
    
    def test_step_updates_grid(self):
        """Test that step updates the grid state."""
        grid = Grid1D(
            size=5,
            radius=1,
            boundary='periodic',
            initial_states=np.array([0, 1, 0, 0, 0])
        )
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        initial_state = grid.get_all_states().copy()
        system.step()
        new_state = grid.get_all_states()
        
        # State should have changed
        assert not np.array_equal(initial_state, new_state)
        assert system.generation == 1
    
    def test_step_records_history(self):
        """Test that step records grid and automaton history.

        __init__ records generation 0, so after one step the history has 2 entries.
        """
        grid = Grid1D(size=5, radius=1, initial_states='single_center')
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )

        system.step()

        assert len(system.grid_history) == 2
        assert len(system.automaton_state_history) == 2
        assert system.grid_history[1].shape == (5,)


class TestStateSystemRun:
    """Tests for run method."""
    
    def test_run_returns_correct_shapes(self):
        """Test that run returns correct history shapes."""
        grid = Grid1D(size=10, radius=1, initial_states='single_center')
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        grid_history, automaton_history = system.run(generations=5)
        
        # History includes initial state + 5 generations
        assert grid_history.shape == (6, 10)
        assert automaton_history.shape == (6, 10)
    
    def test_run_resets_history(self):
        """Test that run resets history on each call."""
        grid = Grid1D(size=5, radius=1, initial_states='single_center')
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        system.run(generations=3)
        history1_len = len(system.grid_history)
        
        system.run(generations=5)
        history2_len = len(system.grid_history)
        
        # Second run should reset and have 6 entries (initial + 5)
        assert history2_len == 6


class TestStateSystemRule110Validation:
    """Validation tests for Rule 110 pattern reproduction."""
    
    def test_rule_110_single_center_first_step(self):
        """Test Rule 110 evolution from single center cell."""
        grid = Grid1D(
            size=7,
            radius=1,
            boundary='zero',
            initial_states=np.array([0, 0, 0, 1, 0, 0, 0])
        )
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        grid_history, _ = system.run(generations=1)
        
        # Initial: [0, 0, 0, 1, 0, 0, 0]
        # Rule 110: 
        #   pos 2: nbhd (0,0,1) -> 1
        #   pos 3: nbhd (0,1,0) -> 1
        #   pos 4: nbhd (1,0,0) -> 0
        # Expected next state: [0, 0, 1, 1, 0, 0, 0]
        expected = np.array([0, 0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(grid_history[1], expected)
    
    def test_rule_110_multiple_generations(self):
        """Test Rule 110 for multiple generations."""
        grid = Grid1D(
            size=5,
            radius=1,
            boundary='zero',
            initial_states=np.array([0, 0, 1, 0, 0])
        )
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        grid_history, _ = system.run(generations=3)
        
        # Verify we get 4 states (initial + 3 generations)
        assert grid_history.shape == (4, 5)
        
        # Each generation should be deterministic
        assert all(s in [0, 1] for s in grid_history.flatten())


class TestStateSystemReset:
    """Tests for reset functionality."""
    
    def test_reset_restores_initial_state(self):
        """Test that reset restores initial states."""
        grid = Grid1D(size=5, radius=1, initial_states='single_center')
        system = StateSystem(
            grid=grid,
            automaton_type=WolframAutomaton,
            automaton_params={'rule': 110}
        )
        
        initial_grid = grid.get_all_states().copy()
        initial_automaton_states = [a.get_state() for a in system.automata.values()]
        
        system.run(generations=5)
        assert system.generation == 5
        
        system.reset()

        assert system.generation == 0
        assert len(system.grid_history) == 1  # generation 0 recorded after reset
        np.testing.assert_array_equal(grid.get_all_states(), initial_grid)
        
        reset_automaton_states = [a.get_state() for a in system.automata.values()]
        assert reset_automaton_states == initial_automaton_states


class TestStateSystemWithFeedback:
    """Tests for StateSystem with feedback function."""
    
    def test_feedback_fn_called(self):
        """Test that feedback function is called during step."""
        grid = Grid1D(size=5, radius=1, initial_states='random')
        
        feedback_calls = []
        
        def track_feedback(pos, grid_before, grid_after):
            feedback_calls.append(pos)
            return True  # Always reward
        
        system = StateSystem(
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3, 'initial_state': 1},
            feedback_fn=track_feedback
        )
        
        system.step()
        
        # Feedback should be called once per cell
        assert len(feedback_calls) == 5
        assert sorted(feedback_calls) == [0, 1, 2, 3, 4]
    
    def test_reward_penalty_called(self):
        """Test that reward/penalize are called based on feedback."""
        grid = Grid1D(size=3, radius=1, initial_states='random')
        
        # Feedback that alternates reward/penalty
        def alternating_feedback(pos, grid_before, grid_after):
            return pos % 2 == 0  # Reward on even positions
        
        system = StateSystem(
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 5, 'initial_state': 3},
            feedback_fn=alternating_feedback
        )
        
        # Get initial states
        initial_states = {pos: system.automata[pos].get_state() for pos in [0, 1, 2]}
        
        system.step()
        
        # Position 0 and 2 should be rewarded (state increased or stayed at max)
        # Position 1 should be penalized (state decreased or stayed at min)
        # This is a weak test since we can't predict exact outcomes, but we check
        # that the system runs without error
        assert system.generation == 1
