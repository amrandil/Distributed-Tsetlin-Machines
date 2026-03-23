"""
Unit tests for RuleSystem.
"""

import pytest
import numpy as np
from cakit.grid import Grid1D
from cakit.automata import TsetlinAutomaton
from cakit.system import RuleSystem


class TestRuleSystemInitialization:
    """Tests for RuleSystem initialization."""
    
    def test_initialization_with_two_rules(self):
        """Test RuleSystem initialization with two rules."""
        grid = Grid1D(size=5, radius=1, boundary='periodic')
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3, 'initial_state': 1}
        )
        
        assert system.rules == [30, 110]
        assert len(system._rule_lookup_tables) == 2
        assert len(system.automata) == 5
    
    def test_invalid_rule_raises(self):
        """Test that invalid rule number raises ValueError."""
        grid = Grid1D(size=5, radius=1)
        
        with pytest.raises(ValueError, match="must be in"):
            RuleSystem(
                rules=[256],  # Out of range
                grid=grid,
                automaton_type=TsetlinAutomaton,
                automaton_params={'n_states': 3}
            )
        
        with pytest.raises(ValueError, match="must be in"):
            RuleSystem(
                rules=[-1],  # Out of range
                grid=grid,
                automaton_type=TsetlinAutomaton,
                automaton_params={'n_states': 3}
            )


class TestRuleSystemInterpret:
    """Tests for _interpret method."""
    
    def test_interpret_applies_correct_rule(self):
        """Test that _interpret applies the rule selected by output."""
        grid = Grid1D(size=5, radius=1)
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3}
        )
        
        # Rule 30 for neighborhood (1,1,0): 0
        # Rule 110 for neighborhood (1,1,0): 1
        neighborhood = np.array([1, 1, 0])
        
        assert system._interpret(0, neighborhood) == 0  # Rule 30
        assert system._interpret(1, neighborhood) == 1  # Rule 110
    
    def test_interpret_with_different_neighborhoods(self):
        """Test interpret with various neighborhoods."""
        grid = Grid1D(size=5, radius=1)
        system = RuleSystem(
            rules=[0, 255],  # All 0s vs all 1s
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3}
        )
        
        neighborhoods = [
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            np.array([1, 0, 1])
        ]
        
        for nbhd in neighborhoods:
            assert system._interpret(0, nbhd) == 0  # Rule 0 always returns 0
            assert system._interpret(1, nbhd) == 1  # Rule 255 always returns 1


class TestRuleSystemStep:
    """Tests for step method."""
    
    def test_step_with_fixed_automaton_output(self):
        """Test step with automata that output fixed values."""
        grid = Grid1D(
            size=5,
            radius=1,
            boundary='zero',
            initial_states=np.array([0, 0, 1, 0, 0])
        )
        
        # All TsetlinAutomata start in arm 0 (output 0), so Rule 30 applies
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3, 'initial_state': 1}  # State 1 -> arm 0
        )
        
        initial_state = grid.get_all_states().copy()
        system.step()
        new_state = grid.get_all_states()
        
        # State should have evolved according to Rule 30
        assert not np.array_equal(initial_state, new_state)
        assert system.generation == 1


class TestRuleSystemRun:
    """Tests for run method."""
    
    def test_run_records_history(self):
        """Test that run records grid and automaton history."""
        grid = Grid1D(size=5, radius=1, initial_states='single_center')
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3, 'initial_state': 'random'}
        )
        
        grid_history, automaton_history = system.run(generations=3)
        
        # History includes initial + 3 generations
        assert grid_history.shape == (4, 5)
        assert automaton_history.shape == (4, 5)
    
    def test_run_with_multiple_generations(self):
        """Test run with multiple generations."""
        grid = Grid1D(size=7, radius=1, boundary='periodic', initial_states='random')
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 5, 'initial_state': 'random'}
        )
        
        grid_history, _ = system.run(generations=10)
        
        assert grid_history.shape == (11, 7)
        assert system.generation == 10


class TestRuleSystemReset:
    """Tests for reset functionality."""
    
    def test_reset_restores_initial_state(self):
        """Test that reset restores initial states."""
        grid = Grid1D(size=5, radius=1, initial_states='single_center')
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3, 'initial_state': 2}
        )
        
        initial_grid = grid.get_all_states().copy()
        initial_automaton_states = [a.get_state() for a in system.automata.values()]
        
        system.run(generations=5)
        system.reset()
        
        assert system.generation == 0
        np.testing.assert_array_equal(grid.get_all_states(), initial_grid)
        
        reset_automaton_states = [a.get_state() for a in system.automata.values()]
        assert reset_automaton_states == initial_automaton_states


class TestRuleSystemWithFeedback:
    """Tests for RuleSystem with feedback function."""
    
    def test_feedback_fn_called(self):
        """Test that feedback function is called during step."""
        grid = Grid1D(size=5, radius=1, initial_states='random')
        
        feedback_calls = []
        
        def track_feedback(pos, grid_before, grid_after):
            feedback_calls.append(pos)
            return True
        
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 3, 'initial_state': 1},
            feedback_fn=track_feedback
        )
        
        system.step()
        
        assert len(feedback_calls) == 5
        assert sorted(feedback_calls) == [0, 1, 2, 3, 4]
    
    def test_with_feedback_and_learning(self):
        """Test RuleSystem with feedback and learning automata."""
        grid = Grid1D(size=5, radius=1, initial_states='random')
        
        def simple_feedback(pos, grid_before, grid_after):
            return grid_after[pos] == 1  # Reward if cell is alive
        
        system = RuleSystem(
            rules=[30, 110],
            grid=grid,
            automaton_type=TsetlinAutomaton,
            automaton_params={'n_states': 5, 'initial_state': 3},
            feedback_fn=simple_feedback
        )
        
        # Run for a few generations to ensure learning happens
        grid_history, automaton_history = system.run(generations=5)
        
        # Verify system runs without error
        assert grid_history.shape == (6, 5)
        assert automaton_history.shape == (6, 5)
        
        # Automaton states should have evolved (learning happened)
        initial_states = automaton_history[0]
        final_states = automaton_history[-1]
        # At least some automata should have changed state
        assert not np.array_equal(initial_states, final_states)
