"""
Unit tests for WolframAutomaton.
"""

import pytest
import numpy as np
from cakit.automata import WolframAutomaton


class TestWolframAutomatonInitialization:
    """Tests for WolframAutomaton initialization."""
    
    def test_valid_rule_initialization(self):
        """Test initialization with valid rule numbers."""
        wa = WolframAutomaton(rule=110)
        assert wa.rule == 110
        assert wa.get_state() == 110
    
    def test_rule_0(self):
        """Test initialization with rule 0."""
        wa = WolframAutomaton(rule=0)
        assert wa.rule == 0
    
    def test_rule_255(self):
        """Test initialization with rule 255."""
        wa = WolframAutomaton(rule=255)
        assert wa.rule == 255
    
    def test_invalid_rule_negative(self):
        """Test that negative rule raises ValueError."""
        with pytest.raises(ValueError, match="rule must be in"):
            WolframAutomaton(rule=-1)
    
    def test_invalid_rule_too_large(self):
        """Test that rule > 255 raises ValueError."""
        with pytest.raises(ValueError, match="rule must be in"):
            WolframAutomaton(rule=256)


class TestWolframAutomatonLookupTable:
    """Tests for lookup table construction."""
    
    def test_rule_0_all_dead(self):
        """Test Rule 0 (all outputs are 0)."""
        wa = WolframAutomaton(rule=0)
        neighborhoods = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        for nbhd in neighborhoods:
            assert wa.get_output(nbhd) == 0
    
    def test_rule_255_all_alive(self):
        """Test Rule 255 (all outputs are 1)."""
        wa = WolframAutomaton(rule=255)
        neighborhoods = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        for nbhd in neighborhoods:
            assert wa.get_output(nbhd) == 1
    
    def test_rule_110_specific_patterns(self):
        """Test Rule 110 outputs for specific neighborhoods."""
        wa = WolframAutomaton(rule=110)
        
        # Rule 110 binary: 01101110
        # 111→0, 110→1, 101→1, 100→0, 011→1, 010→1, 001→1, 000→0
        assert wa.get_output((1, 1, 1)) == 0
        assert wa.get_output((1, 1, 0)) == 1
        assert wa.get_output((1, 0, 1)) == 1
        assert wa.get_output((1, 0, 0)) == 0
        assert wa.get_output((0, 1, 1)) == 1
        assert wa.get_output((0, 1, 0)) == 1
        assert wa.get_output((0, 0, 1)) == 1
        assert wa.get_output((0, 0, 0)) == 0
    
    def test_rule_30_specific_patterns(self):
        """Test Rule 30 outputs for specific neighborhoods."""
        wa = WolframAutomaton(rule=30)
        
        # Rule 30 binary: 00011110
        # 111→0, 110→0, 101→0, 100→1, 011→1, 010→1, 001→1, 000→0
        assert wa.get_output((1, 1, 1)) == 0
        assert wa.get_output((1, 1, 0)) == 0
        assert wa.get_output((1, 0, 1)) == 0
        assert wa.get_output((1, 0, 0)) == 1
        assert wa.get_output((0, 1, 1)) == 1
        assert wa.get_output((0, 1, 0)) == 1
        assert wa.get_output((0, 0, 1)) == 1
        assert wa.get_output((0, 0, 0)) == 0


class TestWolframAutomatonGetOutput:
    """Tests for get_output method."""
    
    def test_get_output_with_list(self):
        """Test get_output with list input."""
        wa = WolframAutomaton(rule=110)
        output = wa.get_output([1, 1, 0])
        assert output == 1
    
    def test_get_output_with_tuple(self):
        """Test get_output with tuple input."""
        wa = WolframAutomaton(rule=110)
        output = wa.get_output((1, 1, 0))
        assert output == 1
    
    def test_get_output_with_numpy_array(self):
        """Test get_output with numpy array input."""
        wa = WolframAutomaton(rule=110)
        output = wa.get_output(np.array([1, 1, 0]))
        assert output == 1
    
    def test_get_output_none_input_raises(self):
        """Test that None input raises ValueError."""
        wa = WolframAutomaton(rule=110)
        with pytest.raises(ValueError, match="requires a neighborhood input"):
            wa.get_output(None)
    
    def test_get_output_wrong_length_raises(self):
        """Test that wrong neighborhood length raises ValueError."""
        wa = WolframAutomaton(rule=110)
        with pytest.raises(ValueError, match="expects 3-cell neighborhood"):
            wa.get_output([1, 1])
        
        with pytest.raises(ValueError, match="expects 3-cell neighborhood"):
            wa.get_output([1, 1, 0, 1])


class TestWolframAutomatonStateless:
    """Tests for stateless behavior."""
    
    def test_get_state_returns_rule(self):
        """Test that get_state returns the rule number."""
        wa = WolframAutomaton(rule=110)
        assert wa.get_state() == 110
    
    def test_reset_is_noop(self):
        """Test that reset doesn't change anything."""
        wa = WolframAutomaton(rule=110)
        initial_rule = wa.rule
        wa.reset()
        assert wa.rule == initial_rule
    
    def test_multiple_calls_same_result(self):
        """Test that multiple calls with same input give same result."""
        wa = WolframAutomaton(rule=110)
        neighborhood = (1, 0, 1)
        result1 = wa.get_output(neighborhood)
        result2 = wa.get_output(neighborhood)
        result3 = wa.get_output(neighborhood)
        assert result1 == result2 == result3


class TestWolframAutomatonDifferentRules:
    """Tests comparing different rules."""
    
    def test_different_rules_different_outputs(self):
        """Test that different rules produce different outputs for same input."""
        wa30 = WolframAutomaton(rule=30)
        wa110 = WolframAutomaton(rule=110)
        
        # For neighborhood (1, 1, 1): Rule 30 → 0, Rule 110 → 0 (same)
        # For neighborhood (1, 1, 0): Rule 30 → 0, Rule 110 → 1 (different)
        neighborhood = (1, 1, 0)
        assert wa30.get_output(neighborhood) == 0
        assert wa110.get_output(neighborhood) == 1
    
    def test_rule_identity(self):
        """Test that each rule maintains its identity."""
        rules = [0, 30, 110, 255]
        automata = [WolframAutomaton(rule=r) for r in rules]
        
        for wa, expected_rule in zip(automata, rules):
            assert wa.rule == expected_rule
            assert wa.get_state() == expected_rule
