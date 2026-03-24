"""
Debug script to investigate why the grid freezes after one iteration.
"""

import numpy as np
import random
from cakit.grid import Grid1D
from cakit.system import StateSystem
from cakit.automata import TsetlinAutomaton
from cakit.feedback import (NeighbourhoodAgreementFeedback, 
                            MinorityDisagreementFeedback,
                            GlobalTargetFeedback)

# Configuration
GRID_SIZE = 11  # Smaller for easier debugging
N_STATES = 4
RADIUS = 1
SEED = 42
GENERATIONS = 5
FEEDBACK_TYPE = 'global_target'  # Options: 'agreement', 'minority', 'global_target'
TARGET_STATE = 1  # For global_target feedback

# Set seed
if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

# Create grid
grid = Grid1D(
    size=GRID_SIZE,
    radius=RADIUS,
    boundary='periodic',
    initial_states='random'
)

# Create feedback
if FEEDBACK_TYPE == 'agreement':
    feedback_fn = NeighbourhoodAgreementFeedback(radius=RADIUS)
elif FEEDBACK_TYPE == 'minority':
    feedback_fn = MinorityDisagreementFeedback(radius=RADIUS)
elif FEEDBACK_TYPE == 'global_target':
    feedback_fn = GlobalTargetFeedback(target_state=TARGET_STATE)
else:
    raise ValueError(f"Unknown feedback type: '{FEEDBACK_TYPE}'")

# Create system (pass seed=None since we already set it)
system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': N_STATES, 'initial_state': 'random'},
    feedback_fn=feedback_fn,
    feedback_radius=RADIUS,
    seed=None  # Already set above
)

print("=" * 80)
print(f"DEBUGGING FREEZE BEHAVIOR - Feedback: {FEEDBACK_TYPE.upper()}")
print("=" * 80)

# Initial state
print(f"\nGeneration 0 (Initial):")
print(f"  Grid:        {grid.get_all_states()}")
ta_states = [system.automata[i].get_state() for i in range(GRID_SIZE)]
ta_arms = [system.automata[i].get_arm() for i in range(GRID_SIZE)]
print(f"  TA States:   {ta_states}")
print(f"  TA Arms:     {ta_arms}")

# Run generations with detailed output
for gen in range(1, GENERATIONS + 1):
    # Save state before step
    grid_before = grid.get_all_states().copy()
    ta_states_before = [system.automata[i].get_state() for i in range(GRID_SIZE)]
    
    # Step
    system.step()
    
    # Get state after step
    grid_after = grid.get_all_states()
    ta_states_after = [system.automata[i].get_state() for i in range(GRID_SIZE)]
    ta_arms_after = [system.automata[i].get_arm() for i in range(GRID_SIZE)]
    
    # Check for changes
    grid_changed = not np.array_equal(grid_before, grid_after)
    ta_changed = ta_states_before != ta_states_after
    
    print(f"\nGeneration {gen}:")
    print(f"  Grid:        {grid_after}")
    print(f"  Grid changed: {grid_changed}")
    print(f"  TA States:   {ta_states_after}")
    print(f"  TA Arms:     {ta_arms_after}")
    
    # Detailed feedback analysis for first few cells
    if gen <= 2:
        print(f"\n  Detailed Feedback Analysis:")
        for i in range(min(5, GRID_SIZE)):
            neighborhood = system.grid.get_neighborhood(i)
            cell_state = grid_after[i]
            
            # Calculate what feedback would be
            feedback_neighborhood = []
            for offset in range(-RADIUS, RADIUS + 1):
                pos = (i + offset) % GRID_SIZE
                feedback_neighborhood.append(grid_after[pos])
            
            if FEEDBACK_TYPE == 'global_target':
                reward = (cell_state == TARGET_STATE)
                feedback_info = f"target={TARGET_STATE}"
            else:
                majority = 1 if sum(feedback_neighborhood) >= len(feedback_neighborhood) / 2 else 0
                if FEEDBACK_TYPE == 'agreement':
                    reward = (cell_state == majority)
                else:
                    reward = (cell_state != majority)
                feedback_info = f"maj={majority}"
            
            ta_change = "+" if ta_states_after[i] > ta_states_before[i] else ("-" if ta_states_after[i] < ta_states_before[i] else "=")
            
            print(f"    Cell {i}: state={cell_state}, nbhd={feedback_neighborhood}, {feedback_info}, "
                  f"reward={reward}, TA: {ta_states_before[i]}->{ta_states_after[i]} ({ta_change})")
    
    # Check if frozen
    if not grid_changed:
        print(f"\n  ⚠️  GRID FROZEN at generation {gen}")
        break

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)
