"""
Experimentation Script for cakit Framework

This script makes it easy to run different CA and CLA experiments by changing
a few parameters at the top. Use this as a template for your own experiments.

To run: python examples/experiments.py
"""

# Set matplotlib backend before any imports
from cakit.visualization import SpaceTimePlot
from cakit.feedback import (NeighbourhoodAgreementFeedback,
                            MinorityDisagreementFeedback,
                            GlobalTargetFeedback)
from cakit.system import StateSystem
from cakit.grid import Grid1D
from cakit.automata import WolframAutomaton, TsetlinAutomaton
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
matplotlib.use('Agg')


# ============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# Choose experiment type: 'wolfram_ca' or 'tsetlin_cla'
EXPERIMENT_TYPE = 'tsetlin_cla'

# --- Wolfram CA Configuration ---
WOLFRAM_RULE = 30  # Try: 30, 110, 90, 184, etc.
WOLFRAM_ITERATIONS = 100  # Number of generations
WOLFRAM_GRID_SIZE = 201  # Odd number recommended for single_center
# Options: 'single_center', 'random', 'alternating'
WOLFRAM_INITIAL = 'single_center'

# --- Tsetlin CLA Configuration ---
TSETLIN_N_STATES = 3  # States per arm (try: 2, 4, 6, 10, 20)
TSETLIN_ITERATIONS = 100  # Number of generations
TSETLIN_GRID_SIZE = 51  # Grid size

# Neighborhood radius for feedback (not used by 'global_target')
TSETLIN_FEEDBACK_RADIUS = 1
# Options: 'agreement' (consensus), 'minority' (diversity), 'global_target' (all cells -> 1)
TSETLIN_FEEDBACK_TYPE = 'minority'
TSETLIN_TARGET_STATE = 1  # Target state for 'global_target' feedback (0 or 1)

# General settings
BOUNDARY = 'periodic'  # Options: 'periodic', 'zero', 'fixed'
# Options: 'random', 'single_center', 'alternating', 'all_ones', 'all_zeros'
TSETLIN_INITIAL = 'random'
# Random seed for full reproducibility (controls grid + TA init). Set None for random.
SEED = 42

# ============================================================================


def run_wolfram_ca():
    """Run a Wolfram elementary CA experiment."""
    print("=" * 70)
    print(f"Wolfram CA Experiment: Rule {WOLFRAM_RULE}")
    print("=" * 70)

    # Set seed before any randomness
    if SEED is not None:
        np.random.seed(SEED)
        random.seed(SEED)

    # Create grid
    grid = Grid1D(
        size=WOLFRAM_GRID_SIZE,
        radius=1,
        boundary=BOUNDARY,
        initial_states=WOLFRAM_INITIAL
    )
    print(f"\nGrid Configuration:")
    print(f"  Size:           {grid.size} cells")
    print(f"  Radius:         {grid.radius}")
    print(f"  Boundary:       '{grid.boundary}'")
    print(f"  Initial state:  '{WOLFRAM_INITIAL}'")

    # Create system
    system = StateSystem(
        grid=grid,
        automaton_type=WolframAutomaton,
        automaton_params={'rule': WOLFRAM_RULE},
        feedback_fn=None,  # No learning for pure CA
        seed=SEED
    )
    print(f"\nSystem Configuration:")
    print(f"  Automaton:      WolframAutomaton(rule={WOLFRAM_RULE})")
    print(f"  Feedback:       None (pure CA)")
    print(f"  Generations:    {WOLFRAM_ITERATIONS}")
    print(f"  Seed:           {SEED}")

    # Run simulation
    print(f"\nRunning simulation...")
    grid_history, _ = system.run(generations=WOLFRAM_ITERATIONS)
    print(f"✓ Completed! History shape: {grid_history.shape}")

    # Visualize
    print("\nGenerating visualization...")
    plotter = SpaceTimePlot(
        grid_history=grid_history,
        figsize=(16, 10)
    )

    # Prepare parameters for legend
    params = {
        'Rule': WOLFRAM_RULE,
        'Grid Size': WOLFRAM_GRID_SIZE,
        'Initial': WOLFRAM_INITIAL,
        'Boundary': BOUNDARY,
        'Generations': WOLFRAM_ITERATIONS,
        'Seed': SEED
    }

    save_path = f"wolfram_rule{WOLFRAM_RULE}_{WOLFRAM_INITIAL}_{WOLFRAM_ITERATIONS}gen.png"
    plotter.plot_standard(save_path=save_path, show=False, params=params)
    print(f"✓ Saved to: {save_path}")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


def run_tsetlin_cla():
    """Run a Tsetlin CLA experiment."""
    print("=" * 70)
    print(f"Tsetlin CLA Experiment: {TSETLIN_N_STATES} states per arm")
    print("=" * 70)

    # Set seed before any randomness
    if SEED is not None:
        np.random.seed(SEED)
        random.seed(SEED)

    # Create grid
    grid = Grid1D(
        size=TSETLIN_GRID_SIZE,
        radius=1,
        boundary=BOUNDARY,
        initial_states=TSETLIN_INITIAL
    )
    print(f"\nGrid Configuration:")
    print(f"  Size:           {grid.size} cells")
    print(f"  Radius:         {grid.radius}")
    print(f"  Boundary:       '{grid.boundary}'")
    print(f"  Initial state:  '{TSETLIN_INITIAL}'")

    # Create feedback function
    if TSETLIN_FEEDBACK_TYPE == 'agreement':
        feedback_fn = NeighbourhoodAgreementFeedback(
            radius=TSETLIN_FEEDBACK_RADIUS)
        feedback_desc = "Rewards majority agreement (convergent)"
        feedback_info = f"Radius: {TSETLIN_FEEDBACK_RADIUS}"
    elif TSETLIN_FEEDBACK_TYPE == 'minority':
        feedback_fn = MinorityDisagreementFeedback(
            radius=TSETLIN_FEEDBACK_RADIUS)
        feedback_desc = "Rewards minority disagreement (diversity)"
        feedback_info = f"Radius: {TSETLIN_FEEDBACK_RADIUS}"
    elif TSETLIN_FEEDBACK_TYPE == 'global_target':
        feedback_fn = GlobalTargetFeedback(target_state=TSETLIN_TARGET_STATE)
        feedback_desc = f"Rewards cells matching global target: {TSETLIN_TARGET_STATE}"
        feedback_info = f"Target state: {TSETLIN_TARGET_STATE}"
    else:
        raise ValueError(f"Unknown feedback type: '{TSETLIN_FEEDBACK_TYPE}'")

    print(f"\nFeedback Configuration:")
    print(f"  Type:           {TSETLIN_FEEDBACK_TYPE}")
    print(f"  Info:           {feedback_info}")
    print(f"  Behavior:       {feedback_desc}")

    # Create system
    total_states = TSETLIN_N_STATES * 2  # n_states per arm
    system = StateSystem(
        grid=grid,
        automaton_type=TsetlinAutomaton,
        automaton_params={'n_states': TSETLIN_N_STATES,
                          'initial_state': 'random'},
        feedback_fn=feedback_fn,
        feedback_radius=TSETLIN_FEEDBACK_RADIUS,
        seed=SEED
    )
    print(f"\nSystem Configuration:")
    print(f"  Automaton:      TsetlinAutomaton(n_states={TSETLIN_N_STATES})")
    print(f"  Total states:   {total_states} ({TSETLIN_N_STATES} per arm)")
    print(f"  Generations:    {TSETLIN_ITERATIONS}")
    print(f"  Seed:           {SEED}")

    # Run simulation
    print(f"\nRunning simulation...")
    grid_history, ta_history = system.run(generations=TSETLIN_ITERATIONS)
    print(f"✓ Completed!")
    print(f"  Grid history shape: {grid_history.shape}")
    print(f"  TA history shape:   {ta_history.shape}")

    # Visualize
    print("\nGenerating visualizations...")
    plotter = SpaceTimePlot(
        grid_history=grid_history,
        ta_state_history=ta_history,
        n_states=TSETLIN_N_STATES,
        figsize=(16, 10)
    )

    # Prepare parameters for legend
    params = {
        'Grid Size': TSETLIN_GRID_SIZE,
        'N States': TSETLIN_N_STATES,
        'Initial': TSETLIN_INITIAL,
        'Feedback': TSETLIN_FEEDBACK_TYPE,
        'FB Radius': TSETLIN_FEEDBACK_RADIUS if TSETLIN_FEEDBACK_TYPE != 'global_target' else 'N/A',
        'Target': TSETLIN_TARGET_STATE if TSETLIN_FEEDBACK_TYPE == 'global_target' else 'N/A',
        'Boundary': BOUNDARY,
        'Generations': TSETLIN_ITERATIONS,
        'Seed': SEED
    }

    # Standard plot
    save_path_std = f"tsetlin_{TSETLIN_N_STATES}states_{TSETLIN_INITIAL}_{TSETLIN_FEEDBACK_TYPE}_{TSETLIN_ITERATIONS}gen_standard.png"
    plotter.plot_standard(save_path=save_path_std, show=False, params=params)
    print(f"✓ Saved standard plot to: {save_path_std}")

    # TA-augmented plot (shows confidence)
    save_path_aug = f"tsetlin_{TSETLIN_N_STATES}states_{TSETLIN_INITIAL}_{TSETLIN_FEEDBACK_TYPE}_{TSETLIN_ITERATIONS}gen_augmented.png"
    plotter.plot_ta_augmented(save_path=save_path_aug,
                              show=False, params=params)
    print(f"✓ Saved augmented plot to: {save_path_aug}")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


def main():
    """Run the configured experiment."""
    if EXPERIMENT_TYPE == 'wolfram_ca':
        run_wolfram_ca()
    elif EXPERIMENT_TYPE == 'tsetlin_cla':
        run_tsetlin_cla()
    else:
        print(f"Error: Unknown experiment type '{EXPERIMENT_TYPE}'")
        print("Valid options: 'wolfram_ca' or 'tsetlin_cla'")


if __name__ == "__main__":
    main()
