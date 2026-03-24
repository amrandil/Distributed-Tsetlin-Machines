"""
Interactive GUI Experiment Launcher for cakit Framework

PyCX-style interactive viewer: start, pause, step, and reset the simulation
live, watching the space-time diagram grow row by row.

To run: python examples/gui_experiments.py
"""

import os
import numpy as np
import random

from cakit.visualization import InteractiveGUI
from cakit.feedback import (NeighbourhoodAgreementFeedback,
                             MinorityDisagreementFeedback,
                             GlobalTargetFeedback)
from cakit.system import StateSystem
from cakit.grid import Grid1D
from cakit.automata import WolframAutomaton, TsetlinAutomaton


# ============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# Choose experiment type: 'wolfram_ca' or 'tsetlin_cla'
EXPERIMENT_TYPE = 'tsetlin_cla'

# --- Wolfram CA Configuration ---
WOLFRAM_RULE = 30          # Try: 30, 110, 90, 184, etc.
WOLFRAM_GRID_SIZE = 201    # Odd number recommended for single_center
WOLFRAM_INITIAL = 'single_center'  # Options: 'single_center', 'random', 'half'

# --- Tsetlin CLA Configuration ---
TSETLIN_N_STATES = 6       # States per arm (try: 2, 4, 6, 10, 20)
TSETLIN_GRID_SIZE = 51     # Grid size
TSETLIN_FEEDBACK_RADIUS = 1
# Options: 'agreement' (consensus), 'minority' (diversity), 'global_target'
TSETLIN_FEEDBACK_TYPE = 'minority'
TSETLIN_TARGET_STATE = 1   # Target state for 'global_target' feedback (0 or 1)
TSETLIN_INITIAL = 'random' # Options: 'random', 'single_center', 'half'

# --- Plot type for Tsetlin CLA ---
# 'standard' = binary black/white
# 'augmented' = red/blue ramp showing TA confidence
TSETLIN_PLOT_TYPE = 'augmented'

# General settings
BOUNDARY = 'periodic'      # Options: 'periodic', 'zero', 'fixed'
MAX_GENERATIONS = 200      # Maximum steps the GUI will allow
SEED = 42                  # Set None for a different random run each time

# Animation speed: milliseconds between steps when running continuously.
# Lower = faster. Try 50 for fast, 200 for slow.
INTERVAL_MS = 80

# ============================================================================


def build_wolfram_system():
    """Construct and return a Wolfram CA system."""
    if SEED is not None:
        np.random.seed(SEED)
        random.seed(SEED)

    grid = Grid1D(
        size=WOLFRAM_GRID_SIZE,
        radius=1,
        boundary=BOUNDARY,
        initial_states=WOLFRAM_INITIAL,
    )
    system = StateSystem(
        grid=grid,
        automaton_type=WolframAutomaton,
        automaton_params={'rule': WOLFRAM_RULE},
        feedback_fn=None,
        seed=SEED,
    )
    params = {
        'Rule': WOLFRAM_RULE,
        'Grid': WOLFRAM_GRID_SIZE,
        'Initial': WOLFRAM_INITIAL,
        'Boundary': BOUNDARY,
        'Seed': SEED,
    }
    return system, None, 'standard', params


def build_tsetlin_system():
    """Construct and return a Tsetlin CLA system."""
    if SEED is not None:
        np.random.seed(SEED)
        random.seed(SEED)

    grid = Grid1D(
        size=TSETLIN_GRID_SIZE,
        radius=1,
        boundary=BOUNDARY,
        initial_states=TSETLIN_INITIAL,
    )

    if TSETLIN_FEEDBACK_TYPE == 'agreement':
        feedback_fn = NeighbourhoodAgreementFeedback(radius=TSETLIN_FEEDBACK_RADIUS)
    elif TSETLIN_FEEDBACK_TYPE == 'minority':
        feedback_fn = MinorityDisagreementFeedback(radius=TSETLIN_FEEDBACK_RADIUS)
    elif TSETLIN_FEEDBACK_TYPE == 'global_target':
        feedback_fn = GlobalTargetFeedback(target_state=TSETLIN_TARGET_STATE)
    else:
        raise ValueError(f"Unknown feedback type: '{TSETLIN_FEEDBACK_TYPE}'")

    system = StateSystem(
        grid=grid,
        automaton_type=TsetlinAutomaton,
        automaton_params={'n_states': TSETLIN_N_STATES, 'initial_state': 'random'},
        feedback_fn=feedback_fn,
        feedback_radius=TSETLIN_FEEDBACK_RADIUS,
        seed=SEED,
    )

    fb_info = (TSETLIN_TARGET_STATE if TSETLIN_FEEDBACK_TYPE == 'global_target'
               else TSETLIN_FEEDBACK_RADIUS)
    params = {
        'Grid': TSETLIN_GRID_SIZE,
        'N States': TSETLIN_N_STATES,
        'Initial': TSETLIN_INITIAL,
        'Feedback': TSETLIN_FEEDBACK_TYPE,
        'FB param': fb_info,
        'Boundary': BOUNDARY,
        'Seed': SEED,
    }
    return system, TSETLIN_N_STATES, TSETLIN_PLOT_TYPE, params


def main():
    if EXPERIMENT_TYPE == 'wolfram_ca':
        print(f"Wolfram CA  |  Rule {WOLFRAM_RULE}  |  Grid {WOLFRAM_GRID_SIZE}")
        system, n_states, plot_type, params = build_wolfram_system()
    elif EXPERIMENT_TYPE == 'tsetlin_cla':
        print(f"Tsetlin CLA  |  {TSETLIN_N_STATES} states  |  "
              f"Feedback: {TSETLIN_FEEDBACK_TYPE}  |  Grid {TSETLIN_GRID_SIZE}")
        system, n_states, plot_type, params = build_tsetlin_system()
    else:
        print(f"Unknown experiment type '{EXPERIMENT_TYPE}'")
        return

    print("Launching GUI — use Start/Pause, Step, and Reset to control the simulation.")

    gui = InteractiveGUI(
        system=system,
        max_generations=MAX_GENERATIONS,
        plot_type=plot_type,
        n_states=n_states,
        params=params,
        figsize=(16, 9),
        interval=INTERVAL_MS,
        view_window=80,
        save_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    gui.show()


if __name__ == '__main__':
    main()
