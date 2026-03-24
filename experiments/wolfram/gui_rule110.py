"""
Interactive GUI — Wolfram Rule 110

Rule 110 is a Turing-complete elementary CA known for its complex, irregular
patterns that never fully settle. Start with a single live cell in the center
and watch it grow.

To run: python examples/gui_rule110.py
"""

import os
import numpy as np
import random

from cakit.visualization import InteractiveGUI
from cakit.system import StateSystem
from cakit.grid import Grid1D
from cakit.automata import WolframAutomaton

GRID_SIZE = 201
MAX_GENERATIONS = 400
SEED = 42

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

grid = Grid1D(
    size=GRID_SIZE,
    radius=1,
    boundary='periodic',
    initial_states='single_center',
)

system = StateSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params={'rule': 110},
    feedback_fn=None,
    seed=SEED,
)

gui = InteractiveGUI(
    system=system,
    max_generations=MAX_GENERATIONS,
    plot_type='standard',
    params={
        'Rule': 110,
        'Grid': GRID_SIZE,
        'Initial': 'single_center',
        'Boundary': 'periodic',
    },
    figsize=(16, 9),
    interval=60,
    view_window=80,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
)

print("Rule 110 — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
