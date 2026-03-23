"""
Interactive GUI — Wolfram Rule 30

Rule 30 is a chaotic elementary CA whose center column is used by Mathematica
as a pseudo-random number generator.  Starting from a single live cell it
produces highly irregular, seemingly random patterns.

To run: python examples/gui_rule30.py
"""

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
    automaton_params={'rule': 30},
    feedback_fn=None,
    seed=SEED,
)

gui = InteractiveGUI(
    system=system,
    max_generations=MAX_GENERATIONS,
    plot_type='standard',
    params={
        'Rule': 30,
        'Grid': GRID_SIZE,
        'Initial': 'single_center',
        'Boundary': 'periodic',
    },
    figsize=(16, 9),
    interval=60,
    view_window=80,
)

print("Rule 30 — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
