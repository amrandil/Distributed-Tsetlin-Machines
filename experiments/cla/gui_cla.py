"""
Interactive GUI — Binary CLA with Tsetlin Automata

Each cell hosts a Tsetlin Automaton learning to synchronise with its
neighbours via Neighbourhood Agreement feedback (majority-vote reward).
Starting from a random initial state the grid typically converges toward
consensus stripes over ~100 generations.

To run: python examples/gui_cla.py
"""

import os
import numpy as np
import random

from cakit.visualization import InteractiveGUI
from cakit.system import StateSystem
from cakit.grid import Grid1D
from cakit.automata import TsetlinAutomaton
from cakit.feedback import NeighbourhoodAgreementFeedback
from cakit.feedback import MinorityDisagreementFeedback
GRID_SIZE = 51
N_STATES = 10
MAX_GENERATIONS = 200
SEED = 42

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

grid = Grid1D(
    size=GRID_SIZE,
    radius=1,
    boundary='periodic',
    initial_states='random',
)


# feedback_fn = NeighbourhoodAgreementFeedback(radius=1)
feedback_fn = MinorityDisagreementFeedback(radius=1)
system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': N_STATES, 'initial_state': 'random'},
    feedback_fn=feedback_fn,
    feedback_radius=1,
    seed=SEED,
)

gui = InteractiveGUI(
    system=system,
    max_generations=MAX_GENERATIONS,
    plot_type='augmented',
    n_states=N_STATES,
    params={
        'Grid': GRID_SIZE,
        'N States': N_STATES,
        'Feedback': 'agreement',
        'FB Radius': 1,
        'Boundary': 'periodic',
        'Initial': 'random',
        'Seed': SEED,
    },
    figsize=(16, 9),
    interval=80,
    view_window=80,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
)

print("Binary CLA — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
