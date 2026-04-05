"""
Elementary CA — Wolfram Rule 30  (Interactive GUI)

STRUCTURE
---------
This is a pure Cellular Automaton (CA): cells have binary states and evolve
by a fixed, hand-coded rule.  There is no learning — every cell applies the
same deterministic lookup table at every step.

COMPOSITION
-----------
  Grid1D          — 1-D array of binary cell states; owns boundary handling
                    and neighbourhood slicing.
  WolframAutomaton — stateless rule object; maps a 3-cell neighbourhood
                    (left, centre, right) to the next centre state using the
                    Wolfram encoding of an 8-bit rule number.
  StateSystem     — simulation loop: calls the automaton for every cell each
                    generation; feedback_fn=None because CAs need no reward.
  InteractiveGUI  — Tk/matplotlib window with Start/Pause/Step/Reset/Save
                    controls; plot_type='standard' gives a B&W space–time
                    diagram (no colour augmentation needed for plain CAs).

RULE 30
-------
Rule 30 is Class III (chaotic).  Its centre-column output passes standard
randomness tests and is the basis of Mathematica's default PRNG.

To run:
    python experiments/wolfram/wolfram_rule30.py
"""

import os
import numpy as np
import random

from cakit.visualization import InteractiveGUI
from cakit.system import StateSystem
from cakit.grid import Grid1D
from cakit.automata import WolframAutomaton

GRID_SIZE = 201       # number of cells; try 11 or 51 for narrower/wider patterns
MAX_GENERATIONS = 400  # rows in the space–time diagram
SEED = 42             # set to None for a non-reproducible run

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

# --- Grid -----------------------------------------------------------------
# radius=1   → each cell sees 3 neighbours (the standard Wolfram neighbourhood)
# boundary   → 'periodic' wraps edges; try 'fixed' (dead boundary) or
#              'reflecting' (mirror boundary) for different edge behaviour
# initial_states → 'single_center' (one live cell) is the canonical Wolfram
#                  starting condition; 'random' gives a noisy space–time plot
grid = Grid1D(
    size=GRID_SIZE,
    radius=1,
    boundary='periodic',
    initial_states='single_center',
)

# --- System ---------------------------------------------------------------
# WolframAutomaton accepts any rule number 0–255 (8-bit table).
#   Rule 30  → chaotic / pseudo-random
#   Rule 90  → Sierpiński triangle (XOR rule)
#   Rule 110 → Turing-complete, complex but ordered
#   Rule 184 → traffic-flow model
# feedback_fn=None because plain CAs are deterministic; no learning occurs.
system = StateSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params={'rule': 30},
    feedback_fn=None,
    seed=SEED,
)

# --- GUI ------------------------------------------------------------------
# plot_type='standard' → greyscale space–time diagram (white=0, black=1).
# interval  → milliseconds between animation frames; lower = faster playback.
# view_window → number of most-recent generations shown in the scrolling view.
gui = InteractiveGUI(
    system=system,
    max_generations=MAX_GENERATIONS,
    plot_type='standard',
    params={
        'Rule': 30,
        'Grid': GRID_SIZE,
        'Initial grid': 'single_center',
        'Boundary': 'periodic',
    },
    figsize=(16, 9),
    interval=60,
    view_window=80,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
)

print("Rule 30 — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
