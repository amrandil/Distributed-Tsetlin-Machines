"""
Cellular Learning Automaton — Binary CLA with Tsetlin Automata  (Interactive GUI)

STRUCTURE
---------
This is a Cellular Learning Automaton (CLA): unlike a plain CA, each cell
hosts an independent Learning Automaton (LA) that adapts its output based on
feedback from its neighbourhood.  The rule is not fixed — it is learned online
as the system runs.

COMPOSITION
-----------
  Grid1D                      — 1-D array of cell states; owns boundary
                                handling and neighbourhood slicing.
  TsetlinAutomaton            — the per-cell LA.  It has two arms (output 0
                                and output 1), each with n_states positions.
                                Position 0 is the boundary (least confident);
                                position n_states-1 is deepest (most confident).
                                The cell's output is the arm it currently sits
                                on.  Reward pushes deeper into the current arm;
                                penalty moves toward the boundary and may flip
                                to the opposite arm.
  feedback_fn                 — evaluates each cell's neighbourhood after every
                                step and emits reward/penalty signals, that is fed to the learning automaton (TsetlinAutomaton in this experiment).
  StateSystem                 — simulation loop: steps every cell's automaton,
                                then evaluates feedback for every cell and
                                applies reward/penalty.
  InteractiveGUI              — Tk/matplotlib window with Start/Pause/Step/
                                Reset/Save controls.  plot_type='augmented'
                                enables the colour view (red = State 1,
                                blue = State 0; saturation encodes confidence)
                                and the B&W toggle button.

FEEDBACK OPTIONS
----------------
  NeighbourhoodAgreementFeedback(radius)
      Reward a cell whose output matches the majority of its neighbourhood
      (radius cells on each side).  Drives the grid toward consensus.

  MinorityDisagreementFeedback(radius)   ← active in this experiment
      Reward a cell that disagrees with the majority of its neighbourhood.
      Drives the grid toward a minority / anti-coordination pattern, which
      can produce stable spatial diversity rather than full consensus.

To run:
    python experiments/cla/tsetlin_cla/tsetlin_cla.py
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
from cakit.feedback import GlobalTargetFeedback

GRID_SIZE = 51        # number of cells; larger grids show richer spatial patterns
N_STATES = 10          # confidence resolution per TA arm; higher = slower learning,
# finer colour gradients in the augmented view
MAX_GENERATIONS = 200  # rows in the space–time diagram
SEED = 42              # set to None for a non-reproducible run

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

# --- Grid -----------------------------------------------------------------
# radius=1   → each cell's neighbourhood spans 3 cells (left, self, right);
#              increase to 2 or 3 to let cells sense a wider context
# boundary   → 'periodic' wraps edges so the grid has no privileged boundary
#              cells; try 'fixed' (dead edges) or 'reflecting' (mirror edges)
# initial_states → 'random' gives each cell a random binary state at t=0;
#                  try 'single_center' (only one live cell) for a seeded start
grid = Grid1D(
    size=GRID_SIZE,
    radius=1,             # neighbourhood radius of 1 results in 8 wolfram rules
    boundary='periodic',
    initial_states='random',
)

# --- Feedback function ----------------------------------------------------
# The feedback function is the sole source of the learning signal.
# Swapping it fundamentally changes what the automata collectively learn:
#
#   NeighbourhoodAgreementFeedback → reward matching the local majority
#   MinorityDisagreementFeedback   → reward differing from the local majority
#
# The radius here must match (or be ≤) the grid's neighbourhood radius.
# feedback_fn = NeighbourhoodAgreementFeedback(radius=5)
feedback_fn = MinorityDisagreementFeedback(radius=1)
# feedback_fn = GlobalTargetFeedback(target_state=1)

# --- System ---------------------------------------------------------------
# automaton_params:
#   n_states       — number of states per arm (confidence depth); must match
#                    the N_STATES constant passed to InteractiveGUI below
#   initial_state  — 'random' scatters TAs across all states at t=0;
#                    'boundary' starts every TA at the arm boundary (state 0),
#                    which maximises initial uncertainty
system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': N_STATES, 'initial_state': 'random'},
    feedback_fn=feedback_fn,
    feedback_radius=1,
    seed=SEED,
)

# --- GUI ------------------------------------------------------------------
# plot_type='augmented' → colour space–time diagram showing both cell state
#   (red = arm 1, blue = arm 0) and confidence (saturation).
#   A "View: B&W / Color" toggle button lets you switch without rerunning.
# n_states   → must match automaton_params['n_states']; drives legend granularity.
# interval   → ms between frames; increase if the animation feels too fast.
# view_window → most-recent N generations kept in the scrolling display.
gui = InteractiveGUI(
    system=system,
    max_generations=MAX_GENERATIONS,
    plot_type='augmented',
    n_states=N_STATES,
    params={
        'Grid': GRID_SIZE,
        'States per arm': N_STATES,
        'Feedback': 'Minority',
        'FB neighbor radius': 1,
        'Boundary': 'periodic',
        'Initial grid': 'random',
        'Seed': SEED,
    },
    figsize=(16, 9),
    interval=80,
    view_window=80,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
)

print("Binary CLA — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
