"""
Rule-Selection CLA — Tsetlin Automata selecting between Wolfram rules  (Interactive GUI)

STRUCTURE
---------
This is a Rule-Selection Cellular Learning Automaton (CLA): each cell hosts a
Tsetlin Automaton (TA) whose job is NOT to output the cell's next state directly,
but to SELECT which Wolfram rule to apply.  The chosen rule then determines the
next state from the cell's neighbourhood.

This separates two concerns:
  - The TA learns WHICH rule is locally beneficial (the decision).
  - The Wolfram rule computes the cell's next state (the effect).

COMPOSITION
-----------
  Grid1D                      — 1-D array of binary cell states; owns boundary
                                handling and neighbourhood slicing.
  TsetlinAutomaton            — the per-cell LA.  It has two arms (arm 0 and
                                arm 1), each with n_states positions.  The arm
                                it currently occupies is its output (0 or 1),
                                which is used as an index into the rules list.
                                Reward pushes deeper into the current arm
                                (more confident); penalty moves toward the
                                boundary and may flip to the opposite arm.
  RuleSystem                  — the key difference from StateSystem.  After the
                                TA emits arm 0 or 1, RuleSystem looks up
                                rules[arm] and applies that Wolfram rule to the
                                3-cell neighbourhood to compute the next state.
                                The TA's output never reaches the grid directly.
  feedback_fn                 — evaluates each cell's neighbourhood after every
                                step and emits reward/penalty signals.  Feedback
                                is based on the resulting grid states (not on
                                which arm the TA chose).
  InteractiveGUI              — Tk/matplotlib window with Start/Pause/Step/
                                Reset/Save controls.  plot_type='augmented'
                                enables the colour view (red = arm 1 / rule 110,
                                blue = arm 0 / rule 30; saturation encodes
                                confidence) and the B&W toggle button.

RULE ASSIGNMENT
---------------
  rules[0] = 30  (arm 0) → chaotic, pseudo-random output
  rules[1] = 110 (arm 1) → Turing-complete, complex structured patterns

  The TA learns which of these two rules produces outcomes that are rewarded
  by the feedback function in its local neighbourhood context.

FEEDBACK OPTIONS
----------------
  NeighbourhoodAgreementFeedback(radius)   ← active in this experiment
      Reward a cell whose next state matches the majority of its neighbourhood.
      With rule selection this drives cells toward rules that produce local
      consensus (agreement with neighbours).

  MinorityDisagreementFeedback(radius)
      Reward a cell whose next state disagrees with the majority.
      With rule selection this drives cells toward rules that produce local
      diversity.

To run:
    python experiments/cla/tsetlin_cla-rule_system/tsetlin_cla-rule_system.py
"""
import os
import numpy as np
import random
from cakit.visualization import InteractiveGUI
from cakit.system import RuleSystem
from cakit.grid import Grid1D
from cakit.automata import TsetlinAutomaton
from cakit.feedback import NeighbourhoodAgreementFeedback
from cakit.feedback import MinorityDisagreementFeedback
from cakit.feedback import GlobalTargetFeedback
GRID_SIZE = 101        # number of cells; larger grids show richer spatial patterns
N_STATES = 10         # confidence resolution per TA arm; higher = slower learning,
# finer colour gradients in the augmented view
MAX_GENERATIONS = 200  # rows in the space–time diagram
SEED = 42             # set to None for a non-reproducible run

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
    radius=1,              # neighbourhood radius of 1 results in 8 wolfram rules
    boundary='periodic',
    initial_states='random',
)

# --- Feedback function ----------------------------------------------------
# The feedback function is the sole source of the learning signal.
# Swapping it changes what rule the TAs collectively learn to prefer:
#
#   NeighbourhoodAgreementFeedback → reward if next state matches local majority
#                                    (TAs learn the rule that drives consensus)
#                                    radius must match (or be <=) the grid's neighbourhood radius.
#   MinorityDisagreementFeedback   → reward if next state differs from majority
#                                    (TAs learn the rule that drives diversity)
#                                    radius must match (or be <=) the grid's neighbourhood radius.
#   GlobalTargetFeedback(target_state=1) rewards a cell if its resulting grid
#                   state (the output of whichever rule was applied) equals 1.  This does NOT
#                   directly reward choosing rule 110; it rewards whichever rule produces a 1
#                   for that cell's current neighbourhood.  TAs converge toward the rule that
#                   more reliably yields state=1 in their local context.
#                   target_state=1 means TAs learn to make the grid all 1s.
feedback_fn = GlobalTargetFeedback(target_state=1)
# feedback_fn = NeighbourhoodAgreementFeedback(radius=1)
# feedback_fn = MinorityDisagreementFeedback(radius=10)

# --- System ---------------------------------------------------------------
# rules         — list of Wolfram rule numbers; length must equal the TA's
#                 action_set_size (2 for TsetlinAutomaton).
#                 rules[0] is applied when the TA is on arm 0,
#                 rules[1] is applied when the TA is on arm 1.
#                 Try [90, 110], [30, 90], or any two rules with contrasting
#                 behaviour to observe how the TAs learn to prefer one over
#                 the other in different spatial contexts.
# automaton_params:
#   n_states       — number of states per arm (confidence depth); must match
#                    the N_STATES constant passed to InteractiveGUI below
#   initial_state  — 'random' scatters TAs across all states at t=0;
#                    'boundary' starts every TA at the arm boundary (state 0),
#                    which maximises initial uncertainty
system = RuleSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': N_STATES, 'initial_state': 'random'},
    rules=[30, 110],
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
    arm_labels=['Rule 30', 'Rule 110'],
    params={
        'Grid': GRID_SIZE,
        'States per arm': N_STATES,
        'Rules': '30 / 110',
        'Feedback': 'global(1)',
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

print("Rule-Selection CLA — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
