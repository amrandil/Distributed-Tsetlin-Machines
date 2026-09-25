"""
Vectorised re-implementation of the cross-class pair CLA run.

Why this exists
---------------
`experiments/cla/run_cross_class_pairs.py` drives cakit's RuleSystem cell by
cell and stores only the rendered PNGs. The follow-up analysis needs quantities
the PNGs do not carry (each cell's neighbourhood at each step, the TA's exact
internal state), and decoding pixels back into arrays is fragile. This module
reproduces the same run with numpy, bit for bit, in a fraction of the time.

Faithfulness
------------
* The initial grid and the initial TA states are taken from cakit itself,
  built exactly the way `_run_pair` builds them (same seeding order), so the
  starting point is not re-implemented, only read.
* The update loop mirrors RuleSystem + TsetlinAutomaton + the two feedback
  functions for radius-1, periodic, 1-D grids (the only setup used by the
  pair batch).
* `python experiments/analysis/fast_sim.py --check` replays a sample of pairs
  through cakit and asserts identical grid and TA histories for both feedbacks.
  Run it once after any change to cakit or to this file.

TA state convention (cakit TsetlinAutomaton, N states per arm)
---------------------------------------------------------------
Internal state s in 1..2N. arm = (s - 1) // N  (0 -> rule A, 1 -> rule B).
pos = (s - 1) % N is the depth inside the arm: 0 = boundary (least confident),
N - 1 = deepest (most confident). Reward moves one step deeper, penalty one
step towards the boundary, and a penalty at pos 0 switches to pos 0 of the
other arm. This is the same pos / (N - 1) the GUI uses for colour saturation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cakit.automata import TsetlinAutomaton  # noqa: E402
from cakit.feedback import (  # noqa: E402
    MinorityDisagreementFeedback,
    NeighbourhoodAgreementFeedback,
)
from cakit.grid import Grid1D  # noqa: E402
from cakit.system import RuleSystem  # noqa: E402

# Must match experiments/cla/run_cross_class_pairs.py
GRID_SIZE = 201
N_STATES = 10
GENERATIONS = 500
SEED = 42
FEEDBACK_RADIUS = 1
FEEDBACKS = ("majority", "minority")


def _build_cakit_system(rule_a: int, rule_b: int, feedback: str, seed: int) -> RuleSystem:
    """Build the system exactly as `_run_pair` does (same seeding order)."""
    np.random.seed(seed)
    random.seed(seed)
    grid = Grid1D(size=GRID_SIZE, radius=1, boundary="periodic", initial_states="random")
    if feedback == "majority":
        feedback_fn = NeighbourhoodAgreementFeedback(radius=FEEDBACK_RADIUS)
    elif feedback == "minority":
        feedback_fn = MinorityDisagreementFeedback(radius=FEEDBACK_RADIUS)
    else:
        raise ValueError(f"Unknown feedback: {feedback!r}")
    return RuleSystem(
        rules=[rule_a, rule_b],
        grid=grid,
        automaton_type=TsetlinAutomaton,
        automaton_params={"n_states": N_STATES, "initial_state": "random"},
        feedback_fn=feedback_fn,
        feedback_radius=FEEDBACK_RADIUS,
        seed=seed,
    )


_INITIAL_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def initial_conditions(seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initial grid and TA states for a seed, read from cakit.

    The pair batch reseeds before every pair, so all pairs share these. The
    rules and feedback do not influence the initial draw; rules 0/0 are used
    only to construct the objects.
    """
    if seed not in _INITIAL_CACHE:
        system = _build_cakit_system(0, 0, "majority", seed)
        grid0 = np.asarray(system.grid.get_all_states(), dtype=np.int64).copy()
        ta0 = np.array(
            [system.automata[p].get_state() for p in system.grid.get_positions()],
            dtype=np.int64,
        )
        _INITIAL_CACHE[seed] = (grid0, ta0)
    grid0, ta0 = _INITIAL_CACHE[seed]
    return grid0.copy(), ta0.copy()


def simulate(
    rule_a: int,
    rule_b: int,
    feedback: str,
    generations: int = GENERATIONS,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """
    Run one pair.

    Returns a dict of arrays, all indexed [generation, cell]:
      grid      (G+1, W) int8   cell states, row 0 = initial grid
      ta        (G+1, W) int8   TA internal states 1..2N, row 0 = initial
      decisive  (G,   W) bool   True where rules A and B disagree on the
                                neighbourhood cell x saw at generation t,
                                i.e. where its arm choice decided grid[t+1, x]
    """
    if feedback not in FEEDBACKS:
        raise ValueError(f"Unknown feedback: {feedback!r}")
    n = N_STATES
    x, s = initial_conditions(seed)
    width = x.size
    grid = np.empty((generations + 1, width), dtype=np.int8)
    ta = np.empty((generations + 1, width), dtype=np.int8)
    decisive = np.empty((generations, width), dtype=bool)
    grid[0], ta[0] = x, s
    differ = rule_a ^ rule_b

    for t in range(generations):
        code = 4 * np.roll(x, 1) + 2 * x + np.roll(x, -1)   # left, self, right
        arm = (s - 1) // n
        rule = np.where(arm == 0, rule_a, rule_b)
        decisive[t] = ((differ >> code) & 1).astype(bool)
        x = (rule >> code) & 1
        majority = (np.roll(x, 1) + x + np.roll(x, -1)) >= 2
        if feedback == "majority":
            reward = x == majority
        else:
            reward = x != majority
        pos = (s - 1) % n
        s = np.where(
            reward,
            np.where(pos < n - 1, s + 1, s),
            np.where(pos > 0, s - 1, np.where(arm == 0, n + 1, 1)),
        )
        grid[t + 1], ta[t + 1] = x, s

    return {"grid": grid, "ta": ta, "decisive": decisive}


def arm_of(ta: np.ndarray) -> np.ndarray:
    """0 = rule A, 1 = rule B."""
    return (ta.astype(np.int64) - 1) // N_STATES


def confidence_of(ta: np.ndarray) -> np.ndarray:
    """Depth inside the current arm, 0 (boundary) .. 1 (deepest)."""
    return ((ta.astype(np.int64) - 1) % N_STATES) / (N_STATES - 1)


def check_against_cakit(pairs: Iterable[Tuple[int, int]], generations: int = GENERATIONS) -> bool:
    ok = True
    for rule_a, rule_b in pairs:
        for feedback in FEEDBACKS:
            system = _build_cakit_system(rule_a, rule_b, feedback, SEED)
            grid_ref, ta_ref = system.run(generations)
            fast = simulate(rule_a, rule_b, feedback, generations)
            same_grid = np.array_equal(np.asarray(grid_ref), fast["grid"])
            same_ta = np.array_equal(np.asarray(ta_ref), fast["ta"])
            status = "ok" if (same_grid and same_ta) else "MISMATCH"
            print(f"  {rule_a:>3} vs {rule_b:>3}  {feedback:<8}  grid={same_grid}  ta={same_ta}  {status}")
            ok &= same_grid and same_ta
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Replay a sample of pairs through cakit and assert identical histories.",
    )
    args = parser.parse_args()
    if not args.check:
        parser.print_help()
        return
    # A spread of classes and both 'rules agree almost everywhere' and chaotic cases.
    sample = [(0, 2), (4, 30), (9, 45), (110, 40), (32, 105), (168, 150), (104, 105), (8, 54)]
    print(f"Checking fast_sim against cakit ({len(sample)} pairs x {len(FEEDBACKS)} feedbacks, "
          f"{GENERATIONS} generations)...")
    if check_against_cakit(sample):
        print("All identical.")
    else:
        print("Mismatch found: do not use fast_sim output until this is fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
