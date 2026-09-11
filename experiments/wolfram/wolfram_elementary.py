"""
Elementary CA — Wolfram rules 0–255 (interactive GUI)

Pure 1-D cellular automaton: fixed lookup table, no learning.

To run:
    python experiments/wolfram/wolfram_elementary.py

Experiment parameters are set explicitly in ``WolframElementaryConfig(...)`` below.
Helpers live in ``experiments/wolfram/config_common.py``.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from cakit.automata import WolframAutomaton
from cakit.system import StateSystem
from cakit.visualization import InteractiveGUI

_wolfram_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "wolfram_config_common",
    _wolfram_dir / "config_common.py",
)
assert _spec is not None and _spec.loader is not None
_cc = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _cc
_spec.loader.exec_module(_cc)

WolframElementaryConfig = _cc.WolframElementaryConfig
apply_seed = _cc.apply_seed
automaton_params = _cc.automaton_params
gui_params = _cc.gui_params
line_from_binary_string = _cc.line_from_binary_string
make_grid = _cc.make_grid
save_dir_for_script = _cc.save_dir_for_script

# --- Experiment parameters (all explicit; edit here) ----------------------
# initial_grid options:
#   * Preset str: "random" | "single_center" | "half"
#   * Binary line: line_from_binary_string("1011", grid_size, align="center")
#   * Full row:    np.array([...], dtype=int) with length == grid_size
#   * Alignment:   align="center" | "left" | "right"
# boundary options:
#   * "periodic"
#   * "zero"
#   * "fixed"
#   * boundary_value=1
cfg = WolframElementaryConfig(
    rule=110,
    grid_size=301,
    neighbourhood_radius=1,
    boundary="periodic",
    # boundary_value=1,
    initial_grid="random",
    # initial_grid=line_from_binary_string( "10000000000000000001", 201, align="center"),
    seed=42,
    max_generations=1000,
    plot_type="standard",
    interval=60,
    view_window=80,
    figsize=(16.0, 9.0),
)

apply_seed(cfg.seed)

grid = make_grid(cfg)

system = StateSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params=automaton_params(cfg),
    feedback_fn=None,
    seed=cfg.seed,
)

gui = InteractiveGUI(
    system=system,
    max_generations=cfg.max_generations,
    plot_type=cfg.plot_type,
    params=gui_params(cfg),
    figsize=cfg.figsize,
    interval=cfg.interval,
    view_window=cfg.view_window,
    save_dir=save_dir_for_script(__file__),
)

print(
    f"Wolfram rule {cfg.rule} — use Start/Pause, Step, Reset to control the simulation."
)
gui.show()
