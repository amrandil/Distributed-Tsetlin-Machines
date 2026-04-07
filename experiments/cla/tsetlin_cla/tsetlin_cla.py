"""
Cellular Learning Automaton — Binary CLA with Tsetlin Automata
"""
import importlib.util
import sys
from pathlib import Path

from cakit.automata import TsetlinAutomaton
from cakit.system import StateSystem
from cakit.visualization import InteractiveGUI

# Load shared config by file path (safe when import sorters reorder normal imports).
_cla_dir = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "cla_config_common",
    _cla_dir / "config_common.py",
)
assert _spec is not None and _spec.loader is not None
_cc = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _cc
_spec.loader.exec_module(_cc)

ClaExperimentConfig = _cc.ClaExperimentConfig
apply_seed = _cc.apply_seed
automaton_params = _cc.automaton_params
gui_params = _cc.gui_params
make_feedback = _cc.make_feedback
make_grid = _cc.make_grid
save_dir_for_script = _cc.save_dir_for_script

# --- Experiment parameters (all explicit; edit here) ----------------------
cfg = ClaExperimentConfig(
    # Grid
    grid_size=101,
    neighbourhood_radius=1,
    boundary="periodic",
    initial_grid="random",
    # Automaton
    n_states=10,
    automaton_initial_state="random",
    # Feedback  (minority / agreement / global — see config_common.FeedbackKind)
    feedback="agreement",
    feedback_neighbour_radius=1,
    global_target_state=1,
    feedback_radius=1,
    # Reproducibility
    seed=42,
    # GUI / run
    max_generations=200,
    plot_type="augmented",
    interval=80,
    view_window=80,
    figsize=(16.0, 9.0),
    machine_label=None,
)

apply_seed(cfg.seed)

grid = make_grid(cfg)
feedback_fn = make_feedback(cfg)

system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params=automaton_params(cfg),
    feedback_fn=feedback_fn,
    feedback_radius=cfg.feedback_radius,
    seed=cfg.seed,
)

gui = InteractiveGUI(
    system=system,
    max_generations=cfg.max_generations,
    plot_type=cfg.plot_type,
    n_states=cfg.n_states,
    params=gui_params(cfg),
    figsize=cfg.figsize,
    interval=cfg.interval,
    view_window=cfg.view_window,
    save_dir=save_dir_for_script(__file__),
    machine_label=cfg.machine_label,
)

print("Binary CLA — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
