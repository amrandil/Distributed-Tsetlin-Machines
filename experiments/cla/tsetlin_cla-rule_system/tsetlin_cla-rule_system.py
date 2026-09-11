"""
Rule-Selection CLA — Tsetlin Automata selecting between Wolfram rules
"""
import importlib.util
import sys
from pathlib import Path

from cakit.automata import TsetlinAutomaton
from cakit.system import RuleSystem
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

RuleSystemExperimentConfig = _cc.RuleSystemExperimentConfig
apply_seed = _cc.apply_seed
automaton_params = _cc.automaton_params
gui_params = _cc.gui_params
make_feedback = _cc.make_feedback
make_grid = _cc.make_grid
save_dir_for_script = _cc.save_dir_for_script

# --- Experiment parameters (all explicit; edit here) ----------------------
cfg = RuleSystemExperimentConfig(
    # Grid
    grid_size=301,
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
    # Rule selection (this experiment only)
    rules=(4, 30),
    arm_labels=("Rule 4", "Rule 30"),
    # GUI / run
    max_generations=600,
    plot_type="augmented",
    interval=80,
    view_window=80,
    figsize=(16.0, 9.0),
    machine_label="CLA (Tsetlin · rule selection)",
)

apply_seed(cfg.seed)

grid = make_grid(cfg)
feedback_fn = make_feedback(cfg)

system = RuleSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params=automaton_params(cfg),
    rules=list(cfg.rules),
    feedback_fn=feedback_fn,
    feedback_radius=cfg.feedback_radius,
    seed=cfg.seed,
)

gui = InteractiveGUI(
    system=system,
    max_generations=cfg.max_generations,
    plot_type=cfg.plot_type,
    n_states=cfg.n_states,
    arm_labels=list(cfg.arm_labels),
    params=gui_params(cfg),
    figsize=cfg.figsize,
    interval=cfg.interval,
    view_window=cfg.view_window,
    save_dir=save_dir_for_script(__file__),
    machine_label=cfg.machine_label,
)

print("Rule-Selection CLA — use Start/Pause, Step, Reset to control the simulation.")
gui.show()
