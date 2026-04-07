"""
Shared configuration for CLA experiment entry points.

Keep tunable parameters in one dataclass per script family; build grid, feedback,
system, and GUI from that single object so values are not duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import random

from cakit.feedback import (
    FeedbackFunction,
    GlobalTargetFeedback,
    MinorityDisagreementFeedback,
    NeighbourhoodAgreementFeedback,
)
from cakit.grid import Grid1D

FeedbackKind = Literal["minority", "agreement", "global"]
BoundaryKind = Literal["periodic", "zero", "fixed"]
InitialGridKind = Literal["random", "single_center", "half"]
AutomatonInitialState = Literal["random", "boundary"]


@dataclass
class ClaExperimentConfig:
    """Base configuration shared by binary CLA and rule-selection CLA scripts."""

    grid_size: int = 51
    neighbourhood_radius: int = 1
    boundary: BoundaryKind = "periodic"
    initial_grid: InitialGridKind = "random"

    n_states: int = 10
    automaton_initial_state: AutomatonInitialState = "random"

    feedback: FeedbackKind = "minority"
    feedback_neighbour_radius: int = 1
    global_target_state: int = 1

    feedback_radius: int = 1
    seed: Optional[int] = 42

    max_generations: int = 200
    plot_type: str = "augmented"
    interval: int = 80
    view_window: int = 80
    figsize: Tuple[float, float] = (16.0, 9.0)

    machine_label: Optional[str] = None


@dataclass
class RuleSystemExperimentConfig(ClaExperimentConfig):
    """Rule-selection CLA: Wolfram rules per TA arm."""

    grid_size: int = 101
    feedback: FeedbackKind = "global"
    rules: Tuple[int, int] = (30, 110)
    arm_labels: Tuple[str, str] = ("Rule 30", "Rule 110")
    machine_label: Optional[str] = "CLA (Tsetlin · rule selection)"


def apply_seed(seed: Optional[int]) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def make_grid(cfg: ClaExperimentConfig) -> Grid1D:
    return Grid1D(
        size=cfg.grid_size,
        radius=cfg.neighbourhood_radius,
        boundary=cfg.boundary,
        initial_states=cfg.initial_grid,
    )


def make_feedback(cfg: ClaExperimentConfig) -> FeedbackFunction:
    r = cfg.feedback_neighbour_radius
    if cfg.feedback == "minority":
        return MinorityDisagreementFeedback(radius=r)
    if cfg.feedback == "agreement":
        return NeighbourhoodAgreementFeedback(radius=r)
    if cfg.feedback == "global":
        return GlobalTargetFeedback(target_state=cfg.global_target_state)
    raise ValueError(f"Unknown feedback: {cfg.feedback!r}")


def automaton_params(cfg: ClaExperimentConfig) -> Dict[str, Union[int, str]]:
    return {
        "n_states": cfg.n_states,
        "initial_state": cfg.automaton_initial_state,
    }


def _feedback_label(cfg: ClaExperimentConfig) -> str:
    if cfg.feedback == "minority":
        return "Minority"
    if cfg.feedback == "agreement":
        return "Agreement"
    return f"global({cfg.global_target_state})"


def gui_params(cfg: ClaExperimentConfig) -> Dict[str, Any]:
    """Parameters shown in the GUI card; also drives default save filenames."""
    out: Dict[str, Any] = {
        "Grid": cfg.grid_size,
        "States per arm": cfg.n_states,
        "Feedback": _feedback_label(cfg),
        "FB neighbor radius": cfg.feedback_radius,
        "Boundary": cfg.boundary,
        "Initial grid": cfg.initial_grid,
        "Seed": cfg.seed,
    }
    if isinstance(cfg, RuleSystemExperimentConfig):
        out["Rules"] = f"{cfg.rules[0]} / {cfg.rules[1]}"
    return out


def save_dir_for_script(script_file: str) -> str:
    """Directory containing the running script (default export location)."""
    import os

    return os.path.dirname(os.path.abspath(script_file))


__all__ = [
    "AutomatonInitialState",
    "BoundaryKind",
    "ClaExperimentConfig",
    "FeedbackKind",
    "InitialGridKind",
    "RuleSystemExperimentConfig",
    "apply_seed",
    "automaton_params",
    "gui_params",
    "make_feedback",
    "make_grid",
    "save_dir_for_script",
]
