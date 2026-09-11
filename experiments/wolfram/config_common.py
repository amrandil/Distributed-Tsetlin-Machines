"""
Shared configuration for elementary Wolfram CA experiment scripts.

Initial conditions (`initial_grid`) can be:

* A **preset string** understood by ``Grid1D``: ``random``, ``single_center``, ``half``.
* A **numpy vector** of length ``grid_size`` with values in ``{0, 1}`` (passed straight
  through to ``Grid1D``).
* A pattern built with **``line_from_binary_string``**, which turns a string of
  ``0``/``1`` characters into a length-``grid_size`` line (center, left, or right).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import random

from cakit.grid import Grid1D

BoundaryKind = Literal["periodic", "zero", "fixed"]
InitialGridKind = Literal["random", "single_center", "half"]
InitialGridSpec = Union[InitialGridKind, np.ndarray]


def line_from_binary_string(
    pattern: str,
    size: int,
    *,
    align: Literal["center", "left", "right"] = "center",
) -> np.ndarray:
    """
    Build a length-``size`` binary state row from a string of '0' and '1' characters.

    Whitespace in ``pattern`` is ignored. If the pattern has more bits than ``size``,
    the middle ``size`` bits are kept. If fewer, the rest are zeros; placement is
    controlled by ``align`` (``center``, ``left``, or ``right``).
    """
    if size < 3:
        raise ValueError(f"size must be >= 3, got {size}")
    bits = [int(c) for c in pattern if c in "01"]
    if not bits:
        raise ValueError("pattern must contain at least one '0' or '1'")
    n = len(bits)
    out = np.zeros(size, dtype=int)
    if n >= size:
        start = (n - size) // 2
        out[:] = bits[start : start + size]
    elif align == "left":
        out[:n] = bits
    elif align == "right":
        out[size - n :] = bits
    else:
        start = (size - n) // 2
        out[start : start + n] = bits
    return out


@dataclass
class WolframElementaryConfig:
    """1-D elementary CA (Wolfram rule 0–255), radius-1 neighbourhood."""

    rule: int
    grid_size: int = 201
    neighbourhood_radius: int = 1
    boundary: BoundaryKind = "periodic"
    #: Cell state used for out-of-range neighbors when ``boundary == "fixed"`` (ignored otherwise).
    boundary_value: int = 0
    initial_grid: InitialGridSpec = "single_center"

    seed: Optional[int] = 42

    max_generations: int = 400
    plot_type: str = "standard"
    interval: int = 60
    view_window: int = 80
    figsize: Tuple[float, float] = (16.0, 9.0)

    def __post_init__(self) -> None:
        if not 0 <= self.rule <= 255:
            raise ValueError(f"rule must be 0–255, got {self.rule}")
        if self.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.grid_size}")
        if self.neighbourhood_radius != 1:
            raise ValueError(
                "elementary Wolfram rules require neighbourhood_radius=1 (3 cells)"
            )


def apply_seed(seed: Optional[int]) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def make_grid(cfg: WolframElementaryConfig) -> Grid1D:
    init = cfg.initial_grid
    if isinstance(init, np.ndarray):
        if init.shape != (cfg.grid_size,):
            raise ValueError(
                f"initial_grid array must have shape ({cfg.grid_size},), got {init.shape}"
            )
        init_spec: Union[str, np.ndarray] = init
    else:
        init_spec = init
    return Grid1D(
        size=cfg.grid_size,
        radius=cfg.neighbourhood_radius,
        boundary=cfg.boundary,
        initial_states=init_spec,
        boundary_value=cfg.boundary_value,
    )


def automaton_params(cfg: WolframElementaryConfig) -> Dict[str, int]:
    return {"rule": cfg.rule}


def gui_params(cfg: WolframElementaryConfig) -> Dict[str, Any]:
    ig = cfg.initial_grid
    if isinstance(ig, np.ndarray):
        initial_label = f"custom ndarray {tuple(ig.shape)}"
    else:
        initial_label = ig
    return {
        "Rule": cfg.rule,
        "Grid": cfg.grid_size,
        "Initial grid": initial_label,
        "Boundary": cfg.boundary,
        "Boundary value": cfg.boundary_value,
        "Seed": cfg.seed,
    }


def save_dir_for_script(script_file: str) -> str:
    import os

    return os.path.dirname(os.path.abspath(script_file))


__all__ = [
    "BoundaryKind",
    "InitialGridKind",
    "InitialGridSpec",
    "WolframElementaryConfig",
    "apply_seed",
    "automaton_params",
    "gui_params",
    "line_from_binary_string",
    "make_grid",
    "save_dir_for_script",
]
