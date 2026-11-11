"""
Game of Life data: the 512-row table, a true Game of Life step, and grids.

A cell's input is its 3x3 neighbourhood read row by row, centre at index 4
(the same order as DiffLogic CA):

    NW  N  NE
     W  C   E       x = [NW, N, NE, W, C, E, SW, S, SE]
    SW  S  SE

Row i of the table is the neighbourhood whose bits, read as a binary number
with NW as the most significant bit, give i. Its label is the centre cell's
next state: 1 if it has exactly 3 live neighbours, or if it is alive and has
exactly 2.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

POSITIONS = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
CENTRE = 4
N_INPUTS = 9


def next_state(x: np.ndarray) -> np.ndarray:
    """Game of Life rule on neighbourhoods x of shape (..., 9)."""
    x = np.asarray(x)
    centre = x[..., CENTRE]
    k = x.sum(axis=-1) - centre
    return ((k == 3) | ((centre == 1) & (k == 2))).astype(np.int64)


def gol_table() -> Tuple[np.ndarray, np.ndarray]:
    """All 512 neighbourhoods (512, 9) and their next states (512,)."""
    i = np.arange(2 ** N_INPUTS)
    X = ((i[:, None] >> (N_INPUTS - 1 - np.arange(N_INPUTS))) & 1).astype(np.int64)
    return X, next_state(X)


def row_index(x: np.ndarray) -> np.ndarray:
    """Inverse of the table order: neighbourhood bits (..., 9) -> row number."""
    weights = 1 << (N_INPUTS - 1 - np.arange(N_INPUTS))
    return (np.asarray(x) * weights).sum(axis=-1)


def patches(grid: np.ndarray) -> np.ndarray:
    """
    The 3x3 neighbourhood of every cell of a 2D grid, with wrap-around edges.
    Returns (H*W, 9), cells in row-major order, bits in the order above.
    """
    grid = np.asarray(grid)
    cols = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # the neighbour at offset (dy, dx) of cell (r, c) is grid[r+dy, c+dx]
            cols.append(np.roll(grid, shift=(-dy, -dx), axis=(0, 1)).reshape(-1))
    return np.stack(cols, axis=1).astype(np.int64)


def gol_step(grid: np.ndarray) -> np.ndarray:
    """One step of Game of Life on a grid with wrap-around edges."""
    grid = np.asarray(grid)
    return next_state(patches(grid)).reshape(grid.shape)


def random_grid(size: int, seed: int, density: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) < density).astype(np.int64)


# Known patterns as lists of live cells (row, col), placed at an offset.
PATTERNS = {
    "block": [(0, 0), (0, 1), (1, 0), (1, 1)],              # still life
    "blinker": [(0, 0), (0, 1), (0, 2)],                    # period 2
    "glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],     # moves (+1, +1) every 4 steps
}


def pattern_grid(name: str, size: int = 16, offset: Tuple[int, int] = (4, 4)) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    for r, c in PATTERNS[name]:
        grid[(r + offset[0]) % size, (c + offset[1]) % size] = 1
    return grid
