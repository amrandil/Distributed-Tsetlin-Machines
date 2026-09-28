"""
TA depth diagram, drawn with the same layout as the GUI exports.

The TA-status plot (spacetime_color.png) encodes two things at once: the hue says
which arm (rule) each automaton is on, and the saturation says how deep it sits
inside that arm. When one arm dominates a region, the hue drowns out the
saturation, so small changes in depth are hard to see.

This diagram drops the hue and draws only the depth, in grey:

    white = deepest state of the arm (position N - 1, fully committed)
    black = boundary state (position 0, one penalty away from switching arm)

Row t is the automaton state after generation t, the same rows as
spacetime_color.png. Under majority feedback the texture of this plot is largely
a record of where cells were recently penalised (a cell that differs from both of
its neighbours), which is why it often echoes the B&W plot.

Geometry, fonts, header strip and params card copy
cakit.visualization.InteractiveGUI._export, so the PNGs of a pair line up when
shown side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

MACHINE_LABEL = "CLA (Tsetlin · rule selection) · TA depth only"


def depth_of(ta: np.ndarray, n_states: int) -> np.ndarray:
    """Position inside the current arm: 0 = boundary, n_states - 1 = deepest."""
    return (ta.astype(np.int64) - 1) % n_states


def save_depth_png(path: Path, ta: np.ndarray, n_states: int, params: dict) -> None:
    depth = depth_of(ta, n_states)
    n, width = depth.shape

    # ---- geometry copied from InteractiveGUI._export -------------------------
    dpi, px = 150, 4
    diag_w = max(4.0, width * px / dpi)
    diag_h = max(3.0, n * px / dpi)
    leg_w, card_h, machine_h = 1.6, 0.65, 0.32
    ml, mr, mb, mt = 0.70, 0.10, 0.40, 0.20
    gap_cb, gap_dm, gap_dl = 0.58, 0.06, 0.10
    total_w = ml + diag_w + gap_dl + leg_w + mr
    total_h = mt + machine_h + gap_dm + diag_h + gap_cb + card_h + mb

    fig = Figure(figsize=(total_w, total_h), dpi=dpi)
    FigureCanvasAgg(fig)

    def _norm(x, w, y, h):
        return [x / total_w, y / total_h, w / total_w, h / total_h]

    card_inset = 0.10
    card_w = diag_w - 2 * card_inset
    card_y0 = mb
    diag_y0 = mb + card_h + gap_cb
    machine_y0 = diag_y0 + diag_h + gap_dm

    ax = fig.add_axes(_norm(ml, diag_w, diag_y0, diag_h))
    leg = fig.add_axes(_norm(ml + diag_w + gap_dl, leg_w - mr, diag_y0, diag_h))
    head = fig.add_axes(_norm(ml + card_inset, card_w, machine_y0, machine_h))
    card = fig.add_axes(_norm(ml + card_inset, card_w, card_y0, card_h))

    # ---- diagram -------------------------------------------------------------
    ax.imshow(depth, cmap="gray", vmin=0, vmax=n_states - 1,
              aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xlabel("Cell Position", fontsize=9, labelpad=6)
    ax.set_ylabel("Generation", fontsize=9)

    # ---- header strip (as _draw_machine_header) -------------------------------
    head.set_xlim(0, 1)
    head.set_ylim(0, 1)
    head.set_axis_off()
    head.axhline(0, color="#cfd8dc", linewidth=0.9, clip_on=False)
    head.text(0.5, 0.5, MACHINE_LABEL, ha="center", va="center",
              fontsize=9, fontweight="bold", color="#37474f", transform=head.transAxes)

    # ---- legend: one block per depth, deepest at the top ---------------------
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.set_axis_off()
    leg.text(0.5, 0.975, "Legend", ha="center", va="top", fontsize=8, fontweight="bold")
    x0, bw = 0.22, 0.30
    y_top, y_bot = 0.80, 0.30
    bh = (y_top - y_bot) / n_states
    for k in range(n_states):
        level = k / (n_states - 1)
        y = y_bot + k * bh
        leg.add_patch(mpatches.Rectangle((x0, y), bw, bh, facecolor=str(level),
                                         edgecolor="#666666", linewidth=0.5))
    leg.text(x0 + bw + 0.05, y_top - bh / 2, "deepest\n(committed)", ha="left", va="center", fontsize=6.5)
    leg.text(x0 + bw + 0.05, y_bot + bh / 2, "boundary\n(uncertain)", ha="left", va="center", fontsize=6.5)
    last = depth[-100:] / (n_states - 1)
    leg.text(0.5, 0.22, f"last 100 gens:\nmean depth {last.mean():.2f}", ha="center", va="top", fontsize=7)

    # ---- params card (as _draw_params_card) ---------------------------------
    card.set_xlim(0, 1)
    card.set_ylim(0, 1)
    card.set_facecolor("#ffffff")
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_edgecolor("#5c6bc0")
    items = list(params.items())
    cols = (items[0::2], items[1::2])
    nrows = max(len(cols[0]), len(cols[1]), 1)
    for x_col, column in zip((0.02, 0.52), cols):
        for i, (k, v) in enumerate(column):
            y = 0.94 - (i + 0.5) / nrows * 0.88
            card.text(x_col, y, f"{k}:  {v}", ha="left", va="center",
                      fontsize=8.5, fontfamily="monospace", transform=card.transAxes)

    fig.savefig(path, dpi=dpi)
