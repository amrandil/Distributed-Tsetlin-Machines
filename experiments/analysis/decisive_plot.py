"""
Decisive cell-step diagram, drawn with the same layout as the GUI exports.

A cell-step (x, t) is decisive when rules A and B give different outputs for the
neighbourhood (left, self, right) that cell x sees at generation t. On those
cell-steps the arm the TA is on decides the cell's state at t + 1; everywhere
else both rules would produce the same next state.

Row t of the image therefore marks where the rule choice made at generation t
matters for generation t + 1. The last row (t = GENERATIONS) is computed the
same way so the image has exactly as many rows as spacetime_bw.png.

Geometry, fonts, header strip and params card copy
cakit.visualization.InteractiveGUI._export, so the three PNGs of a pair line up
when shown side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

DECISIVE_COLOR = "#1b5e20"
AGREE_COLOR = "#ffffff"
MACHINE_LABEL = "CLA (Tsetlin · rule selection) · decisive cell-steps"


def decisive_mask(grid: np.ndarray, rule_a: int, rule_b: int) -> np.ndarray:
    """(G+1, W) bool: rules A and B disagree on the neighbourhood at generation t."""
    g = grid.astype(np.int64)
    code = 4 * np.roll(g, 1, axis=1) + 2 * g + np.roll(g, -1, axis=1)
    return (((rule_a ^ rule_b) >> code) & 1).astype(bool)


def save_decisive_png(
    path: Path,
    grid: np.ndarray,
    rule_a: int,
    rule_b: int,
    params: dict,
) -> None:
    mask = decisive_mask(grid, rule_a, rule_b)
    n, width = mask.shape

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
    ax.imshow(mask.astype(np.int8), cmap=ListedColormap([AGREE_COLOR, DECISIVE_COLOR]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xlabel("Cell Position", fontsize=9, labelpad=6)
    ax.set_ylabel("Generation", fontsize=9)

    # ---- header strip (as _draw_machine_header) -------------------------------
    head.set_xlim(0, 1)
    head.set_ylim(0, 1)
    head.set_axis_off()
    head.axhline(0, color="#cfd8dc", linewidth=0.9, clip_on=False)
    head.text(0.5, 0.5, MACHINE_LABEL, ha="center", va="center",
              fontsize=9, fontweight="bold", color="#37474f", transform=head.transAxes)

    # ---- legend (layout of _draw_legend_standard) ---------------------------
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.set_axis_off()
    leg.text(0.5, 0.975, "Legend", ha="center", va="top", fontsize=8, fontweight="bold")
    pw, ph = 0.55, 0.18
    x0 = (1 - pw) / 2
    leg.add_patch(mpatches.FancyBboxPatch((x0, 0.56), pw, ph, boxstyle="round,pad=0.01",
                                          facecolor=AGREE_COLOR, edgecolor="black", linewidth=1))
    leg.text(0.5, 0.56 + ph / 2, "rules\nagree", ha="center", va="center", fontsize=8)
    leg.add_patch(mpatches.FancyBboxPatch((x0, 0.26), pw, ph, boxstyle="round,pad=0.01",
                                          facecolor=DECISIVE_COLOR, edgecolor="black", linewidth=1))
    leg.text(0.5, 0.26 + ph / 2, "rules\ndisagree", ha="center", va="center",
             fontsize=8, color="white")
    # Same window as decisive_frac in metrics.json: rows for generations 400..499.
    share = float(mask[-101:-1].mean())
    leg.text(0.5, 0.18, f"last 100 gens:\n{share:.1%} decisive", ha="center", va="top", fontsize=7)

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
