"""
Figure for the E0b report: seeds (of 50) that learn the exact XOR rule with 4
clauses, for every setting tested. Reads results/e0b_*_m4.json.

Usage:
    python -m experiments.relaxed_tm.classification.plot_e0b
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "reports" / "figures"

BLUE, ORANGE, GREY, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#b4b2a9", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160,
})

# (hypothesis file, setting index, label); the E0 setting is drawn once, in the first panel
PANELS = [
    ("H1 · starting point of w (p at start)", [
        ("init", 0, "mean −3 (p ≈ 0.05)"),
        ("init", 5, "mean −1.5, spread 0.1"),
        ("init", 1, "mean −1.5 (p ≈ 0.18) · E0"),
        ("init", 2, "mean −0.5 (p ≈ 0.38)"),
        ("init", 3, "mean 0 (p ≈ 0.5)"),
        ("init", 4, "mean 0, spread 0.1 · like Granmo's TM"),
    ]),
    ("H2 · T-like margin", [
        ("margin", 1, "margin 0.5"),
        ("margin", 2, "margin 1"),
        ("margin", 3, "margin 2"),
    ]),
    ("H3 · output steepness τ", [
        ("tau", 0, "τ = 0.5"),
        ("tau", 2, "τ = 2"),
        ("tau", 3, "τ = 4"),
    ]),
    ("Learning rate", [
        ("speed", 1, "boundary start, lr 0.01"),
        ("speed", 2, "boundary start, lr 0.002"),
        ("speed", 3, "E0 start, lr 0.01"),
        ("followup", 0, "mean −3, lr 0.15"),
    ]),
    ("Combined", [
        ("followup", 1, "mean −3 and τ = 0.5"),
    ]),
]


def main() -> None:
    data = {h: json.loads((RESULTS_DIR / f"e0b_{h}_m4.json").read_text())
            for h in ("init", "margin", "tau", "speed", "followup")}
    rows = []  # (label, exact, duplicate, panel title or None)
    for title, items in PANELS:
        rows.append((title, None, None, True))
        for h, i, lab in items:
            s = data[h]["settings"][i]["summary"]
            rows.append((lab, s["exact_xor_rule"], s["duplicated_clause"], False))

    fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 0.8))
    y = 0
    yticks, ylabels = [], []
    for lab, exact, dup, is_title in rows:
        if is_title:
            ax.text(-1, y, lab, fontweight="bold", color=INK, va="center", ha="left")
        else:
            ax.barh(y, exact, height=0.62, color=BLUE)
            ax.barh(y, dup, left=exact, height=0.62, color=ORANGE)
            ax.barh(y, 50 - exact - dup, left=exact + dup, height=0.62, color=GREY)
            ax.text(exact - 0.6, y, f"{exact}", va="center", ha="right", color="white", fontsize=8)
            yticks.append(y)
            ylabels.append(lab)
        y -= 1
    ax.axvline(35, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.text(35.4, 0.4, "E0 setting: 35", color=INK_2, fontsize=8, va="bottom")
    ax.set_yticks(yticks, ylabels)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Seeds (of 50)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="exact XOR rule"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="duplicated clause"),
               plt.Rectangle((0, 0), 1, 1, color=GREY, label="other wrong rule (half-built or contradictory clause)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.07), ncol=3, fontsize=8)
    ax.set_title("E0b · clean XOR, 4 clauses: what prevents duplicate clauses?", pad=24, fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "e0b_settings.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
