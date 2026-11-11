"""
G0 figures, from results/g0_*.json (no torch needed).

Usage:
    python -m experiments.relaxed_tm.game_of_life.plot_g0
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "reports" / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK_2, GRID = "#b4b2a9", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK_2, "ytick.color": INK_2,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 160,
})

MODELS = [  # file, label, colour
    ("g0_relaxed.json", "Relaxed TM (cross-entropy)", BLUE),
    ("g0_relaxed_squared.json", "Relaxed TM (squared error)", YELLOW),
    ("g0_standard.json", "Standard TM (Granmo 2018)", ORANGE),
]

# table rows grouped by centre state and number of live neighbours k
_i = np.arange(512)
_X = (_i[:, None] >> (8 - np.arange(9))) & 1
CENTRE = _X[:, 4]
K = _X.sum(axis=1) - CENTRE


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figures
def exact_rates() -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    y = np.arange(len(MODELS))[::-1]
    for yi, (f, label, col) in zip(y, MODELS):
        s = load(f)["summary"]
        final, ever = s["n_exact"], s["ever_exact"]
        ax.barh(yi + 0.18, ever, height=0.34, color=col, alpha=0.35)
        ax.barh(yi - 0.18, final, height=0.34, color=col)
        ax.text(ever + 1, yi + 0.18, f"ever exact: {ever}", va="center", fontsize=8, color=INK_2)
        ax.text(final + 1, yi - 0.18, f"exact at final epoch: {final}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_xlim(0, 135)
    ax.set_xlabel("seeds (of 100) with all 512 rows right, hard")
    ax.grid(axis="y", visible=False)
    ax.set_title("G0 · exact-rule rate, 256 clauses, 1,000 epochs")
    return save(fig, "g0_exact_rates.png")


def curves() -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for f, label, col in MODELS:
        H = 100 * np.array([r["hard_curve"] for r in load(f)["runs"]])
        ep = np.arange(1, H.shape[1] + 1)
        a1.plot(ep, H.mean(0), color=col, lw=1.6, label=label)
        a1.fill_between(ep, np.percentile(H, 10, 0), np.percentile(H, 90, 0), color=col, alpha=0.15, lw=0)
    a1.set_xscale("log")
    a1.set_ylim(70, 100.5)
    a1.set_xlabel("epoch")
    a1.set_ylabel("hard accuracy over the 512 rows (%)")
    a1.set_title("Hard accuracy (mean; band: 10th–90th percentile)")
    a1.legend(loc="lower right", fontsize=8)
    for f, label, col in MODELS[:2]:
        L = np.array([r["loss_curve"] for r in load(f)["runs"]])
        ep = np.arange(1, L.shape[1] + 1)
        a2.plot(ep, L.mean(0), color=col, lw=1.6, label=label)
        a2.fill_between(ep, np.percentile(L, 10, 0), np.percentile(L, 90, 0), color=col, alpha=0.15, lw=0)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("epoch")
    a2.set_ylabel("mean training loss")
    a2.set_title("Training loss (relaxed TM only; not comparable across losses)")
    a2.legend(loc="lower left", fontsize=8)
    return save(fig, "g0_curves.png")


def epochs_to_exact() -> Path:
    runs = load("g0_relaxed.json")["runs"]
    e = [r["first_exact_epoch"] for r in runs if r["first_exact_epoch"] is not None]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.hist(e, bins=np.arange(0, max(e) + 10, 10), color=BLUE)
    ax.axvline(np.median(e), color=INK, lw=1, ls="--")
    ax.text(np.median(e) + 5, ax.get_ylim()[1] * 0.9, f"median {np.median(e):.0f}", fontsize=8)
    ax.set_xlabel("first epoch with all 512 rows right (hard)")
    ax.set_ylabel("seeds")
    ax.set_title(f"Relaxed TM (cross-entropy): when it first becomes exact ({len(e)} of {len(runs)} seeds)")
    return save(fig, "g0_epochs_to_exact.png")


def wrong_rows() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6), sharey=True)
    groups = np.zeros((2, 9))
    for c in (0, 1):
        for k in range(9):
            groups[c, k] = np.sum((CENTRE == c) & (K == k))
    for ax, (f, label, col) in zip(axes, MODELS):
        runs = load(f)["runs"]
        wrong = np.zeros((2, 9))
        for r in runs:
            for w in r["wrong_rows"]:
                wrong[CENTRE[w], K[w]] += 1
        share = wrong / len(runs) / groups
        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("white")
        ax.imshow(np.ma.masked_equal(share, 0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for c in (0, 1):
            for k in range(9):
                v = 100 * share[c, k]
                if v > 0:
                    ax.text(k, c, "<1" if v < 1 else f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 50 else INK)
        for c in (0, 1):  # outline the groups whose next state is alive
            for k in ((3,) if c == 0 else (2, 3)):
                ax.add_patch(plt.Rectangle((k - 0.5, c - 0.5), 1, 1, fill=False, ec=BLUE, lw=1.2))
        ax.set_xticks(range(9))
        ax.set_xlabel("live neighbours k")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["centre dead", "centre alive"])
        ax.grid(False)
        ax.set_title(label, fontsize=9)
    fig.suptitle("Share of rows wrong at the final epoch (%), mean over 100 seeds, per group of neighbourhoods; "
                 "blue outline: groups whose next state is alive", fontsize=9, y=1.04)
    return save(fig, "g0_wrong_rows.png")


def clause_lengths() -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.2
    x = np.arange(19)
    series = []
    for f, label, col in MODELS:
        s = load(f)["summary"]
        runs = load(f)["runs"]
        if "clauses" in runs[0]:
            hist = np.mean([r["clauses"]["total"]["literals_hist"] for r in runs], axis=0)
        else:
            hist = np.mean([np.add(r["clauses_dead_bank"]["total"]["literals_hist"],
                                   r["clauses_alive_bank"]["total"]["literals_hist"]) for r in runs], axis=0)
        series.append((label, col, hist))
    ref = np.zeros(19)
    for n, lits in [(56, 8), (28, 9), (70, 4), (8, 7), (28, 9)]:
        ref[lits] += n
    series.append(("Reference set (190 clauses)", GREY, ref))
    for j, (label, col, hist) in enumerate(series):
        ax.bar(x + (j - 1.5) * width, hist, width=width, color=col, label=label)
    ax.set_xlim(-0.6, 10.6)
    ax.set_xticks(range(11))
    ax.set_xlabel("literals in a clause (active clauses only)")
    ax.set_ylabel("clauses per model (mean)")
    ax.set_title("Clause lengths: learned models vs the reference set")
    ax.legend(fontsize=8)
    return save(fig, "g0_clause_lengths.png")


def main() -> None:
    for fn in (exact_rates, curves, epochs_to_exact, wrong_rows, clause_lengths):
        print(f"wrote {fn().relative_to(HERE)}")


if __name__ == "__main__":
    main()
