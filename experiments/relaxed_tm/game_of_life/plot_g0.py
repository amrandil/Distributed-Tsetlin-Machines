"""
G0 figures, from results/g0_*.json (no torch needed).

Usage:
    python -m experiments.relaxed_tm.game_of_life.plot_g0
"""

from __future__ import annotations

import json
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

MODELS = [  # results file, label, colour, line style
    ("g0_relaxed_single", "Relaxed, single-output", BLUE, "-"),
    ("g0_relaxed_multi", "Relaxed, multi-class", AQUA, "-"),
    ("g0_standard_single", "Standard, single-output", ORANGE, "--"),
    ("g0_standard_multi", "Standard, multi-class", YELLOW, "--"),
]

_i = np.arange(512)
_X = (_i[:, None] >> (8 - np.arange(9))) & 1
CENTRE = _X[:, 4]
K = _X.sum(axis=1) - CENTRE


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def exact_rates() -> Path:
    fig, ax = plt.subplots(figsize=(7, 2.9))
    y = np.arange(len(MODELS))[::-1]
    for yi, (f, label, col, _) in zip(y, MODELS):
        s = load(f)["summary"]
        final, ever = s["n_exact"], s["ever_exact"]
        ax.barh(yi + 0.18, ever, height=0.34, color=col, alpha=0.35)
        ax.barh(yi - 0.18, final, height=0.34, color=col)
        ax.text(ever + 1, yi + 0.18, f"ever exact: {ever}", va="center", fontsize=8, color=INK_2)
        ax.text(final + 1, yi - 0.18, f"exact at final epoch: {final}   (hard accuracy {s['hard']['mean']:.2f}%)",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_xlim(0, 175)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("seeds (of 100) with all 512 rows right, hard")
    ax.grid(axis="y", visible=False)
    ax.set_title("G0 · exact-rule rate, 744 clauses per TM, 1,000 epochs")
    return save(fig, "g0_exact_rates.png")


def curves() -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for f, label, col, ls in MODELS:
        H = 100 * np.array([r["hard_curve"] for r in load(f)["runs"]])
        ep = np.arange(1, H.shape[1] + 1)
        a1.plot(ep, H.mean(0), color=col, ls=ls, lw=1.6, label=label)
        a1.fill_between(ep, np.percentile(H, 10, 0), np.percentile(H, 90, 0), color=col, alpha=0.13, lw=0)
    a1.set_xscale("log")
    a1.set_ylim(60, 100.5)
    a1.set_xlabel("epoch")
    a1.set_ylabel("hard accuracy over the 512 rows (%)")
    a1.set_title("Hard accuracy (mean; band: 10th–90th percentile)")
    a1.legend(loc="lower right", fontsize=8)
    for f, label, col, ls in MODELS[:2]:
        Lc = np.array([r["loss_curve"] for r in load(f)["runs"]])
        ep = np.arange(1, Lc.shape[1] + 1)
        a2.plot(ep, Lc.mean(0), color=col, lw=1.6, label=label)
        a2.fill_between(ep, np.percentile(Lc, 10, 0), np.percentile(Lc, 90, 0), color=col, alpha=0.13, lw=0)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("epoch")
    a2.set_ylabel("mean training loss (cross-entropy)")
    a2.set_title("Training loss (relaxed TMs; the standard TM has no loss)")
    a2.legend(loc="lower left", fontsize=8)
    return save(fig, "g0_curves.png")


def epochs_to_exact() -> Path:
    fig, ax = plt.subplots(figsize=(7, 2.9))
    bins = np.arange(0, 330, 10)
    for f, label, col, _ in MODELS[:2]:
        e = [r["first_exact_epoch"] for r in load(f)["runs"] if r["first_exact_epoch"] is not None]
        ax.hist(e, bins=bins, color=col, alpha=0.6, label=f"{label} (median {np.median(e):.0f}, max {max(e)})")
    ax.set_xlabel("first epoch with all 512 rows right (hard)")
    ax.set_ylabel("seeds (of 100)")
    ax.set_title("Relaxed TMs: when they first become exact")
    ax.legend(fontsize=8)
    return save(fig, "g0_epochs_to_exact.png")


def wrong_rows() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
    groups = np.zeros((2, 9))
    for c in (0, 1):
        for k in range(9):
            groups[c, k] = np.sum((CENTRE == c) & (K == k))
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    for ax, (f, label, col, _) in zip(axes, MODELS[2:]):
        runs = load(f)["runs"]
        wrong = np.zeros((2, 9))
        for r in runs:
            for w in r["wrong_rows"]:
                wrong[CENTRE[w], K[w]] += 1
        share = wrong / len(runs) / groups
        ax.imshow(np.ma.masked_equal(share, 0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for c in (0, 1):
            for k in range(9):
                v = 100 * share[c, k]
                if v > 0:
                    ax.text(k, c, "<1" if v < 1 else f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 50 else INK)
            for k in ((3,) if c == 0 else (2, 3)):
                ax.add_patch(plt.Rectangle((k - 0.5, c - 0.5), 1, 1, fill=False, ec=BLUE, lw=1.2))
        ax.set_xticks(range(9))
        ax.set_xlabel("live neighbours k")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["centre dead", "centre alive"])
        ax.grid(False)
        ax.set_title(label, fontsize=9)
    fig.suptitle("Standard TMs: share of rows wrong at the final epoch (%), mean over 100 seeds; "
                 "blue outline: next state alive. (Relaxed TMs: no wrong rows.)", fontsize=9, y=1.05)
    return save(fig, "g0_wrong_rows.png")


def clause_lengths() -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.3))
    series = []
    for f, label, col, _ in MODELS:
        h = np.mean([r["clauses"]["total"]["literals_hist"] for r in load(f)["runs"]], axis=0)
        if "multi" in f:
            h = h / 2  # per TM
        series.append((label, col, h))
    ref = np.zeros(19)
    for n, lits in [(56, 8), (28, 9), (70, 4), (8, 7), (28, 9)]:
        ref[lits] += n
    series.append(("190-clause set", GREY, ref))
    width = 0.16
    x = np.arange(19)
    for j, (label, col, h) in enumerate(series):
        ax.bar(x + (j - 2) * width, h, width=width, color=col, label=label)
    ax.set_xlim(0.4, 9.6)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("literals in a clause (active clauses only)")
    ax.set_ylabel("clauses per TM (mean)")
    ax.set_title("Clause lengths per TM (744 clauses each)")
    ax.legend(fontsize=8, ncol=2)
    return save(fig, "g0_clause_lengths.png")


def main() -> None:
    for fn in (exact_rates, curves, epochs_to_exact, wrong_rows, clause_lengths):
        print(f"wrote {fn().relative_to(HERE)}")


if __name__ == "__main__":
    main()
