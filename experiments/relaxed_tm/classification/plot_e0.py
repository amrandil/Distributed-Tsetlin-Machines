"""
Figures for the E0 report (clean XOR).

Reads results/e0_clean_xor_m{4,6,8,10,20}.json (written by run_e0.py) and
retrains two 4-clause seeds to record include-probabilities per epoch
(deterministic: the same seed gives the same run as run_e0.py).

Usage:
    python -m experiments.relaxed_tm.classification.plot_e0
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cakit.tm import RelaxedTM, format_rules, harden  # noqa: E402
from experiments.relaxed_tm.classification.data import clean_xor  # noqa: E402
from experiments.relaxed_tm.common.train import TrainConfig, set_seed, train  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "reports" / "figures"
CLAUSE_COUNTS = [4, 6, 8, 10, 20]

# Reference categorical palette (validated for colour-vision deficiency)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 160,
})


def load(m: int) -> dict:
    return json.loads((RESULTS_DIR / f"e0_clean_xor_m{m}.json").read_text())


def training_curves() -> Path:
    rows = [
        ("loss", "Training loss (BCE)", True),
        ("train_soft", "Soft accuracy", False),
        ("train_hard", "Hard accuracy", False),
        ("undecided", "Undecided literals\n(0.1 < p < 0.9)", False),
    ]
    fig, axes = plt.subplots(len(rows), len(CLAUSE_COUNTS), figsize=(13, 9), sharex=True, sharey="row")
    for c, m in enumerate(CLAUSE_COUNTS):
        runs = load(m)["runs"]
        for r_i, (key, label, log) in enumerate(rows):
            ax = axes[r_i, c]
            # failing seeds drawn last so they sit on top
            for run in sorted(runs, key=lambda r: r["exact_xor_rule"], reverse=True):
                y = np.asarray(run["history"][key])
                x = np.arange(1, len(y) + 1)
                ok = run["exact_xor_rule"]
                ax.plot(x, y, color=BLUE if ok else ORANGE, lw=1.2 if ok else 1.8,
                        alpha=0.45 if ok else 0.95)
            if log:
                ax.set_yscale("log")
            if key in ("train_soft", "train_hard"):
                ax.set_ylim(0.45, 1.03)
            if c == 0:
                ax.set_ylabel(label)
            if r_i == 0:
                n_ok = sum(r["exact_xor_rule"] for r in runs)
                ax.set_title(f"{m} clauses  ·  exact rule {n_ok}/10")
            if r_i == len(rows) - 1:
                ax.set_xlabel("Epoch")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.6, label="seed that learns the exact XOR rule"),
        plt.Line2D([], [], color=ORANGE, lw=1.8, label="seed that does not"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("E0 · clean XOR · training curves, 10 seeds per clause count", y=1.03, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "e0_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def by_clause_count() -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(CLAUSE_COUNTS))
    exact = [load(m)["summary"]["seeds_with_exact_rule"] for m in CLAUSE_COUNTS]
    bars = a1.bar(x, exact, width=0.55, color=BLUE)
    for b, v in zip(bars, exact):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v}/10", ha="center", color=INK, fontsize=9)
    a1.set_xticks(x, [str(m) for m in CLAUSE_COUNTS])
    a1.set_ylim(0, 11.5)
    a1.set_xlabel("Clauses")
    a1.set_ylabel("Seeds (of 10)")
    a1.set_title("Seeds that learn the exact XOR rule")
    a1.grid(axis="x", visible=False)

    rng = np.random.default_rng(0)
    for i, m in enumerate(CLAUSE_COUNTS):
        e = [r["epochs_to_hard_100"] for r in load(m)["runs"]]
        reached = [v for v in e if v is not None]
        jitter = rng.uniform(-0.12, 0.12, len(reached))
        a2.scatter(np.full(len(reached), i) + jitter, reached, s=22, color=BLUE, alpha=0.7,
                   edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        a2.plot([i - 0.22, i + 0.22], [np.median(reached)] * 2, color=INK, lw=2, zorder=4)
        never = len(e) - len(reached)
        if never:
            a2.text(i, max(reached) + 1.2, f"{never} never", ha="center", color=ORANGE, fontsize=9)
    a2.set_xticks(x, [str(m) for m in CLAUSE_COUNTS])
    a2.set_xlabel("Clauses")
    a2.set_ylabel("Epochs")
    a2.set_ylim(0, 18)
    a2.set_title("Epochs until 100% hard accuracy (dots: seeds, bar: median)")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    out = FIG_DIR / "e0_by_clause_count.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def record_probs(m: int, seed: int, epochs: int = 100):
    """Retrain one seed exactly as run_e0.py does, recording p per epoch."""
    X, y = clean_xor(n_repeats=100)
    config = TrainConfig(epochs=epochs, batch_size=32, lr=0.05)
    set_seed(seed)
    model = RelaxedTM(2, m, tau=1.0, init_mean=-1.5)
    trace = [model.include_probs().detach().numpy().copy()]  # epoch 0 = initialisation
    train(model, X, y, config, on_epoch_end=lambda e, mdl: trace.append(mdl.include_probs().detach().numpy().copy()))
    return np.stack(trace), format_rules(harden(model), skip_empty=False)


def clause_probs(m: int = 4, seeds=(0, 7), epochs_shown: int = 40) -> Path:
    literal_names = ["x1", "x2", "¬x1", "¬x2"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(len(seeds), m, figsize=(3.0 * m, 2.7 * len(seeds)), sharex=True, sharey=True)
    for r_i, seed in enumerate(seeds):
        trace, rules = record_probs(m, seed)
        ok = "exact XOR rule" if set(rules) == {"+ : x1 ∧ ¬x2", "+ : ¬x1 ∧ x2", "− : x1 ∧ x2", "− : ¬x1 ∧ ¬x2"} \
            else "duplicate clause, a case uncovered"
        epochs = np.arange(trace.shape[0])[: epochs_shown + 1]
        for j in range(m):
            ax = axes[r_i, j]
            for k in range(4):
                p = trace[: epochs_shown + 1, j, k]
                ax.plot(epochs, p, color=colors[k], lw=2)
            ax.axhline(0.5, color=INK_2, lw=0.8, ls=(0, (3, 3)))
            ax.set_title(rules[j], fontsize=9)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlim(0, epochs_shown)
            if j == 0:
                ax.set_ylabel(f"seed {seed}: {ok}\ninclude-probability p")
            if r_i == len(seeds) - 1:
                ax.set_xlabel("Epoch")
    handles = [plt.Line2D([], [], color=c, lw=2, label=n) for c, n in zip(colors, literal_names)]
    handles.append(plt.Line2D([], [], color=INK_2, lw=0.8, ls=(0, (3, 3)), label="p = 0.5 (hardening threshold)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"E0 · {m} clauses · include-probabilities per clause (panel title: final hardened clause)",
                 y=1.07, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"e0_clause_probs_m{m}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for out in (training_curves(), by_clause_count(), clause_probs()):
        print(f"Saved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
