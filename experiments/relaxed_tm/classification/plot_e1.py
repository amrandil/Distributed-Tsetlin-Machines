"""
Figures for the E1 report (noisy XOR).

- e1_accuracy_per_run.png: final test accuracy of every run, standard TM
  (Granmo's code, his setup) vs the relaxed TM (2x10 and 1x20)
- e1_tuning.png: validation accuracy vs tau, per structure and start
- e1_training_curves.png: training loss, test accuracy (hard and soft) and
  undecided literals over the epochs

The final runs did not record the training loss, so 10 seeds per structure
are re-run here (cached in results/e1_relaxed_loss_<structure>.json); the
script checks that they end at the same test accuracy as the recorded runs.

Usage:
    python -m experiments.relaxed_tm.classification.plot_e1
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.relaxed_tm.classification.data import load_noisy_xor  # noqa: E402
from experiments.relaxed_tm.classification.run_e1_relaxed import one_run  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "reports" / "figures"
STRUCTURES = {"2x10": (-1.5, 15.0), "1x20": (-3.0, 15.0)}   # chosen in the tuning pass
LOSS_SEEDS = range(10)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK_2, GRID = "#b4b2a9", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK_2, "ytick.color": INK_2,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 160,
})


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def accuracy_per_run() -> Path:
    groups = [
        ("Standard TM (hard by nature)\n(Granmo's code, his setup)", [r["test"] for r in load("e1_granmo2018_demo_noisy_xor.json")["runs"]], GREY),
        ("Relaxed TM\n2 × 10", [r["eval_hard"] for r in load("e1_relaxed_final_2x10.json")["settings"][0]["runs"]], BLUE),
        ("Relaxed TM\n1 × 20", [r["eval_hard"] for r in load("e1_relaxed_final_1x20.json")["settings"][0]["runs"]], AQUA),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rng = np.random.default_rng(0)
    for i, (label, acc, color) in enumerate(groups):
        acc = 100 * np.asarray(acc)
        at_100 = acc == 100.0
        ax.scatter(i + rng.uniform(-0.25, 0.25, (~at_100).sum()), acc[~at_100], s=18, color=color,
                   edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
        ax.scatter(i + rng.uniform(-0.25, 0.25, at_100.sum()), acc[at_100], s=18, color=color, alpha=0.25,
                   edgecolor="none", zorder=2)
        ax.text(i, 100.25, f"{at_100.sum()} runs at 100%", ha="center", color=INK_2, fontsize=8)
        ax.plot([i - 0.32, i + 0.32], [acc.mean()] * 2, color=INK, lw=2, zorder=4)
        ax.text(i + 0.35, acc.mean(), f"mean {acc.mean():.2f}", va="center", fontsize=8, color=INK)
    ax.axhline(99.3, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
    ax.text(2.55, 99.3, "Granmo (2018),\npublished mean 99.3", color=ORANGE, fontsize=8, va="center")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.5, 3.2)
    ax.set_ylim(94.5, 100.6)
    ax.set_ylabel("Hard test accuracy after the final epoch (%)")
    ax.set_title("E1 · noisy XOR · hard test accuracy of every run (100 runs each)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = FIG_DIR / "e1_accuracy_per_run.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def tuning() -> Path:
    # ceiling: share of validation labels that equal the true XOR (a perfect model's score)
    X, y, _, _ = load_noisy_xor()
    val = np.random.default_rng(0).permutation(len(X))[:1000]
    ceiling = 100 * np.mean((X[val, 0] ^ X[val, 1]) == y[val])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, structure in zip(axes, STRUCTURES):
        d = load(f"e1_relaxed_tune_{structure}.json")
        for start, color in ((-3.0, BLUE), (-1.5, ORANGE)):
            pts = sorted((st["tau"], st["summary"]["eval_hard"]) for st in d["settings"] if st["init_mean"] == start)
            taus = [t for t, _ in pts]
            ax.errorbar(taus, [s["mean"] for _, s in pts], yerr=[s["ci95"] for _, s in pts], color=color,
                        marker="o", ms=4, lw=1.8, capsize=2, label=f"start w mean {start}")
        ax.axhline(ceiling, color=INK_2, lw=0.9, ls=(0, (3, 3)))
        ax.text(0.5, ceiling + 0.15, f"perfect model ({ceiling:.1f}%)", color=INK_2, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 3, 5, 15, 30], ["0.5", "1", "3", "5", "15", "30"])
        ax.set_xlabel("τ (output steepness)")
        ax.set_title(f"Relaxed TM {structure.replace('x', ' × ')}")
    axes[0].set_ylabel("Hard validation accuracy, noisy labels (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("E1 · tuning pass: hard validation accuracy (5 seeds per point, ± 95% CI)", y=1.02, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "e1_tuning.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def loss_reruns(structure: str) -> dict:
    """Re-run seeds 0-9 with the chosen setting to record the training loss (cached)."""
    path = RESULTS_DIR / f"e1_relaxed_loss_{structure}.json"
    if path.exists():
        return json.loads(path.read_text())
    init_mean, tau = STRUCTURES[structure]
    data = load_noisy_xor()
    jobs = [(structure, init_mean, tau, s, 200, 0.05, 128, data) for s in LOSS_SEEDS]
    with Pool(4) as pool:
        runs = pool.map(one_run, jobs)
    recorded = {r["seed"]: r["eval_hard"] for r in load(f"e1_relaxed_final_{structure}.json")["settings"][0]["runs"]}
    out = {
        "note": "Re-run of seeds 0-9 of the final runs, to record the training loss per epoch",
        "same_final_test_accuracy_as_recorded_runs": sum(r["eval_hard"] == recorded[r["seed"]] for r in runs),
        "runs": [{"seed": r["seed"], "loss": r["loss"], "eval_hard": r["eval_hard"]} for r in runs],
    }
    path.write_text(json.dumps(out, indent=1))
    return out


def training_curves() -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.2), sharex=True)
    for ax in axes[2]:
        ax.sharey(axes[2, 0])
    reference = np.mean([r["test"] for r in load("e1_granmo2018_demo_noisy_xor.json")["runs"]]) * 100
    # lowest possible training loss: the entropy of the label noise (share of inverted labels f)
    X, y, _, _ = load_noisy_xor()
    f = np.mean((X[:, 0] ^ X[:, 1]) != y)
    loss_floor = -(f * np.log(f) + (1 - f) * np.log(1 - f))
    for c, structure in enumerate(STRUCTURES):
        runs = load(f"e1_relaxed_final_{structure}.json")["settings"][0]["runs"]
        # train() logs after epochs 1, 11, 21, ..., 191 and after the last epoch (200)
        epochs = np.r_[np.arange(1, 200, 10), 200]
        hard = 100 * np.array([r["curve"]["eval_hard"][:len(epochs)] for r in runs])
        soft = 100 * np.array([r["curve"]["eval_soft"][:len(epochs)] for r in runs])
        und = 100 * np.array([r["curve"]["undecided"][:len(epochs)] for r in runs])

        lr = loss_reruns(structure)
        ax = axes[0, c]
        for r in lr["runs"]:
            ax.plot(np.arange(1, 201), r["loss"], color=BLUE, lw=1, alpha=0.5)
        ax.axhline(loss_floor, color=INK_2, lw=0.9, ls=(0, (3, 3)))
        ax.text(200, loss_floor - 0.0006, f"lowest possible: {loss_floor:.3f} ({100 * f:.1f}% of labels inverted)",
                color=INK_2, fontsize=8, ha="right", va="top")
        ax.set_ylim(loss_floor - 0.003, None)
        ax.set_title(f"Relaxed TM {structure.replace('x', ' × ')}")
        if c == 0:
            ax.set_ylabel("Training loss\n(10 seeds)")

        ax = axes[1, c]
        ax.fill_between(epochs, np.percentile(hard, 5, axis=0), np.percentile(hard, 95, axis=0), color=BLUE, alpha=0.15, lw=0)
        ax.plot(epochs, hard.mean(0), color=BLUE, lw=2, label="hard (mean; band: 5th–95th percentile)")
        ax.plot(epochs, soft.mean(0), color=ORANGE, lw=2, label="soft (mean)")
        ax.axhline(reference, color=INK_2, lw=0.9, ls=(0, (3, 3)))
        ax.text(200, reference + 0.15, f"standard TM, reproduced: {reference:.2f}", color=INK_2, fontsize=8, ha="right")
        ax.set_ylim(90, 100.6)
        ax.text(200, 90.4, f"after epoch 1 (below the axis): hard {hard[:, 0].mean():.1f}%, soft {soft[:, 0].mean():.1f}%",
                fontsize=8, color=INK_2, ha="right")
        if c == 0:
            ax.set_ylabel("Test accuracy, hard and soft (%)\n(100 runs)")
            ax.legend(loc="center right", fontsize=8)

        ax = axes[2, c]
        ax.plot(epochs, und.mean(0), color=BLUE, lw=2)
        start = STRUCTURES[structure][0]
        ax.text(200, und.mean(0)[0] * 0.9, f"start w mean {start} (p ≈ {1 / (1 + np.exp(-start)):.2f})",
                fontsize=8, color=INK_2, ha="right")
        ax.set_xlabel("Epoch")
        if c == 0:
            ax.set_ylabel("Undecided literals (%)\n0.1 < p < 0.9 (100 runs)")
    fig.suptitle("E1 · noisy XOR · relaxed TM training", y=1.0, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "e1_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (accuracy_per_run, tuning, training_curves):
        print(f"Saved {f().name}")
    for s in STRUCTURES:
        lr = loss_reruns(s)
        print(f"{s}: loss re-runs end at the recorded test accuracy in "
              f"{lr['same_final_test_accuracy_as_recorded_runs']}/{len(lr['runs'])} seeds")


if __name__ == "__main__":
    main()
