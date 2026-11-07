"""
Figures and rule analysis for the E4 report (Iris).

Figures (reports/figures/):
- e4_accuracy_distribution.png: how many runs reach each test score (0-30
  flowers right), per model and encoding
- e4_paired_difference.png: relaxed minus standard TM on the same split, in flowers
- e4_tuning.png: validation accuracy (hard) for every tuning setting
- e4_training_curves.png: training loss, test accuracy (hard and soft) and
  undecided literals over the epochs
- e4_learning_speed.png: standard vs relaxed TM test accuracy over epochs and
  over training time

Rule analysis (printed): distinct clauses per class, and clauses rewritten as
one interval per measurement (thermometer encoding only), for the runs whose
rules were saved (seeds 0-19).

Reads only the results files; needs no torch.

Usage:
    python -m experiments.relaxed_tm.classification.plot_e4
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "reports" / "figures"
ENCODINGS = ("binary", "thermometer")
N_TEST = 30

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


def flowers(acc) -> np.ndarray:
    """Test accuracy -> number of the 30 test flowers right."""
    return np.rint(np.asarray(acc, dtype=float) * N_TEST).astype(int)


def standard_runs(enc: str) -> list:
    return load(f"e4_standard_tm_{enc}.json")["runs"]


def relaxed_final(enc: str) -> dict:
    return load(f"e4_relaxed_final_{enc}.json")["settings"][0]


# ---------------------------------------------------------------- figures
def accuracy_distribution() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, enc in zip(axes, ENCODINGS):
        groups = []
        if enc == "binary":
            groups.append(("Standard TM, his code his way", [r["test"] for r in load("e4_granmo2018_demo_binary_iris.json")["runs"]], GREY))
        groups.append(("Standard TM, our harness", [r["test"] for r in standard_runs(enc)], ORANGE))
        groups.append(("Relaxed TM (hard)", [r["eval_hard"] for r in relaxed_final(enc)["runs"]], BLUE))
        lo = min(flowers(g[1]).min() for g in groups)
        xs = np.arange(lo, N_TEST + 1)
        width = 0.8 / len(groups)
        for i, (label, acc, color) in enumerate(groups):
            counts = np.array([(flowers(acc) == x).sum() for x in xs])
            mean = 100 * np.mean(acc)
            ax.bar(xs + (i - (len(groups) - 1) / 2) * width, counts, width, color=color, label=f"{label}: mean {mean:.2f}%")
        ax.set_xticks(xs, [f"{x}\n{100 * x / N_TEST:.0f}%" for x in xs], fontsize=7)
        ax.set_xlabel("Test flowers right (of 30) and test accuracy")
        ax.set_title(f"{enc.capitalize()} encoding")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Runs (of 1,000)")
    fig.suptitle("E4 · Iris · test accuracy of every run", y=1.01, fontsize=11)
    fig.tight_layout()
    return save(fig, "e4_accuracy_distribution.png")


def paired_difference() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, enc in zip(axes, ENCODINGS):
        std = {r["seed"]: r["test"] for r in standard_runs(enc)}
        rel = {r["seed"]: r["eval_hard"] for r in relaxed_final(enc)["runs"]}
        d = np.array([flowers(rel[k]) - flowers(std[k]) for k in sorted(rel)])
        xs = np.arange(d.min(), d.max() + 1)
        counts = np.array([(d == x).sum() for x in xs])
        colors = [BLUE if x > 0 else ORANGE if x < 0 else GREY for x in xs]
        ax.bar(xs, counts, 0.7, color=colors)
        for x, c in zip(xs, counts):
            ax.text(x, c + 8, str(c), ha="center", fontsize=8, color=INK_2)
        diff = 100 * d / N_TEST
        ci = 1.96 * diff.std() / np.sqrt(len(diff))
        ax.set_title(f"{enc.capitalize()}: relaxed ahead {int((d > 0).sum())}, behind {int((d < 0).sum())}, "
                     f"same {int((d == 0).sum())}\nmean difference {diff.mean():+.2f} ± {ci:.2f} points")
        ax.set_xticks(xs)
        ax.set_xlabel("Relaxed TM (hard) minus standard TM, test flowers right (same split)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Splits (of 1,000)")
    fig.tight_layout()
    return save(fig, "e4_paired_difference.png")


def tuning() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    styles = {(-3.0, 0.01): (AQUA, "--"), (-3.0, 0.05): (AQUA, "-"), (-1.5, 0.01): (BLUE, "--"), (-1.5, 0.05): (BLUE, "-")}
    for ax, enc in zip(axes, ENCODINGS):
        settings = load(f"e4_relaxed_tune_{enc}.json")["settings"]
        chosen = relaxed_final(enc)
        for (start, lr), (color, ls) in styles.items():
            pts = sorted((st["tau"], st["summary"]["eval_hard"]["mean"]) for st in settings
                         if st["init_mean"] == start and st["lr"] == lr)
            ax.plot([t for t, _ in pts], [m for _, m in pts], color=color, ls=ls, marker="o", ms=4, lw=1.6,
                    label=f"start {start}, learning rate {lr}")
        best = next(st for st in settings if (st["init_mean"], st["tau"], st["lr"]) ==
                    (chosen["init_mean"], chosen["tau"], chosen["lr"]))
        ax.scatter([best["tau"]], [best["summary"]["eval_hard"]["mean"]], s=110, facecolor="none", edgecolor=INK, lw=1.2, zorder=5)
        ax.annotate("chosen", (best["tau"], best["summary"]["eval_hard"]["mean"]), xytext=(8, -14),
                    textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        ax.set_ylim(40, 100)
        ax.set_xlabel("τ (output steepness)")
        ax.set_title(f"{enc.capitalize()} encoding")
    axes[0].set_ylabel("Validation accuracy, hard (%)\n(20 splits, 24 flowers each)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("E4 · tuning pass: mean validation accuracy (hard) per setting", y=1.02, fontsize=11)
    fig.tight_layout()
    return save(fig, "e4_tuning.png")


def training_curves() -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.2), sharex=True)
    for c, enc in enumerate(ENCODINGS):
        st = relaxed_final(enc)
        runs = st["runs"]
        n_points = len(runs[0]["curve"]["eval_hard"])
        epochs = np.r_[np.arange(1, 500, 10), 500][:n_points]  # logged after epochs 1, 11, ..., 491 and 500
        hard = 100 * np.array([r["curve"]["eval_hard"] for r in runs])
        soft = 100 * np.array([r["curve"]["eval_soft"] for r in runs])
        und = 100 * np.array([r["curve"]["undecided"] for r in runs])
        loss = np.array([r["loss"] for r in runs])
        std_curve = 100 * np.array([r["test_curve"] for r in standard_runs(enc)])

        ax = axes[0, c]
        ax.fill_between(np.arange(1, 501), np.percentile(loss, 5, 0), np.percentile(loss, 95, 0), color=BLUE, alpha=0.15, lw=0)
        ax.plot(np.arange(1, 501), loss.mean(0), color=BLUE, lw=1.8)
        ax.set_title(f"Relaxed TM, {enc} encoding (start {st['init_mean']}, τ = {st['tau']:g}, lr {st['lr']:g})")
        if c == 0:
            ax.set_ylabel("Training loss\n(mean; band: 5th–95th pct)")

        ax = axes[1, c]
        ax.plot(epochs, hard.mean(0), color=BLUE, lw=2, label="relaxed TM, hard")
        ax.plot(epochs, soft.mean(0), color=YELLOW, lw=1.6, label="relaxed TM, soft")
        ax.plot(np.arange(10, 501, 10), std_curve.mean(0), color=ORANGE, lw=1.6, ls="--", label="standard TM (our harness)")
        ax.set_ylim(80, 97)
        ax.text(500, 80.6, f"after epoch 1 (below the axis): hard {hard[:, 0].mean():.1f}%, soft {soft[:, 0].mean():.1f}%",
                fontsize=8, color=INK_2, ha="right")
        if c == 0:
            ax.set_ylabel("Test accuracy (%)\n(mean of 1,000 runs)")
            ax.legend(loc="center right", fontsize=8)

        ax = axes[2, c]
        ax.plot(epochs, und.mean(0), color=BLUE, lw=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Epoch")
        if c == 0:
            ax.set_ylabel("Undecided literals (%)\n0.1 < p < 0.9")
    fig.suptitle("E4 · Iris · relaxed TM training (standard TM test accuracy for comparison)", y=1.0, fontsize=11)
    fig.tight_layout()
    return save(fig, "e4_training_curves.png")


def learning_speed() -> Path:
    """
    Mean test accuracy over training, standard TM (our harness) vs relaxed TM,
    on the same 1,000 splits. Top: per epoch (log scale). Bottom: per second of
    training, using each run's own time per epoch (total training time / 500,
    which includes the accuracy logging every 10 epochs, so approximate).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
    for c, enc in enumerate(ENCODINGS):
        rel = relaxed_final(enc)["runs"]
        std = standard_runs(enc)
        n_points = len(rel[0]["curve"]["eval_hard"])
        rel_epochs = np.r_[np.arange(1, 500, 10), 500][:n_points]  # relaxed: after epochs 1, 11, ..., 491, 500
        std_epochs = np.arange(10, 501, 10)                         # standard: after epochs 10, 20, ..., 500
        series = [
            ("Standard TM", std_epochs, 100 * np.array([r["test_curve"] for r in std]),
             np.mean([r["train_time_s"] for r in std]) / 500, ORANGE, "-"),
            ("Relaxed TM, hard", rel_epochs, 100 * np.array([r["curve"]["eval_hard"] for r in rel]),
             np.mean([r["train_time_s"] for r in rel]) / 500, BLUE, "-"),
            ("Relaxed TM, soft", rel_epochs, 100 * np.array([r["curve"]["eval_soft"] for r in rel]),
             np.mean([r["train_time_s"] for r in rel]) / 500, YELLOW, "--"),
        ]
        for row, xkind in enumerate(("epoch", "time")):
            ax = axes[row, c]
            for label, ep, acc, sec_per_epoch, color, ls in series:
                x = ep if xkind == "epoch" else ep * sec_per_epoch
                m = acc.mean(0)
                ci = 1.96 * acc.std(0) / np.sqrt(len(acc))
                ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.18, lw=0)
                early = ep <= 51  # mark the logged points where the curve is sparse
                ax.plot(x, m, color=color, ls=ls, lw=1.9,
                        label=f"{label} ({1000 * sec_per_epoch:.1f} ms per epoch)" if xkind == "time" else label)
                ax.plot(x[early], m[early], color=color, ls="none", marker="o", ms=3.5)
            final_std = series[0][2][:, -1].mean()
            ax.axhline(final_std, color=INK_2, lw=0.8, ls=(0, (3, 3)))
            ax.set_xscale("log")
            ax.set_ylim(80, 96)
            if xkind == "epoch":
                ax.set_xlabel("Epoch (log scale)")
                ax.set_xticks([1, 3, 10, 30, 100, 300, 500], ["1", "3", "10", "30", "100", "300", "500"])
                # first logged epoch at which the relaxed TM's mean hard accuracy reaches the standard TM's final mean
                hard = series[1][2].mean(0)
                reached = np.where(hard >= final_std)[0]
                note = (f"relaxed TM (hard) reaches the standard TM's final {final_std:.2f}% after epoch {rel_epochs[reached[0]]}"
                        if len(reached) else f"relaxed TM (hard) stays below the standard TM's final {final_std:.2f}%")
                ax.set_title(f"{enc.capitalize()} encoding\n{note}", fontsize=9)
            else:
                ax.set_xlabel("Training time per run, seconds (log scale; Mac CPU, approximate)")
            if c == 0:
                ax.set_ylabel("Test accuracy (%)\nmean of 1,000 runs ± 95% CI")
                ax.legend(loc="lower right", fontsize=8)
    axes[1, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("E4 · Iris · how fast the standard and the relaxed TM learn (same 1,000 splits)\n"
                 "dots: logged points (standard TM after epochs 10, 20, ...; relaxed TM after 1, 11, 21, ...); "
                 "dotted line: standard TM's final accuracy", y=1.02, fontsize=10)
    fig.tight_layout()
    return save(fig, "e4_learning_speed.png")


def save(fig, name: str) -> Path:
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- rules
LITERAL = re.compile(r"^(¬?)(\w+)>([0-9.]+)$")


def banks(lines: list) -> "OrderedDict[str, list]":
    """Rule lines from format_multiclass_rules -> {class: [clause lines]}."""
    out: "OrderedDict[str, list]" = OrderedDict()
    for line in lines:
        if not line.startswith("  "):
            out[line.strip("[] ")] = []
        else:
            out[next(reversed(out))].append(line.strip())
    return out


def simplify_thermometer(clause: str) -> str:
    """
    '+ : ¬a>6.10 ∧ ¬a>6.52 ∧ b>2.70' -> '+ : a ≤ 6.10 ∧ b > 2.70'. Thermometer
    bits are ordered (a > 6.52 implies a > 6.10), so each measurement's
    literals reduce to one interval. Empty clauses stay '(empty)'.
    """
    sign, _, body = clause.partition(" : ")
    if not body.strip():
        return f"{sign} : (empty)"
    low, high = {}, {}
    for lit in body.split(" ∧ "):
        neg, name, t = LITERAL.match(lit.strip()).groups()
        t = float(t)
        if neg:  # value <= t
            high[name] = min(high.get(name, np.inf), t)
        else:  # value > t
            low[name] = max(low.get(name, -np.inf), t)
    parts = []
    for name in dict.fromkeys(list(low) + list(high)):
        lo, hi = low.get(name), high.get(name)
        if lo is not None and hi is not None:
            parts.append(f"{lo:.2f} < {name} ≤ {hi:.2f}" if lo < hi else f"{name}: never (contradiction)")
        elif lo is not None:
            parts.append(f"{name} > {lo:.2f}")
        else:
            parts.append(f"{name} ≤ {hi:.2f}")
    return f"{sign} : " + " ∧ ".join(parts)


# Bits of Granmo's Binary Iris file that are 0 for all 150 flowers (his bins
# only go up to 3, so the top bit of each number is never set; bits 5, 6 are
# sepal width's top two bits)
CONSTANT_BITS = ("sepal_length.bit1", "sepal_width.bit1", "sepal_width.bit2", "petal_length.bit1",
                 "petal_width.bit1", "petal_width.bit2", "petal_width.bit3")


def strip_constant_bits(clause: str) -> str:
    """Drop literals ¬b for always-0 bits b (always true, so they change nothing)."""
    sign, _, body = clause.partition(" : ")
    keep = [lit for lit in body.split(" ∧ ") if lit.strip().lstrip("¬") not in CONSTANT_BITS or not lit.strip().startswith("¬")]
    return f"{sign} : " + " ∧ ".join(keep)


def rule_analysis() -> None:
    for enc in ENCODINGS:
        models = {"standard": {r["seed"]: (r["test"], r["rules"]) for r in standard_runs(enc) if "rules" in r},
                  "relaxed": {r["seed"]: (r["eval_hard"], r["rules"]) for r in relaxed_final(enc)["runs"] if "rules" in r}}
        print(f"\n=== {enc}: rules saved for seeds {sorted(models['relaxed'])[0]}-{sorted(models['relaxed'])[-1]}")
        for name, runs in models.items():
            distinct, distinct_simple, lengths, n_const = [], [], [], []
            for acc, lines in runs.values():
                for clauses in banks(lines).values():
                    active = [c for c in clauses if c.split(" : ")[1].strip()]
                    distinct.append(len(set(active)))
                    if enc == "binary":
                        stripped = [strip_constant_bits(c) for c in active]
                        n_const.append(sum(a != b for a, b in zip(active, stripped)) / max(len(active), 1))
                        distinct_simple.append(len(set(stripped)))
                    if enc == "thermometer":
                        simple = [simplify_thermometer(c) for c in active]
                        distinct_simple.append(len(set(simple)))
                        lengths.append(np.mean([s.count("∧") + 1 for s in simple]))
            msg = f"  {name:8s}: distinct active clauses per class {np.mean(distinct):.1f} (of 100)"
            if enc == "binary":
                msg += (f"; {100 * np.mean(n_const):.0f}% of them include a negated always-0 bit (always true, "
                        f"no effect); without those literals {np.mean(distinct_simple):.1f} distinct")
            if enc == "thermometer":
                msg += (f"; after rewriting as intervals {np.mean(distinct_simple):.1f}, "
                        f"measurements per clause {np.mean(lengths):.2f}")
            print(msg)
    # example: thermometer, one seed where both models get 29/30 or better
    std = {r["seed"]: r for r in standard_runs("thermometer") if "rules" in r}
    rel = {r["seed"]: r for r in relaxed_final("thermometer")["runs"] if "rules" in r}
    seed = next(s for s in sorted(rel) if flowers(rel[s]["eval_hard"]) >= 29 and flowers(std[s]["test"]) >= 29)
    print(f"\n=== example, thermometer, seed {seed}: relaxed {flowers(rel[seed]['eval_hard'])}/30, standard {flowers(std[seed]['test'])}/30")
    for name, lines in (("relaxed", rel[seed]["rules"]), ("standard", std[seed]["rules"])):
        print(f"--- {name}")
        for cls, clauses in banks(lines).items():
            ctr = Counter(simplify_thermometer(c) for c in clauses if c.split(" : ")[1].strip())
            print(f"  [{cls}] {len(ctr)} distinct, top 6 by count:")
            for c, n in ctr.most_common(6):
                print(f"    ×{n:<3d} {c}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (accuracy_distribution, paired_difference, tuning, training_curves, learning_speed):
        print(f"Saved {f().name}")
    rule_analysis()


if __name__ == "__main__":
    main()
