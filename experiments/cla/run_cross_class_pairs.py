"""
Headless batch runner: cross-class Wolfram rule pair experiments.

Runs all 1,659 cross-class rule pairs and saves analysis artifacts per pair,
using the same export logic as the interactive GUI (identical layout and proportions):
  - spacetime_bw.png       : binary cell-state diagram (black & white)
  - spacetime_color.png    : TA-confidence rule-choice diagram (blue = rule A, red = rule B;
                             shade encodes how deeply committed the TA is to that rule)
  - hamming_distance.png   : normalized row-to-row Hamming distance time series
  - metrics.json           : classification and summary metrics

Output tree (not tracked by git):
  experiments/results/
    I_x_II/rule<A>_vs_rule<B>/spacetime_bw.png
                              /spacetime_color.png
                              /hamming_distance.png
                              /metrics.json
    I_x_III/...
    I_x_IV/...
    II_x_III/...
    II_x_IV/...
    III_x_IV/...

Usage:
    python experiments/cla/run_cross_class_pairs.py

Already-completed pairs (all artifacts present) are skipped, so the run is resumable.
"""

import csv
import json
import os
import random
import tempfile
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

# Headless backend must be selected before importing project visualization code.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dtm_matplotlib_cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "dtm_xdg_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cakit.visualization import InteractiveGUI
from cakit.system import RuleSystem
from cakit.rules import (
    CLASS_SHORT_LABELS,
    WOLFRAM_CLASSES,
    get_cross_class_pairs,
)
from cakit.grid import Grid1D
from cakit.feedback import (
    MinorityDisagreementFeedback,
    NeighbourhoodAgreementFeedback,
)
from cakit.automata import TsetlinAutomaton


# ---- Experiment configuration ------------------------------------------------

GRID_SIZE = 201
N_STATES = 10       # states per TA arm → 2N = 20 internal states total
GENERATIONS = 500
SEED = 42
FEEDBACK_RADIUS = 1

# Feedback kind: "minority"  → MinorityDisagreementFeedback  (rewards local minority)
#                "majority"  → NeighbourhoodAgreementFeedback (rewards local majority)
FEEDBACK_KIND = "majority"

# Results root is derived from feedback kind so runs never overwrite each other.
RESULTS_ROOT = Path(__file__).resolve().parent.parent / \
    f"results_{FEEDBACK_KIND}"
SUMMARY_PATH = RESULTS_ROOT / "summary.csv"

# Hamming classification settings.
HAMMING_BURN_IN = 100
HAMMING_FINAL_WINDOW = 100
MAX_PERIOD = 20
FIXED_TOL = 1e-12
ACTIVE_TOL = 1e-12

_PAIR_SUBDIR: dict = {
    (1, 2): "I_x_II",
    (1, 3): "I_x_III",
    (1, 4): "I_x_IV",
    (2, 3): "II_x_III",
    (2, 4): "II_x_IV",
    (3, 4): "III_x_IV",
}


# ---- Hamming-distance analysis -----------------------------------------------

def _compute_hamming_distance(grid_history: np.ndarray) -> np.ndarray:
    """Return normalized Hamming distance between consecutive generations."""
    return np.mean(grid_history[1:] != grid_history[:-1], axis=1)


def _detect_series_period(
    series: np.ndarray,
    max_period: int = MAX_PERIOD,
    tol: float = FIXED_TOL,
) -> Optional[int]:
    """Detect an exact or near-exact short period in a numeric time series."""
    if len(series) < 2:
        return None

    for period in range(1, min(max_period, len(series) // 2) + 1):
        if np.allclose(series[period:], series[:-period], atol=tol, rtol=0):
            return period

    return None


def _detect_grid_period(
    grid_history: np.ndarray,
    max_period: int = MAX_PERIOD,
) -> Optional[int]:
    """Detect a short period in the actual grid rows."""
    if len(grid_history) < 2:
        return None

    for period in range(1, min(max_period, len(grid_history) // 2) + 1):
        if np.array_equal(grid_history[period:], grid_history[:-period]):
            return period

    return None


def _classify_hamming(
    grid_history: np.ndarray,
    hamming: np.ndarray,
) -> Dict[str, object]:
    """Classify a completed run using the Hamming series and final grid tail."""
    final_window = min(HAMMING_FINAL_WINDOW, len(hamming))
    final_hamming = hamming[-final_window:] if final_window else hamming
    final_grid = grid_history[-(final_window + 1):] if final_window else grid_history
    post_burn_in = hamming[min(HAMMING_BURN_IN, len(hamming)):]

    final_max = float(np.max(final_hamming)) if len(final_hamming) else 0.0
    final_mean = float(np.mean(final_hamming)) if len(final_hamming) else 0.0
    final_std = float(np.std(final_hamming)) if len(final_hamming) else 0.0
    overall_mean = float(np.mean(hamming)) if len(hamming) else 0.0
    overall_std = float(np.std(hamming)) if len(hamming) else 0.0
    post_burn_in_mean = float(np.mean(post_burn_in)) if len(post_burn_in) else 0.0

    post_burn_in_grid = grid_history[min(HAMMING_BURN_IN, len(grid_history) - 1):]
    post_burn_in_grid_period = _detect_grid_period(post_burn_in_grid)
    final_grid_period = _detect_grid_period(final_grid)
    hamming_period = _detect_series_period(final_hamming)
    active_after_burn_in_before_final = bool(
        len(post_burn_in) > final_window
        and np.max(post_burn_in[:-final_window]) > ACTIVE_TOL
    )

    if final_max <= FIXED_TOL:
        label = (
            "transient_active_then_fixed"
            if active_after_burn_in_before_final
            else "fixed"
        )
    elif final_grid_period is not None:
        label = (
            "transient_active_then_periodic"
            if post_burn_in_grid_period is None
            else "periodic"
        )
    else:
        label = "active"

    return {
        "label": label,
        "mean_hamming": overall_mean,
        "std_hamming": overall_std,
        "mean_hamming_after_burn_in": post_burn_in_mean,
        "mean_hamming_final_window": final_mean,
        "std_hamming_final_window": final_std,
        "max_hamming_final_window": final_max,
        "detected_grid_period": final_grid_period,
        "detected_grid_period_after_burn_in": post_burn_in_grid_period,
        "detected_hamming_period": hamming_period,
        "burn_in": HAMMING_BURN_IN,
        "final_window": final_window,
        "max_period": MAX_PERIOD,
    }


def _plot_hamming_distance(hamming: np.ndarray, save_path: Path) -> None:
    """Save a normalized Hamming-distance time-series plot."""
    generations = np.arange(1, len(hamming) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(generations, hamming, color="#1f77b4", linewidth=1.4)
    ax.axvline(
        HAMMING_BURN_IN,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        label=f"burn-in = {HAMMING_BURN_IN}",
    )
    ax.set_title("Row-to-Row Hamming Distance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fraction of cells changed")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _write_metrics(metrics_path: Path, metrics: Dict[str, object]) -> None:
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _read_metrics(metrics_path: Path) -> Dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_summary(rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "rule_a",
        "rule_b",
        "class_pair",
        "label",
        "mean_hamming",
        "std_hamming",
        "mean_hamming_after_burn_in",
        "mean_hamming_final_window",
        "std_hamming_final_window",
        "max_hamming_final_window",
        "detected_grid_period",
        "detected_grid_period_after_burn_in",
        "detected_hamming_period",
        "plot_path",
        "metrics_path",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---- Per-pair runner ---------------------------------------------------------

def _run_pair(rule_a: int, rule_b: int, out_dir: Path) -> Tuple[bool, Dict[str, object]]:
    """
    Run a single rule pair and save diagrams plus Hamming artifacts to out_dir.

    Uses InteractiveGUI's export path (headlessly) so the saved images are
    identical in layout and proportions to what the GUI produces on Save.

    Returns (ran, metrics), where ran is False only if all artifacts already existed.
    """
    bw_path = out_dir / "spacetime_bw.png"
    color_path = out_dir / "spacetime_color.png"
    hamming_path = out_dir / "hamming_distance.png"
    metrics_path = out_dir / "metrics.json"

    if (
        bw_path.exists()
        and color_path.exists()
        and hamming_path.exists()
        and metrics_path.exists()
    ):
        return False, _read_metrics(metrics_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    random.seed(SEED)

    grid = Grid1D(
        size=GRID_SIZE,
        radius=1,
        boundary="periodic",
        initial_states="random",
    )
    if FEEDBACK_KIND == "majority":
        feedback_fn = NeighbourhoodAgreementFeedback(radius=FEEDBACK_RADIUS)
    else:
        feedback_fn = MinorityDisagreementFeedback(radius=FEEDBACK_RADIUS)

    system = RuleSystem(
        rules=[rule_a, rule_b],
        grid=grid,
        automaton_type=TsetlinAutomaton,
        automaton_params={"n_states": N_STATES, "initial_state": "random"},
        feedback_fn=feedback_fn,
        feedback_radius=FEEDBACK_RADIUS,
        seed=SEED,
    )

    cls_a = WOLFRAM_CLASSES[rule_a]
    cls_b = WOLFRAM_CLASSES[rule_b]
    arm_label_a = f"Rule {rule_a} · Class {CLASS_SHORT_LABELS[cls_a]}"
    arm_label_b = f"Rule {rule_b} · Class {CLASS_SHORT_LABELS[cls_b]}"

    params = {
        "Grid": GRID_SIZE,
        "Generations": GENERATIONS,
        "States/arm": N_STATES,
        "Feedback": FEEDBACK_KIND,
        "FB radius": FEEDBACK_RADIUS,
        "Seed": SEED,
        "Rules": f"{rule_a} vs {rule_b}",
    }

    # Instantiate GUI without calling show() — no window is created.
    # It allocates buffers and captures generation 0.
    gui = InteractiveGUI(
        system=system,
        max_generations=GENERATIONS,
        plot_type="augmented",
        n_states=N_STATES,
        arm_labels=[arm_label_a, arm_label_b],
        params=params,
        machine_label="CLA (Tsetlin · rule selection)",
        save_dir=str(out_dir),
    )

    # Drive the simulation manually, capturing each generation into GUI buffers.
    for gen in range(1, GENERATIONS + 1):
        system.step()
        gui._current_gen = gen
        gui._capture_row(gen)

    # Export both views using the GUI's own layout engine.
    # _export_all() appends _bw.png and _color.png automatically.
    gui._export_all(str(out_dir / "spacetime"))

    grid_history = gui._buf_std[:GENERATIONS + 1].astype(np.int8)
    hamming = _compute_hamming_distance(grid_history)
    metrics = _classify_hamming(grid_history, hamming)
    metrics.update({
        "rule_a": rule_a,
        "rule_b": rule_b,
        "grid_size": GRID_SIZE,
        "generations": GENERATIONS,
        "feedback": FEEDBACK_KIND,
        "feedback_radius": FEEDBACK_RADIUS,
        "seed": SEED,
        "hamming_plot": str(hamming_path),
    })

    _plot_hamming_distance(hamming, hamming_path)
    _write_metrics(metrics_path, metrics)

    return True, metrics


# ---- Main --------------------------------------------------------------------

def main() -> None:
    pairs = get_cross_class_pairs()
    total = len(pairs)

    print(f"Cross-class pair batch run")
    print(f"  Pairs     : {total}")
    print(f"  Grid      : {GRID_SIZE} cells  ×  {GENERATIONS} generations")
    print(f"  TA states : {N_STATES} per arm  (2N = {2 * N_STATES})")
    print(f"  Feedback  : {FEEDBACK_KIND}  (radius {FEEDBACK_RADIUS})")
    print(f"  Seed      : {SEED}")
    print(f"  Output    : {RESULTS_ROOT}")
    print()

    t_start = time.time()
    skipped = 0
    summary_rows: List[Dict[str, object]] = []

    for idx, (rule_a, rule_b) in enumerate(pairs, start=1):
        cls_a = WOLFRAM_CLASSES[rule_a]
        cls_b = WOLFRAM_CLASSES[rule_b]
        pair_key: Tuple[int, int] = (min(cls_a, cls_b), max(cls_a, cls_b))
        pair_dir = RESULTS_ROOT / \
            _PAIR_SUBDIR[pair_key] / f"rule{rule_a}_vs_rule{rule_b}"

        t0 = time.time()
        ran, metrics = _run_pair(rule_a, rule_b, pair_dir)
        elapsed = time.time() - t0

        if not ran:
            skipped += 1
            status = "skip"
        else:
            status = f"{elapsed:.1f}s"

        avg = (time.time() - t_start) / idx
        eta_min = avg * (total - idx) / 60
        eta_str = f"  ETA {eta_min:.0f} min" if idx < total else ""

        print(
            f"[{idx:>4}/{total}]  "
            f"Rule {rule_a:>3} vs {rule_b:>3}  "
            f"({_PAIR_SUBDIR[pair_key]:<8})  "
            f"{status}  {metrics['label']}{eta_str}"
        )

        summary_rows.append({
            "rule_a": rule_a,
            "rule_b": rule_b,
            "class_pair": _PAIR_SUBDIR[pair_key],
            "label": metrics["label"],
            "mean_hamming": metrics["mean_hamming"],
            "std_hamming": metrics["std_hamming"],
            "mean_hamming_after_burn_in": metrics["mean_hamming_after_burn_in"],
            "mean_hamming_final_window": metrics["mean_hamming_final_window"],
            "std_hamming_final_window": metrics["std_hamming_final_window"],
            "max_hamming_final_window": metrics["max_hamming_final_window"],
            "detected_grid_period": metrics["detected_grid_period"],
            "detected_grid_period_after_burn_in": metrics[
                "detected_grid_period_after_burn_in"
            ],
            "detected_hamming_period": metrics["detected_hamming_period"],
            "plot_path": str(pair_dir / "hamming_distance.png"),
            "metrics_path": str(pair_dir / "metrics.json"),
        })

    total_min = (time.time() - t_start) / 60
    ran_count = total - skipped
    _write_summary(summary_rows)
    print()
    print(
        f"Done.  {ran_count} ran, {skipped} skipped  —  {total_min:.1f} min total")
    print(f"Results: {RESULTS_ROOT}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
