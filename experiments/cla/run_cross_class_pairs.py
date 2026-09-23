"""
Headless batch runner: cross-class Wolfram rule pair experiments.

Runs all 1,659 cross-class rule pairs and saves analysis artifacts per pair,
using the same export logic as the interactive GUI (identical layout and proportions):
  - spacetime_bw.png       : binary cell-state diagram (black & white)
  - spacetime_color.png    : TA-confidence rule-choice diagram (blue = rule A, red = rule B;
                             shade encodes how deeply committed the TA is to that rule)
  - hamming_distance.png   : row-to-row activity (fraction of cells that flip each step)
  - lag_hamming.png        : lag Hamming with and without spatial rolling
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
    python experiments/cla/run_cross_class_pairs.py --feedback minority --backfill-hamming

Already-completed pairs (all artifacts present) are skipped, so the run is resumable.
`--backfill-hamming` rebuilds Hamming plots, domination, settle time, and
metrics.json from existing spacetime_bw.png / spacetime_color.png files
without re-running the CLA (same diagrams, same seed).
"""

import argparse
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

# Lag-Hamming classification (after burn-in, allowing circular spatial rolls).
HAMMING_BURN_IN = 100
EXACT_TOL = 1e-6
VALLEY_RATIO = 0.5
NEAR_MAX = 0.25
NEW_LABELS = frozenset({"fixed", "cycle", "drift", "near_cycle", "aperiodic"})

# Color-plot domination: last DOMINATION_WINDOW rows of the decoded arm field.
DOMINATION_WINDOW = 100
DOMINATE_SHARE = 0.9

# B&W settle time: classify the last SETTLE_TAIL rows, then walk down the plot.
SETTLE_TAIL = 167
SETTLE_STEP = 50
SETTLE_EARLY_MAX = 100

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


def _post_burn_in(grid_history: np.ndarray) -> np.ndarray:
    start = min(HAMMING_BURN_IN, max(0, len(grid_history) - 2))
    return grid_history[start:]


def _signed_shift(index: int, width: int) -> int:
    """Map a circular roll of 0..G-1 onto a signed shift in [-G//2, G//2]."""
    if index == 0:
        return 0
    if index > width // 2:
        return index - width
    return index


def _best_shift(hamming_by_shift: np.ndarray, unshifted: float) -> int:
    """Pick the smallest |roll| that attains the minimum; prefer no roll if tied."""
    width = len(hamming_by_shift)
    best = float(np.min(hamming_by_shift))
    # One cell of slack: ignore rolls that only win on noise, keep true gliders
    # (a single 1 that moves disagrees in two cells, which is above this).
    if unshifted <= best + max(EXACT_TOL, 1.0 / width):
        return 0
    tied = np.flatnonzero(hamming_by_shift <= best + EXACT_TOL)
    signed = [_signed_shift(int(idx), width) for idx in tied]
    return min(signed, key=lambda shift: (abs(shift), shift != 0, shift))


def _lag_curves_on(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lag Hamming on a grid slice as given (no extra burn-in).

    Returns (unshifted, shift_aware, best_signed_shift) each of length max_lag,
    where index 0 is lag 1. Shift-aware uses a circular roll of the later row.
    """
    grid = grid.astype(np.int8)
    n_rows, width = grid.shape
    max_lag = max(1, n_rows // 3)
    unshifted = np.empty(max_lag, dtype=np.float64)
    shift_aware = np.empty(max_lag, dtype=np.float64)
    shifts = np.empty(max_lag, dtype=np.int16)

    signed = (2.0 * grid.astype(np.float64)) - 1.0
    spectra = np.fft.fft(signed, axis=1)

    for tau in range(1, max_lag + 1):
        unshifted[tau - 1] = float(np.mean(grid[tau:] != grid[:-tau]))
        mean_cross = np.mean(spectra[:-tau] * np.conjugate(spectra[tau:]), axis=0)
        correlation = np.fft.ifft(mean_cross).real
        hamming_by_shift = np.clip((width - correlation) / (2.0 * width), 0.0, 1.0)
        shift_aware[tau - 1] = float(np.min(hamming_by_shift))
        shifts[tau - 1] = _best_shift(hamming_by_shift, unshifted[tau - 1])

    return unshifted, shift_aware, shifts


def _lag_curves(grid_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lag Hamming after burn-in."""
    return _lag_curves_on(_post_burn_in(grid_history))


def _shift_aware_at(grid: np.ndarray, tau: int) -> float:
    """Shift-aware lag Hamming at a single lag, on the slice as given."""
    grid = grid.astype(np.int8)
    n_rows, width = grid.shape
    if tau < 1 or n_rows < tau + 1:
        return 1.0
    signed = (2.0 * grid.astype(np.float64)) - 1.0
    spectra = np.fft.fft(signed, axis=1)
    mean_cross = np.mean(spectra[:-tau] * np.conjugate(spectra[tau:]), axis=0)
    correlation = np.fft.ifft(mean_cross).real
    hamming_by_shift = np.clip((width - correlation) / (2.0 * width), 0.0, 1.0)
    return float(np.min(hamming_by_shift))


def _is_local_min(values: np.ndarray, index: int) -> bool:
    left = values[index - 1] if index > 0 else values[index]
    right = values[index + 1] if index + 1 < len(values) else values[index]
    return bool(values[index] <= left and values[index] <= right)


def _label_from_curves(
    unshifted: np.ndarray,
    shift_aware: np.ndarray,
    shifts: np.ndarray,
) -> Dict[str, object]:
    """Turn lag-Hamming curves into label / period / shift / residual."""
    clipped = np.where(shift_aware <= EXACT_TOL, 0.0, shift_aware)
    min_idx = int(np.argmin(clipped)) if len(clipped) else 0
    min_shift_hamming = float(shift_aware[min_idx]) if len(shift_aware) else 0.0
    min_shift_lag = min_idx + 1 if len(shift_aware) else None
    baseline = float(np.median(shift_aware)) if len(shift_aware) else 0.0

    exact_idx = None
    near_idx = None
    for idx, residual in enumerate(shift_aware):
        if residual <= EXACT_TOL:
            exact_idx = idx
            break
        if (
            near_idx is None
            and residual <= NEAR_MAX
            and residual <= VALLEY_RATIO * max(baseline, EXACT_TOL)
            and _is_local_min(shift_aware, idx)
        ):
            near_idx = idx

    if exact_idx is not None:
        period = exact_idx + 1
        shift = int(shifts[exact_idx])
        residual = float(shift_aware[exact_idx])
        residual_unshifted = float(unshifted[exact_idx])
        if period == 1 and shift == 0:
            label = "fixed"
        elif shift == 0:
            label = "cycle"
        else:
            label = "drift"
    elif near_idx is not None:
        period = near_idx + 1
        shift = int(shifts[near_idx])
        residual = float(shift_aware[near_idx])
        residual_unshifted = float(unshifted[near_idx])
        label = "near_cycle"
    else:
        period = None
        shift = None
        residual = min_shift_hamming
        residual_unshifted = (
            float(unshifted[min_idx]) if len(unshifted) else 0.0
        )
        label = "aperiodic"

    return {
        "label": label,
        "period": period,
        "shift": shift,
        "residual": residual,
        "residual_unshifted": residual_unshifted,
        "min_shift_hamming": min_shift_hamming,
        "min_shift_lag": min_shift_lag,
        "max_lag": int(len(shift_aware)),
    }


def _classify_lag(
    grid_history: np.ndarray,
    hamming: np.ndarray,
    unshifted: np.ndarray,
    shift_aware: np.ndarray,
    shifts: np.ndarray,
) -> Dict[str, object]:
    """Label the post-burn-in orbit from lag Hamming, allowing spatial rolls."""
    post_burn_in = hamming[min(HAMMING_BURN_IN, len(hamming)):]
    overall_mean = float(np.mean(hamming)) if len(hamming) else 0.0
    overall_std = float(np.std(hamming)) if len(hamming) else 0.0
    post_burn_in_mean = float(np.mean(post_burn_in)) if len(post_burn_in) else 0.0

    metrics = _label_from_curves(unshifted, shift_aware, shifts)
    metrics.update({
        "mean_hamming": overall_mean,
        "std_hamming": overall_std,
        "mean_hamming_after_burn_in": post_burn_in_mean,
        "burn_in": HAMMING_BURN_IN,
        "exact_tol": EXACT_TOL,
        "valley_ratio": VALLEY_RATIO,
        "near_max": NEAR_MAX,
    })
    return metrics


def _settle_metrics(grid_history: np.ndarray) -> Dict[str, object]:
    """
    When the B&W pattern at the bottom of the plot starts.

    Classify the last SETTLE_TAIL rows, then find the earliest generation
    from which the remaining plot is at least as clean at that period.
    """
    tail_len = min(SETTLE_TAIL, len(grid_history))
    tail = grid_history[-tail_len:]
    dest = _label_from_curves(*_lag_curves_on(tail))
    late_label = dest["label"]
    late_period = dest["period"]
    late_residual = float(dest["residual"])

    if late_label == "aperiodic" or late_period is None:
        return {
            "late_label": late_label,
            "late_period": late_period,
            "settled_at": None,
            "settle": "never",
        }

    threshold = late_residual + EXACT_TOL
    settled_at = None
    last_start = max(0, len(grid_history) - tail_len)
    for t in range(0, last_start + 1, SETTLE_STEP):
        slice_ = grid_history[t:]
        if len(slice_) < 2 * int(late_period) + 1:
            continue
        residual = _shift_aware_at(slice_, int(late_period))
        if residual <= threshold:
            settled_at = t
            break

    if settled_at is None:
        settled_at = last_start

    settle = "early" if settled_at <= SETTLE_EARLY_MAX else "late"
    return {
        "late_label": late_label,
        "late_period": late_period,
        "settled_at": settled_at,
        "settle": settle,
    }


def _domination_metrics(arm: np.ndarray) -> Dict[str, object]:
    """One-rule vs two-rule occupancy from the decoded color plot (arm 0 = A)."""
    window = arm[-min(DOMINATION_WINDOW, len(arm)):]
    n = int(window.size)
    n_b = int(np.sum(window))
    n_a = n - n_b
    share_a = float(n_a / n) if n else 0.0
    share_b = float(n_b / n) if n else 0.0
    complete_a = bool(n > 0 and n_a == n)
    complete_b = bool(n > 0 and n_b == n)

    if complete_a or complete_b:
        domination = "dominate_complete"
        dominant_rule = "a" if complete_a else "b"
    elif max(share_a, share_b) >= DOMINATE_SHARE:
        domination = "dominate"
        dominant_rule = "a" if share_a >= share_b else "b"
    else:
        domination = "coexist"
        dominant_rule = None

    return {
        "rule_share_a": share_a,
        "rule_share_b": share_b,
        "domination": domination,
        "dominant_rule": dominant_rule,
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


def _plot_lag_hamming(
    unshifted: np.ndarray,
    shift_aware: np.ndarray,
    period: Optional[int],
    save_path: Path,
) -> None:
    """Save unshifted vs roll-aware lag Hamming after burn-in."""
    lags = np.arange(1, len(shift_aware) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(lags, unshifted, color="#999999", linewidth=1.2, label="same positions")
    ax.plot(
        lags,
        shift_aware,
        color="#1f77b4",
        linewidth=1.6,
        label="best circular roll",
    )
    if period is not None:
        ax.axvline(
            period,
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label=f"detected period = {period}",
        )
    ax.set_title("Lag Hamming after burn-in", fontsize=13, fontweight="bold")
    ax.set_xlabel("Lag (generations)")
    ax.set_ylabel("Fraction of cells that differ")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _diagram_cell_samples(path: Path) -> np.ndarray:
    """
    Sample one pixel per cell from a GUI-exported spacetime PNG.

    The export writes 4×4 pixels per cell with nearest-neighbour interpolation.
    Sampling the interior of each block reconstructs the (T, G[, C]) array.
    """
    from PIL import Image

    n_rows = GENERATIONS + 1
    px_per_cell = 4
    dpi = 150

    diag_w = max(4.0, GRID_SIZE * px_per_cell / dpi)
    diag_h = max(3.0, n_rows * px_per_cell / dpi)
    legend_w = 1.6
    card_h = 0.65
    machine_h = 0.32
    margin_l, margin_r, margin_b, margin_t = 0.70, 0.10, 0.40, 0.20
    gap_card_diag = 0.58
    gap_diag_machine = 0.06
    gap_diag_legend = 0.10

    total_w = margin_l + diag_w + gap_diag_legend + legend_w + margin_r
    total_h = (
        margin_t + machine_h + gap_diag_machine + diag_h
        + gap_card_diag + card_h + margin_b
    )
    diag_y0 = margin_b + card_h + gap_card_diag

    x0 = int(round(margin_l * dpi))
    y0 = int(round(total_h * dpi - (diag_y0 + diag_h) * dpi))
    width = int(round(diag_w * dpi))
    height = int(round(diag_h * dpi))

    image = np.array(Image.open(path))
    crop = image[y0:y0 + height, x0:x0 + width]
    if crop.shape[0] < height or crop.shape[1] < width:
        raise ValueError(
            f"{path}: cropped diagram {crop.shape[:2]} is smaller than "
            f"expected {(height, width)}"
        )

    rows = np.arange(n_rows) * px_per_cell + 1
    cols = np.arange(GRID_SIZE) * px_per_cell + 1
    return crop[np.ix_(rows, cols)]


def _decode_grid_from_spacetime_bw(path: Path) -> np.ndarray:
    """Recover the binary grid from a GUI-exported spacetime_bw.png."""
    sample = _diagram_cell_samples(path)
    channel = sample[..., 0]
    unique = set(np.unique(channel).tolist())
    if not unique.issubset({0, 255}):
        raise ValueError(
            f"{path}: expected binary black/white cells, got {sorted(unique)[:12]}"
        )
    return (channel < 128).astype(np.int8)


def _decode_arm_from_spacetime_color(path: Path) -> np.ndarray:
    """
    Recover which TA arm each cell chose from spacetime_color.png.

    Blue shades (rule A) have B > R; red shades (rule B) have R > B.
    Returns 0/1 array (T, G).
    """
    sample = _diagram_cell_samples(path)
    if sample.ndim < 3 or sample.shape[-1] < 3:
        raise ValueError(f"{path}: expected an RGB color diagram")
    red = sample[..., 0].astype(np.int16)
    blue = sample[..., 2].astype(np.int16)
    return (red > blue).astype(np.int8)


def _pair_metrics(
    rule_a: int,
    rule_b: int,
    out_dir: Path,
    grid_history: np.ndarray,
) -> Dict[str, object]:
    hamming = _compute_hamming_distance(grid_history)
    unshifted, shift_aware, shifts = _lag_curves(grid_history)
    metrics = _classify_lag(
        grid_history, hamming, unshifted, shift_aware, shifts
    )
    metrics.update(_settle_metrics(grid_history))
    color_path = out_dir / "spacetime_color.png"
    if color_path.exists():
        arm = _decode_arm_from_spacetime_color(color_path)
        if arm.shape != grid_history.shape:
            raise ValueError(
                f"{color_path}: decoded arm {arm.shape} != grid {grid_history.shape}"
            )
        metrics.update(_domination_metrics(arm))
    hamming_path = out_dir / "hamming_distance.png"
    lag_path = out_dir / "lag_hamming.png"
    metrics.update({
        "rule_a": rule_a,
        "rule_b": rule_b,
        "grid_size": GRID_SIZE,
        "generations": GENERATIONS,
        "feedback": FEEDBACK_KIND,
        "feedback_radius": FEEDBACK_RADIUS,
        "seed": SEED,
        "hamming_plot": str(hamming_path),
        "lag_hamming_plot": str(lag_path),
    })
    if not hamming_path.exists():
        _plot_hamming_distance(hamming, hamming_path)
    if not lag_path.exists():
        _plot_lag_hamming(unshifted, shift_aware, metrics["period"], lag_path)
    _write_metrics(out_dir / "metrics.json", metrics)
    return metrics


def _summary_row(
    rule_a: int,
    rule_b: int,
    class_pair: str,
    pair_dir: Path,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    return {
        "rule_a": rule_a,
        "rule_b": rule_b,
        "class_pair": class_pair,
        "label": metrics["label"],
        "period": metrics["period"],
        "shift": metrics["shift"],
        "residual": metrics["residual"],
        "residual_unshifted": metrics["residual_unshifted"],
        "mean_hamming": metrics["mean_hamming"],
        "std_hamming": metrics["std_hamming"],
        "mean_hamming_after_burn_in": metrics["mean_hamming_after_burn_in"],
        "domination": metrics.get("domination"),
        "dominant_rule": metrics.get("dominant_rule"),
        "rule_share_a": metrics.get("rule_share_a"),
        "rule_share_b": metrics.get("rule_share_b"),
        "settle": metrics.get("settle"),
        "settled_at": metrics.get("settled_at"),
        "late_label": metrics.get("late_label"),
        "late_period": metrics.get("late_period"),
        "plot_path": str(pair_dir / "hamming_distance.png"),
        "lag_plot_path": str(pair_dir / "lag_hamming.png"),
        "metrics_path": str(pair_dir / "metrics.json"),
    }


def _configure_feedback(kind: str) -> None:
    global FEEDBACK_KIND, RESULTS_ROOT, SUMMARY_PATH
    if kind not in {"majority", "minority"}:
        raise ValueError(f"Unknown feedback kind: {kind}")
    FEEDBACK_KIND = kind
    RESULTS_ROOT = Path(__file__).resolve().parent.parent / f"results_{kind}"
    SUMMARY_PATH = RESULTS_ROOT / "summary.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or backfill cross-class CLA pair experiments."
    )
    parser.add_argument(
        "--feedback",
        choices=("majority", "minority"),
        default=FEEDBACK_KIND,
        help="Feedback function / results_* directory (default: %(default)s)",
    )
    parser.add_argument(
        "--backfill-hamming",
        action="store_true",
        help=(
            "Rebuild Hamming / lag-Hamming plots, domination, settle time, "
            "and metrics.json from spacetime_bw.png and spacetime_color.png "
            "instead of re-running the simulation. Always overwrites analysis artifacts."
        ),
    )
    return parser.parse_args()


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
        "period",
        "shift",
        "residual",
        "residual_unshifted",
        "mean_hamming",
        "std_hamming",
        "mean_hamming_after_burn_in",
        "domination",
        "dominant_rule",
        "rule_share_a",
        "rule_share_b",
        "settle",
        "settled_at",
        "late_label",
        "late_period",
        "plot_path",
        "lag_plot_path",
        "metrics_path",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---- Per-pair runner ---------------------------------------------------------

def _analysis_complete(out_dir: Path) -> bool:
    metrics_path = out_dir / "metrics.json"
    if not (
        (out_dir / "hamming_distance.png").exists()
        and (out_dir / "lag_hamming.png").exists()
        and metrics_path.exists()
    ):
        return False
    metrics = _read_metrics(metrics_path)
    return (
        metrics.get("label") in NEW_LABELS
        and "period" in metrics
        and "shift" in metrics
        and "residual" in metrics
        and "domination" in metrics
        and "settle" in metrics
    )


def _run_pair(rule_a: int, rule_b: int, out_dir: Path) -> Tuple[bool, Dict[str, object]]:
    """
    Run a single rule pair and save diagrams plus Hamming artifacts to out_dir.

    Uses InteractiveGUI's export path (headlessly) so the saved images are
    identical in layout and proportions to what the GUI produces on Save.

    Returns (ran, metrics), where ran is False only if all artifacts already existed.
    """
    bw_path = out_dir / "spacetime_bw.png"
    color_path = out_dir / "spacetime_color.png"
    metrics_path = out_dir / "metrics.json"

    if bw_path.exists() and color_path.exists():
        if _analysis_complete(out_dir):
            return False, _read_metrics(metrics_path)
        grid_history = _decode_grid_from_spacetime_bw(bw_path)
        return True, _pair_metrics(rule_a, rule_b, out_dir, grid_history)

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
    return True, _pair_metrics(rule_a, rule_b, out_dir, grid_history)


def _backfill_pair(rule_a: int, rule_b: int, out_dir: Path) -> Tuple[bool, Dict[str, object]]:
    """Rebuild Hamming artifacts from spacetime_bw.png."""
    bw_path = out_dir / "spacetime_bw.png"
    if not bw_path.exists():
        raise FileNotFoundError(
            f"Cannot backfill {out_dir}: spacetime_bw.png is missing"
        )

    grid_history = _decode_grid_from_spacetime_bw(bw_path)
    return True, _pair_metrics(rule_a, rule_b, out_dir, grid_history)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _configure_feedback(args.feedback)

    pairs = get_cross_class_pairs()
    total = len(pairs)
    mode = "backfill Hamming from spacetime_bw.png" if args.backfill_hamming else "simulate"

    print(f"Cross-class pair batch run")
    print(f"  Mode      : {mode}")
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
        if args.backfill_hamming:
            ran, metrics = _backfill_pair(rule_a, rule_b, pair_dir)
        else:
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
            f"{status}  {metrics['label']}"
            f"  {metrics.get('domination', '-')}"
            f"  {metrics.get('settle', '-')}{eta_str}"
        )

        summary_rows.append(
            _summary_row(rule_a, rule_b, _PAIR_SUBDIR[pair_key], pair_dir, metrics)
        )

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
