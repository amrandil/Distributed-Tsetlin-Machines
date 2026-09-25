"""
Add the TA-layer metrics and the two refined axis tags to every pair.

Run AFTER `experiments/cla/run_cross_class_pairs.py` (including any
`--backfill-hamming` run, which rewrites metrics.json and summary.csv from
scratch and drops the fields added here).

    python experiments/analysis/axes_metrics.py                  # both feedbacks
    python experiments/analysis/axes_metrics.py --feedback majority
    python experiments/analysis/axes_metrics.py --verify-png     # also compare with the PNGs
    python experiments/analysis/axes_metrics.py --save-histories # also write history.npz per pair

What it changes
---------------
For every pair it re-simulates the run with fast_sim (bit-identical to cakit)
and merges these fields into metrics.json, then adds them as columns to
summary.csv. All windowed quantities use the last WINDOW generations, the same
window as the existing domination axis.

Axis 1, domination (tag values: dominate_complete, dominate, coexist, neutral)
  domination            'neutral' replaces 'coexist' when the rules hardly
                        ever competed: decisive_frac < NEUTRAL_MAX.
  domination_occupancy  the original occupancy-only tag, kept for reference.
  decisive_frac         share of cell-steps where rules A and B disagree on the
                        cell's neighbourhood, i.e. where the arm choice actually
                        decided the next state.
  decisive_share_a      among decisive cell-steps, share that used rule A
                        (null when there are none).

Axis 3, settle (tag values: early, late, tail_only, never)
  settle                'tail_only' replaces 'late' when settled_at is the
                        fallback of _settle_metrics (no starting point before
                        the tail reached the tail's cleanliness, so the run was
                        never shown to settle inside the plot).
  settle_raw            the original tag, kept for reference.

TA-layer descriptors (for the text, not used as filters)
  arm_flip_rate         share of cells switching arm per generation.
  freeze_gen            generation of the last arm switch anywhere on the grid
                        (0 = never switched; GENERATIONS = still switching at
                        the end).
  ta_confidence         mean depth inside the arm, 0 = boundary, 1 = deepest.
  arm_walls_initial     number of A/B boundaries around the ring at gen 0.
  arm_walls_final       same at the last generation (about 100 on 201 cells
                        means salt-and-pepper, a handful means territories).
  same_arm_as_initial   share of cells whose final arm equals their initial arm.
  occupancy_check       |simulated arm-A share - stored rule_share_a| (should be
                        0; a non-zero value means the PNG decode and the
                        simulation disagree).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO_ROOT = EXPERIMENTS.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cakit.rules import WOLFRAM_CLASSES, get_cross_class_pairs  # noqa: E402
from fast_sim import GENERATIONS, N_STATES, arm_of, confidence_of, simulate  # noqa: E402

# ---- Parameters (keep in sync with run_cross_class_pairs.py) -----------------
WINDOW = 100            # same as DOMINATION_WINDOW
DOMINATE_SHARE = 0.9    # same as DOMINATE_SHARE
NEUTRAL_MAX = 0.02      # decisive_frac below this -> 'neutral'
SETTLE_TAIL = 167       # same as SETTLE_TAIL
SETTLE_STEP = 50        # same as SETTLE_STEP

PAIR_SUBDIR = {
    (1, 2): "I_x_II", (1, 3): "I_x_III", (1, 4): "I_x_IV",
    (2, 3): "II_x_III", (2, 4): "II_x_IV", (3, 4): "III_x_IV",
}

NEW_FIELDS = [
    "domination_occupancy",
    "decisive_frac",
    "decisive_share_a",
    "settle_raw",
    "arm_flip_rate",
    "freeze_gen",
    "ta_confidence",
    "arm_walls_initial",
    "arm_walls_final",
    "same_arm_as_initial",
    "occupancy_check",
]


def pair_dir(results_root: Path, rule_a: int, rule_b: int) -> Tuple[str, Path]:
    ca, cb = WOLFRAM_CLASSES[rule_a], WOLFRAM_CLASSES[rule_b]
    sub = PAIR_SUBDIR[(min(ca, cb), max(ca, cb))]
    return sub, results_root / sub / f"rule{rule_a}_vs_rule{rule_b}"


def occupancy_domination(share_a: float) -> str:
    """Same rule as _domination_metrics in run_cross_class_pairs.py."""
    if share_a == 1.0 or share_a == 0.0:
        return "dominate_complete"
    if max(share_a, 1.0 - share_a) >= DOMINATE_SHARE:
        return "dominate"
    return "coexist"


def settle_tag(settle_raw: Optional[str], settled_at: Optional[int], n_rows: int) -> Optional[str]:
    """
    Flag the fallback of _settle_metrics.

    When no suffix start 0, 50, ... reaches the tail's cleanliness,
    _settle_metrics sets settled_at = n_rows - SETTLE_TAIL (334 here). That
    value is not on the 50-step grid, so it identifies the fallback exactly.
    """
    last_start = max(0, n_rows - SETTLE_TAIL)
    if (
        settle_raw == "late"
        and settled_at is not None
        and int(settled_at) == last_start
        and last_start % SETTLE_STEP != 0
    ):
        return "tail_only"
    return settle_raw


def ta_metrics(run: Dict[str, np.ndarray]) -> Dict[str, object]:
    grid, ta, decisive = run["grid"], run["ta"], run["decisive"]
    arm = arm_of(ta)
    n_gen = grid.shape[0] - 1

    # Occupancy exactly as the colour-plot axis: last WINDOW rows of the arm field.
    occ_a = float(np.mean(arm[-WINDOW:] == 0))

    # Decisive cell-steps for the transitions INTO the last WINDOW generations:
    # neighbourhood read at t, arm used at t, result written at t + 1.
    dec = decisive[-WINDOW:]
    arm_used = arm[-WINDOW - 1:-1]
    n_dec = int(dec.sum())
    decisive_frac = float(dec.mean())
    decisive_share_a = float(np.mean(arm_used[dec] == 0)) if n_dec else None

    switches = arm[1:] != arm[:-1]
    switched_rows = np.flatnonzero(switches.any(axis=1))
    freeze_gen = int(switched_rows[-1] + 1) if switched_rows.size else 0

    walls = np.sum(arm != np.roll(arm, 1, axis=1), axis=1)

    return {
        "occ_a": occ_a,
        "decisive_frac": decisive_frac,
        "decisive_share_a": decisive_share_a,
        "arm_flip_rate": float(switches[-WINDOW:].mean()),
        "freeze_gen": freeze_gen if freeze_gen < n_gen else n_gen,
        "ta_confidence": float(confidence_of(ta[-WINDOW:]).mean()),
        "arm_walls_initial": int(walls[0]),
        "arm_walls_final": int(walls[-1]),
        "same_arm_as_initial": float(np.mean(arm[-1] == arm[0])),
    }


def process_feedback(feedback: str, verify_png: bool, save_histories: bool) -> None:
    results_root = EXPERIMENTS / f"results_{feedback}"
    summary_path = results_root / "summary.csv"
    if not results_root.exists():
        print(f"  {results_root} not found, skipping {feedback}.")
        return

    decode = None
    if verify_png:
        sys.path.insert(0, str(EXPERIMENTS / "cla"))
        import run_cross_class_pairs as runner  # noqa: E402
        decode = runner._decode_grid_from_spacetime_bw

    pairs = get_cross_class_pairs()
    new_by_pair: Dict[Tuple[int, int], Dict[str, object]] = {}
    png_mismatch: List[str] = []
    occ_mismatch: List[str] = []
    t0 = time.time()

    for idx, (rule_a, rule_b) in enumerate(pairs, start=1):
        _, out_dir = pair_dir(results_root, rule_a, rule_b)
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  missing {metrics_path}, skipped")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        run = simulate(rule_a, rule_b, feedback)
        tm = ta_metrics(run)

        # Axis 1: occupancy tag recomputed from the exact arm field, then neutral.
        dom_occ = occupancy_domination(tm["occ_a"])
        domination = "neutral" if (dom_occ == "coexist" and tm["decisive_frac"] < NEUTRAL_MAX) else dom_occ

        # Axis 3: keep the original tag, flag the fallback.
        settle_raw = metrics.get("settle_raw", metrics.get("settle"))
        settle = settle_tag(settle_raw, metrics.get("settled_at"), run["grid"].shape[0])

        stored_share = metrics.get("rule_share_a")
        occupancy_check = abs(tm["occ_a"] - float(stored_share)) if stored_share is not None else None
        if occupancy_check is not None and occupancy_check > 1e-9:
            occ_mismatch.append(f"{rule_a} vs {rule_b}")

        new = {
            "domination": domination,
            "domination_occupancy": dom_occ,
            "decisive_frac": tm["decisive_frac"],
            "decisive_share_a": tm["decisive_share_a"],
            "settle": settle,
            "settle_raw": settle_raw,
            "arm_flip_rate": tm["arm_flip_rate"],
            "freeze_gen": tm["freeze_gen"],
            "ta_confidence": tm["ta_confidence"],
            "arm_walls_initial": tm["arm_walls_initial"],
            "arm_walls_final": tm["arm_walls_final"],
            "same_arm_as_initial": tm["same_arm_as_initial"],
            "occupancy_check": occupancy_check,
            "neutral_max": NEUTRAL_MAX,
        }
        metrics.update(new)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        new_by_pair[(rule_a, rule_b)] = new

        if decode is not None:
            bw = out_dir / "spacetime_bw.png"
            if bw.exists():
                decoded = decode(bw)
                if not np.array_equal(decoded.astype(np.int8), run["grid"]):
                    png_mismatch.append(f"{rule_a} vs {rule_b}")

        if save_histories:
            np.savez_compressed(
                out_dir / "history.npz",
                grid=run["grid"], ta=run["ta"], decisive=run["decisive"],
                n_states=N_STATES,
            )

        if idx % 200 == 0 or idx == len(pairs):
            print(f"  [{idx:>4}/{len(pairs)}]  {time.time() - t0:.0f}s")

    _update_summary(summary_path, new_by_pair)

    print(f"  metrics.json updated for {len(new_by_pair)} pairs; summary.csv rewritten.")
    print(f"  occupancy disagreements with stored rule_share_a: {len(occ_mismatch)}"
          + (f"  e.g. {occ_mismatch[:5]}" if occ_mismatch else ""))
    if decode is not None:
        print(f"  B&W PNG vs simulation mismatches: {len(png_mismatch)}"
              + (f"  e.g. {png_mismatch[:5]}" if png_mismatch else ""))


def _update_summary(summary_path: Path, new_by_pair: Dict[Tuple[int, int], Dict[str, object]]) -> None:
    if not summary_path.exists():
        print(f"  {summary_path} not found; metrics.json were updated but no summary written.")
        return
    with summary_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    # Insert the new columns right after the existing axis columns, once.
    for field in NEW_FIELDS:
        if field not in fieldnames:
            anchor = "late_period" if "late_period" in fieldnames else fieldnames[-1]
            fieldnames.insert(fieldnames.index(anchor) + 1 + NEW_FIELDS.index(field), field)
    for row in rows:
        new = new_by_pair.get((int(row["rule_a"]), int(row["rule_b"])))
        if new is None:
            continue
        for key, value in new.items():
            if key in fieldnames:
                row[key] = "" if value is None else value
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add TA-layer metrics, 'neutral' and 'tail_only' tags.")
    parser.add_argument("--feedback", choices=["majority", "minority", "both"], default="both")
    parser.add_argument("--verify-png", action="store_true",
                        help="Decode every spacetime_bw.png and check it equals the simulated grid.")
    parser.add_argument("--save-histories", action="store_true",
                        help="Write history.npz (grid, TA states, decisive mask) next to each pair's PNGs.")
    args = parser.parse_args()
    feedbacks = ["majority", "minority"] if args.feedback == "both" else [args.feedback]
    for fb in feedbacks:
        print(f"== {fb}")
        process_feedback(fb, args.verify_png, args.save_histories)


if __name__ == "__main__":
    main()
