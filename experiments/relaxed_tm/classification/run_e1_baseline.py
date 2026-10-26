"""
E1, part 1 - Standard Tsetlin Machine on noisy XOR (reproduces Granmo 2018).

Granmo's released data (5,000 training examples with 40% of labels
inverted, 5,000 clean test examples), Multi-Class TM with 2 classes x 10
clauses, T = 15, s = 3.9, 200 epochs, final epoch reported, 100 runs.
Published result: 99.3% ± 0.3 mean test accuracy.

Usage:
    python -m experiments.relaxed_tm.classification.run_e1_baseline
    python -m experiments.relaxed_tm.classification.run_e1_baseline --runs 4 --epochs 20 --no-save
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.relaxed_tm.classification.baselines import TMU_SOURCE, StandardTMConfig, run_standard_tm
from experiments.relaxed_tm.classification.data import load_noisy_xor
from experiments.relaxed_tm.classification.xor_rules import check_xor_rule
from experiments.relaxed_tm.common.train import summarize

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def one_run(job):
    seed, config = job
    X_train, y_train, X_test, y_test = load_noisy_xor()
    r = run_standard_tm(X_train, y_train, X_test, y_test, config, seed)
    hard = r.pop("hard")
    return {
        "seed": seed,
        **r,
        "rule_y1": check_xor_rule(hard.banks[1], target=1),
        "rule_y0": check_xor_rule(hard.banks[0], target=0),
        **hard.stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clauses-per-class", type=int, default=10)
    parser.add_argument("--T", type=int, default=15)
    parser.add_argument("--s", type=float, default=3.9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = StandardTMConfig(
        clauses_per_class=args.clauses_per_class, T=args.T, s=args.s, epochs=args.epochs
    )
    jobs = [(seed, config) for seed in range(args.runs)]

    start = time.perf_counter()
    runs = []
    with Pool(args.workers) as pool:
        for r in pool.imap(one_run, jobs):
            runs.append(r)
            print(
                f"seed {r['seed']:3d}: test {r['test']:.4f}  train {r['train']:.4f}  "
                f"XOR clauses present: {r['rule_y1']['core_present']}  "
                f"extra clauses (y=1 bank): {r['rule_y1']['n_extra']}  "
                f"{r['train_time_s']:.0f}s",
                flush=True,
            )

    test = [r["test"] for r in runs]
    summary = {
        "test_accuracy": summarize(test),
        "train_accuracy": summarize([r["train"] for r in runs]),
        "runs_at_or_above_99": int(sum(t >= 0.99 for t in test)),
        "runs_with_xor_core_y1": int(sum(r["rule_y1"]["core_present"] for r in runs)),
        "runs_with_exact_rule_y1": int(sum(r["rule_y1"]["exact"] for r in runs)),
        "mean_extra_clauses_y1": float(np.mean([r["rule_y1"]["n_extra"] for r in runs])),
        "hard_matches_tmu_all_runs": all(r["hard_matches_tmu"] for r in runs),
        "mean_train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
    }
    t = summary["test_accuracy"]
    print(
        f"\nTest accuracy {t['mean']:.2f}% ± {t['ci95']:.2f} "
        f"(5th-95th pct {t['p5']:.2f}-{t['p95']:.2f}, min {t['min']:.2f}, max {t['max']:.2f}); "
        f"{summary['runs_at_or_above_99']}/{args.runs} runs ≥ 99%; "
        f"wall time {time.perf_counter() - start:.0f}s"
    )

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "e1_standard_tm_noisy_xor.json"
        out.write_text(
            json.dumps(
                {"args": vars(args), "config": asdict(config), "tmu": TMU_SOURCE,
                 "summary": summary, "runs": runs},
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
