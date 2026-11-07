"""
E4 - Standard Tsetlin Machine on Iris, Granmo's April 2018 code in our harness.

His Binary Iris settings (Multi-Class TM, 300 clauses = 3 classes x 100,
T = 10, s = 3.0, 100 states per action, boost_true_positive_feedback = 1,
500 epochs, final epoch reported), but seeded, on the same splits as the
relaxed TM (seed k -> iris.iris_split(encoding, k)). Two encodings:
  binary       his Binary Iris file: the paired comparison for part 2
  thermometer  our encoding: the reference for part 3, since his published
               95.0% is on his own encoding

Usage:
    python -m experiments.relaxed_tm.classification.run_e4_baseline --encoding binary
    python -m experiments.relaxed_tm.classification.run_e4_baseline --encoding thermometer --runs 4 --no-save
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cakit.tm import format_multiclass_rules
from experiments.relaxed_tm.classification.baselines import Granmo2018Config, run_granmo2018_tm
from experiments.relaxed_tm.classification.data import IRIS_CLASSES
from experiments.relaxed_tm.classification.iris import ENCODINGS, iris_split
from experiments.relaxed_tm.common.train import summarize

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOURCE = ("https://github.com/cair/TsetlinMachine, MultiClassTsetlinMachine.pyx at commit b9a1491 "
          "(21 April 2018; see granmo2018/april2018/)")
KEEP_RULES_FOR_SEEDS = 20  # rules of the first seeds are stored; all runs store clause statistics


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def one_run(job):
    seed, encoding, config = job
    X_fit, y_fit, X_test, y_test, names = iris_split(encoding, seed)
    r = run_granmo2018_tm(X_fit, y_fit, X_test, y_test, config, seed)
    hard = r.pop("hard")
    out = {"seed": seed, **r, **hard.stats()}
    if seed < KEEP_RULES_FOR_SEEDS:
        out["rules"] = format_multiclass_rules(hard, IRIS_CLASSES, names)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoding", choices=ENCODINGS, required=True)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = Granmo2018Config(clauses_total=300, T=10, s=3.0, states=100, epochs=args.epochs,
                              eval_every=10, version="april2018", boost_true_positive_feedback=1)
    from experiments.relaxed_tm.classification.granmo2018 import april2018  # noqa: F401  (compile once, before the workers start)

    print(f"Standard TM (Granmo, April 2018 code), {args.encoding} encoding: {args.runs} runs x "
          f"{args.epochs} epochs, {args.workers} workers", flush=True)
    start = time.perf_counter()
    runs = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(one_run, [(s, args.encoding, config) for s in range(args.runs)]):
            runs.append(r)
            elapsed = time.perf_counter() - start
            eta = elapsed / len(runs) * (args.runs - len(runs))
            mean = 100 * np.mean([x["test"] for x in runs])
            print(f"  run {len(runs):4d}/{args.runs} (seed {r['seed']:3d}): test {r['test']:.4f}  "
                  f"train {r['train']:.4f}  running mean {mean:.2f}%  {r['train_time_s']:.1f}s  "
                  f"elapsed {fmt_time(elapsed)}  left ~{fmt_time(eta)}", flush=True)
    runs.sort(key=lambda r: r["seed"])

    summary = {
        "test_accuracy": summarize([r["test"] for r in runs]),
        "train_accuracy": summarize([r["train"] for r in runs]),
        "runs_at_100": int(sum(r["test"] == 1.0 for r in runs)),
        "hard_model_matches_implementation_all_runs": all(r["hard_matches_original"] for r in runs),
        "mean_train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
        "mean_clause_length": float(np.mean([r["mean_clause_length"] for r in runs])),
        "mean_empty_clauses": float(np.mean([r["n_empty"] for r in runs])),
    }
    t = summary["test_accuracy"]
    print(f"\nTest accuracy {t['mean']:.2f}% ± {t['ci95']:.2f} (5th-95th pct {t['p5']:.1f}-{t['p95']:.1f}, "
          f"min {t['min']:.1f}, max {t['max']:.1f}); hard model matches his code in all runs: "
          f"{summary['hard_model_matches_implementation_all_runs']}; wall time {fmt_time(time.perf_counter() - start)}",
          flush=True)
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"e4_standard_tm_{args.encoding}.json"
        out.write_text(json.dumps({"args": vars(args), "config": asdict(config), "implementation": SOURCE,
                                   "summary": summary, "runs": runs}, indent=2, ensure_ascii=False))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
