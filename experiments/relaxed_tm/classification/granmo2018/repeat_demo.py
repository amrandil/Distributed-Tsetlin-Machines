"""
Run Granmo's noisy XOR demo (NoisyXORDemo.py) the way he ran it: each
replication is a fresh Python process running his script unchanged, with no
seeding by us. Collects the test and training accuracy each run prints and
summarises them with the statistics of his Table 15.

Usage, from the repository root:
    python -m experiments.relaxed_tm.classification.granmo2018.repeat_demo --runs 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.relaxed_tm.classification.data import DATA_DIR, load_noisy_xor
from experiments.relaxed_tm.common.train import summarize

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
TEST_RE = re.compile(r"Accuracy on test data \(no noise\): ([0-9.]+)")
TRAIN_RE = re.compile(r"Accuracy on training data \(40% noise\): ([0-9.]+)")


def one_replication(i: int) -> dict:
    # His script reads the data files from the current folder and imports
    # MultiClassTsetlinMachine.pyx (compiled by pyximport) from its own folder.
    env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(
        [sys.executable, str(HERE / "NoisyXORDemo.py")],
        cwd=DATA_DIR, env=env, capture_output=True, text=True, check=True,
    ).stdout
    return {
        "replication": i,
        "test": float(TEST_RE.search(out).group(1)),
        "train": float(TRAIN_RE.search(out).group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    load_noisy_xor()  # make sure his data files are downloaded into DATA_DIR
    one_replication(-1)  # compile his .pyx once before running in parallel

    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        runs = list(pool.map(one_replication, range(args.runs)))
    for r in runs:
        print(f"replication {r['replication']:3d}: test {r['test']:.4f}  train {r['train']:.4f}")

    summary = {
        "test_accuracy": summarize([r["test"] for r in runs]),
        "train_accuracy": summarize([r["train"] for r in runs]),
        "runs_at_or_above_99": int(sum(r["test"] >= 0.99 for r in runs)),
    }
    t = summary["test_accuracy"]
    print(
        f"\nTest accuracy {t['mean']:.2f}% ± {t['ci95']:.2f} "
        f"(5th-95th pct {t['p5']:.2f}-{t['p95']:.2f}, min {t['min']:.2f}, max {t['max']:.2f}); "
        f"{summary['runs_at_or_above_99']}/{args.runs} runs ≥ 99%; wall time {time.perf_counter() - start:.0f}s"
    )
    if not args.no_save:
        out = RESULTS_DIR / "e1_granmo2018_demo_noisy_xor.json"
        out.write_text(json.dumps({
            "how": "Granmo's NoisyXORDemo.py run unchanged (Python 3 print only), one process per "
                   "replication, no seeding (C rand() starts from its default seed in every process; "
                   "NumPy seeds itself from the operating system)",
            "platform": sys.platform,
            "args": vars(args), "summary": summary, "runs": runs,
        }, indent=2))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
