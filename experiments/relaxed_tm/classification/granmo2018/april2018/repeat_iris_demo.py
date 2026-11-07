"""
E4, part 1 - Run Granmo's Binary Iris demo (BinaryIrisDemo.py) the way he ran
it: one Python process running his script, which itself loops over 1,000
ensembles (random 80/20 split, fresh Tsetlin Machine, 500 epochs each). No
seeding by us: NumPy seeds itself from the operating system, and C rand()
starts from its default seed. Prints one progress line per ensemble and saves
every ensemble's test and training accuracy with the statistics of his Table 6.

His script imports the module by the plain name MultiClassTsetlinMachine, as
does the E1 noisy XOR demo with the later version of the file. pyximport caches
compiled modules by name in ~/.pyxbld, so the subprocess gets its own HOME
(in the git-ignored data folder) to keep the two versions apart.

Usage, from the repository root:
    python -m experiments.relaxed_tm.classification.granmo2018.april2018.repeat_iris_demo
    python -m experiments.relaxed_tm.classification.granmo2018.april2018.repeat_iris_demo --ensembles 3 --no-save   # quick check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from experiments.relaxed_tm.classification.data import DATA_DIR, load_granmo_binary_iris
from experiments.relaxed_tm.common.train import summarize

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parents[1] / "results"
RESULT_NAME = "e4_granmo2018_demo_binary_iris.json"
ENSEMBLE_RE = re.compile(r"Ensemble accuracy: test ([0-9.]+) train ([0-9.]+)")


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ensembles", type=int, default=None,
                        help="quick check only: run a copy of his script with this ensemble_size "
                             "instead of his 1,000 (implies --no-save)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if args.ensembles is not None:
        args.no_save = True

    load_granmo_binary_iris()  # make sure his data file is downloaded into DATA_DIR
    script = HERE / "BinaryIrisDemo.py"
    n_total = 1000
    if args.ensembles is not None:
        src = script.read_text()
        assert src.count("ensemble_size = 1000") == 1
        script = DATA_DIR / "_BinaryIrisDemo_quick.py"
        script.write_text(src.replace("ensemble_size = 1000", f"ensemble_size = {args.ensembles}"))
        n_total = args.ensembles

    home = DATA_DIR / ".home_april2018"
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE="1", HOME=str(home),
               PYTHONUNBUFFERED="1")
    print(f"Granmo's Binary Iris demo, his way: {n_total} ensembles in one process "
          f"(compiles his Cython code first)", flush=True)

    start = time.perf_counter()
    runs = []
    proc = subprocess.Popen([sys.executable, str(script)], cwd=DATA_DIR, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    for line in proc.stdout:
        output.append(line)
        m = ENSEMBLE_RE.search(line)
        if m:
            runs.append({"ensemble": len(runs) + 1, "test": float(m.group(1)), "train": float(m.group(2))})
            elapsed = time.perf_counter() - start
            mean = 100 * sum(r["test"] for r in runs) / len(runs)
            eta = elapsed / len(runs) * (n_total - len(runs))
            print(f"  ensemble {len(runs):4d}/{n_total}: test {runs[-1]['test']:.4f}  "
                  f"running mean {mean:.2f}%  elapsed {fmt_time(elapsed)}  left ~{fmt_time(eta)}", flush=True)
    if proc.wait() != 0:
        sys.stderr.write("".join(output[-30:]))
        raise SystemExit(f"his demo exited with code {proc.returncode}")

    summary = {
        "test_accuracy": summarize([r["test"] for r in runs]),
        "train_accuracy": summarize([r["train"] for r in runs]),
    }
    t = summary["test_accuracy"]
    print(f"\nTest accuracy {t['mean']:.2f}% ± {t['ci95']:.2f} (5th-95th pct {t['p5']:.1f}-{t['p95']:.1f}, "
          f"min {t['min']:.1f}, max {t['max']:.1f}); published 95.0 ± 0.2; "
          f"wall time {fmt_time(time.perf_counter() - start)}", flush=True)
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / RESULT_NAME
        out.write_text(json.dumps({
            "how": "Granmo's BinaryIrisDemo.py (April 2018) with the April 2018 MultiClassTsetlinMachine.pyx, "
                   "run unchanged apart from Python 3 syntax and one added print line per ensemble; one "
                   "process, 1,000 ensembles, no seeding (NumPy seeds itself from the operating system; "
                   "C rand() starts from its default seed and runs on across ensembles)",
            "settings": {"clauses_total": 300, "T": 10, "s": 3.0, "states": 100, "epochs": 500,
                         "boost_true_positive_feedback": 1, "split": "random 80/20 per ensemble"},
            "platform": sys.platform, "wall_time_s": time.perf_counter() - start,
            "summary": summary, "runs": runs,
        }, indent=2))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
