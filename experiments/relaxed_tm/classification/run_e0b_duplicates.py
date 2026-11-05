"""
E0b - Why do duplicate clauses form? (clean XOR, 4 clauses)

With the minimum of 4 clauses, E0 found that two clauses of the same team can
learn the same XOR case and leave another case uncovered (5/10 seeds). This
script tests three hypotheses, each as a grid of settings with many seeds:

  init    H1: the starting values matter. Start at the include/exclude
              boundary (w near 0, p near 0.5), as Granmo's TM starts its
              automata, instead of mostly excluded (w mean -1.5, p near 0.18).
  margin  H2: cross-entropy never stops rewarding an already covered case.
              A T-like margin stops the loss once the vote sum is far enough
              on the correct side, as the standard TM's target T does.
  tau     H3: the output steepness tau decides when a covered case stops
              pushing.
  speed   Follow-up to H1: if clauses that commit fast commit blindly, a
              smaller learning rate should help a start at the boundary
              (300 epochs, so slow runs can finish).
  followup    If the number of steps a weight needs to cross 0 matters, a
              larger learning rate should bring duplicates back at init -3.
              And: do the two best settings (init -3, tau 0.5) add up?

All other settings as in E0 (100 epochs, Adam lr 0.05, batch 32, b = 0).

Usage:
    python -m experiments.relaxed_tm.classification.run_e0b_duplicates --hypothesis init
    python -m experiments.relaxed_tm.classification.run_e0b_duplicates --hypothesis all --seeds 50
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import torch

from cakit.tm import RelaxedTM, format_rules, harden
from experiments.relaxed_tm.classification.data import clean_xor
from experiments.relaxed_tm.common.train import TrainConfig, set_seed, train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
XOR_RULE = {"+ : x1 ∧ ¬x2", "+ : ¬x1 ∧ x2", "− : x1 ∧ x2", "− : ¬x1 ∧ ¬x2"}
INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]

BASE = dict(init_mean=-1.5, init_std=1.0, tau=1.0, margin=None, lr=0.05, epochs=None)
GRIDS = {
    "init": [
        dict(BASE, init_mean=-3.0),
        dict(BASE),                                   # E0 setting
        dict(BASE, init_mean=-0.5),
        dict(BASE, init_mean=0.0),                    # at the boundary, wide spread
        dict(BASE, init_mean=0.0, init_std=0.1),      # at the boundary, like Granmo's TM
        dict(BASE, init_mean=-1.5, init_std=0.1),     # mostly excluded, little spread
    ],
    "margin": [
        dict(BASE),                                   # plain cross-entropy (E0)
        dict(BASE, margin=0.5),
        dict(BASE, margin=1.0),                       # satisfied once one clause votes for it
        dict(BASE, margin=2.0),
    ],
    "tau": [
        dict(BASE, tau=0.5),
        dict(BASE),                                   # E0 setting
        dict(BASE, tau=2.0),
        dict(BASE, tau=4.0),
    ],
    "speed": [
        dict(BASE, init_mean=0.0, init_std=0.1, lr=0.05, epochs=300),
        dict(BASE, init_mean=0.0, init_std=0.1, lr=0.01, epochs=300),
        dict(BASE, init_mean=0.0, init_std=0.1, lr=0.002, epochs=300),
        dict(BASE, lr=0.01, epochs=300),              # E0 start, slower learning
    ],
    "followup": [
        dict(BASE, init_mean=-3.0, lr=0.15),          # far start, 3x larger steps
        dict(BASE, init_mean=-3.0, tau=0.5),          # the two best settings combined
    ],
}


def one_run(job):
    torch.set_num_threads(1)
    setting, seed, clauses, epochs = job
    X, y = clean_xor(n_repeats=100)
    config = TrainConfig(epochs=setting["epochs"] or epochs, batch_size=32, lr=setting["lr"], margin=setting["margin"])
    set_seed(seed)
    model = RelaxedTM(2, clauses, tau=setting["tau"], init_mean=setting["init_mean"], init_std=setting["init_std"])
    history = train(model, X, y, config)
    hard = harden(model)
    rules = format_rules(hard)
    # hard vote sum for each of the four inputs: 0 means no clause covers it
    votes = hard.vote_sum(torch.tensor(INPUTS)).tolist()
    duplicated = any(n > 1 for n in Counter(rules).values())
    hard_acc = history["train_hard"]
    return {
        "seed": seed,
        "rules": rules,
        "exact_xor_rule": set(rules) == XOR_RULE,
        "duplicated_clause": duplicated if clauses == 4 else None,
        "uncovered_inputs": [list(i) for i, v in zip(INPUTS, votes) if v == 0],
        "final_hard_accuracy": hard_acc[-1],
        "epochs_to_hard_100": next((i + 1 for i, a in enumerate(hard_acc) if a == 1.0), None),
    }


def label(setting) -> str:
    m = "none" if setting["margin"] is None else setting["margin"]
    return (f"init N({setting['init_mean']}, {setting['init_std']}), tau {setting['tau']}, margin {m}, "
            f"lr {setting['lr']}, epochs {setting['epochs'] or 100}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hypothesis", choices=[*GRIDS, "all"], default="all")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--clauses", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--select", type=str, default=None,
                        help="comma-separated setting numbers (0-based) to run; results are merged into the saved file")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    names = list(GRIDS) if args.hypothesis == "all" else [args.hypothesis]
    for name in names:
        start = time.perf_counter()
        path = RESULTS_DIR / f"e0b_{name}_m{args.clauses}.json"
        grid = GRIDS[name]
        selected = range(len(grid)) if args.select is None else [int(i) for i in args.select.split(",")]
        out = {"hypothesis": name, "args": vars(args), "settings": [None] * len(grid)}
        if args.select is not None and path.exists():
            out["settings"] = json.loads(path.read_text())["settings"]
        print(f"\n== {name} ({args.seeds} seeds, {args.clauses} clauses)")
        for i in selected:
            setting = grid[i]
            jobs = [(setting, seed, args.clauses, args.epochs) for seed in range(args.seeds)]
            with Pool(args.workers) as pool:
                runs = pool.map(one_run, jobs)
            reached = sorted(r["epochs_to_hard_100"] for r in runs if r["epochs_to_hard_100"])
            summary = {
                "exact_xor_rule": sum(r["exact_xor_rule"] for r in runs),
                "hard_100": sum(r["final_hard_accuracy"] == 1.0 for r in runs),
                "duplicated_clause": sum(bool(r["duplicated_clause"]) for r in runs),
                "median_epochs_to_hard_100": reached[len(reached) // 2] if reached else None,
            }
            out["settings"][i] = {"setting": setting, "summary": summary, "runs": runs}
            print(f"  {label(setting):48s} exact rule {summary['exact_xor_rule']:2d}/{args.seeds}  "
                  f"duplicate {summary['duplicated_clause']:2d}  100% hard {summary['hard_100']:2d}  "
                  f"median epochs {summary['median_epochs_to_hard_100']}")
        print(f"  ({time.perf_counter() - start:.0f}s)")
        if not args.no_save:
            RESULTS_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(f"  saved {path.name}")


if __name__ == "__main__":
    main()
