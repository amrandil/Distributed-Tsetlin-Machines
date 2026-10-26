"""
E0 - Sanity check: clean 2-bit XOR, no noise.

Question: does the relaxed TM learn XOR at all, and does the hardened model
recover the XOR rule? Expect 100% hard accuracy within a few epochs.

Usage:
    python -m experiments.relaxed_tm.classification.run_e0
    python -m experiments.relaxed_tm.classification.run_e0 --clauses 4 --seeds 10 --epochs 100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cakit.tm import RelaxedTM, format_rules, harden
from experiments.relaxed_tm.classification.data import clean_xor
from experiments.relaxed_tm.common.train import TrainConfig, evaluate, set_seed, summarize, train

XOR_RULE = {"+ : x1 ∧ ¬x2", "+ : ¬x1 ∧ x2", "− : x1 ∧ x2", "− : ¬x1 ∧ ¬x2"}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clauses", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init-mean", type=float, default=-1.5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    X, y = clean_xor(n_repeats=100)
    config = TrainConfig(epochs=args.epochs, batch_size=32, lr=args.lr)

    runs = []
    for seed in range(args.seeds):
        set_seed(seed)
        model = RelaxedTM(2, args.clauses, tau=args.tau, init_mean=args.init_mean)
        history = train(model, X, y, config)
        metrics = evaluate(model, X, y)
        hard = harden(model)
        rules = format_rules(hard)
        hard_epochs = history["train_hard"]
        # Number of epochs until hard accuracy first reached 100% (1 = after the first epoch).
        epochs_to_100 = next((i + 1 for i, a in enumerate(hard_epochs) if a == 1.0), None)
        runs.append(
            {
                "seed": seed,
                **metrics,
                "rules": rules,
                "exact_xor_rule": set(rules) == XOR_RULE,
                "epochs_to_hard_100": epochs_to_100,
                **hard.stats(),
            }
        )
        print(
            f"seed {seed:2d}: hard {metrics['hard']:.3f}  soft {metrics['soft']:.3f}  "
            f"100% hard after {epochs_to_100} epochs  exact rule: {runs[-1]['exact_xor_rule']}"
        )

    print("\nRules learned with seed 0:")
    for line in runs[0]["rules"]:
        print("  " + line)

    summary = {
        "hard_accuracy": summarize([r["hard"] for r in runs]),
        "soft_accuracy": summarize([r["soft"] for r in runs]),
        "seeds_at_100_hard": sum(r["hard"] == 1.0 for r in runs),
        "seeds_with_exact_rule": sum(r["exact_xor_rule"] for r in runs),
    }
    print(
        f"\nHard accuracy {summary['hard_accuracy']['mean']:.1f}% "
        f"(± {summary['hard_accuracy']['ci95']:.1f}); "
        f"{summary['seeds_at_100_hard']}/{args.seeds} seeds at 100%; "
        f"{summary['seeds_with_exact_rule']}/{args.seeds} with the exact XOR rule."
    )

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"e0_clean_xor_m{args.clauses}.json"
        out.write_text(
            json.dumps(
                {"args": vars(args), "train_config": asdict(config), "summary": summary, "runs": runs},
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Saved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
