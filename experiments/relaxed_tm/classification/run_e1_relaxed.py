"""
E1, part 2 - Relaxed Tsetlin Machine on noisy XOR (Granmo's data and protocol).

Two structures, both with 20 clauses:
  2x10   MultiClassRelaxedTM: 2 classes x 10 clauses, as Granmo's Multi-Class TM
  1x20   RelaxedTM: one bank of 20 clauses, output 1 if the vote sum >= 0

Two modes:
  tune   Settings chosen on a validation split: 1,000 of the 5,000 training
         examples are held out (fixed split), the model trains on the other
         4,000. The test set is not used. Grid: start (w mean) x tau.
  final  Chosen setting, all 5,000 training examples, 100 seeds, test accuracy
         after the final epoch (Granmo's protocol).

Usage:
    python -m experiments.relaxed_tm.classification.run_e1_relaxed --structure 2x10 --mode tune
    python -m experiments.relaxed_tm.classification.run_e1_relaxed --structure 2x10 --mode final --init-mean -3 --tau 0.5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from cakit.tm import HardMultiClassTM, MultiClassRelaxedTM, RelaxedTM, harden, undecided_fraction
from experiments.relaxed_tm.classification.data import load_noisy_xor
from experiments.relaxed_tm.classification.xor_rules import check_xor_rule
from experiments.relaxed_tm.common.train import TrainConfig, evaluate, set_seed, summarize, train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NOISE_BIT = re.compile(r"x(1[0-2]|[3-9])\b")
TUNE_INIT_MEANS = [-3.0, -1.5]
TUNE_TAUS = [0.5, 1.0, 3.0, 5.0, 15.0]


def make_model(structure: str, init_mean: float, tau: float):
    if structure == "2x10":
        return MultiClassRelaxedTM(12, 2, 10, tau=tau, init_mean=init_mean)
    return RelaxedTM(12, 20, tau=tau, init_mean=init_mean)


def rule_measures(hard) -> dict:
    """XOR-rule checks on the hardened model (both banks for 2x10)."""
    if isinstance(hard, HardMultiClassTM):
        banks = {"y1": check_xor_rule(hard.banks[1], target=1), "y0": check_xor_rule(hard.banks[0], target=0)}
    else:
        banks = {"y1": check_xor_rule(hard, target=1)}
    lines = [l for b in banks.values() for l in b["rules"]]
    return {
        "rules": {k: b["rules"] for k, b in banks.items()},
        "xor_core_any_bank": any(b["core_present"] for b in banks.values()),
        "xor_core_all_banks": all(b["core_present"] for b in banks.values()),
        "exact_all_banks": all(b["exact"] for b in banks.values()),
        "extra_clauses": sum(b["n_extra"] for b in banks.values()),
        "noise_bit_clauses": sum(bool(NOISE_BIT.search(l)) for l in lines),
        **hard.stats(),
    }


def one_run(job):
    torch.set_num_threads(1)
    structure, init_mean, tau, seed, epochs, lr, batch, data = job
    X_fit, y_fit, X_eval, y_eval = data
    set_seed(seed)
    model = make_model(structure, init_mean, tau)
    config = TrainConfig(epochs=epochs, batch_size=batch, lr=lr, log_every=10)
    start = time.perf_counter()
    history = train(model, X_fit, y_fit, config, X_val=X_eval, y_val=y_eval)
    train_time = time.perf_counter() - start
    ev = evaluate(model, X_eval, y_eval)
    fit = evaluate(model, X_fit, y_fit)
    return {
        "seed": seed,
        "eval_hard": ev["hard"], "eval_soft": ev["soft"], "gap": ev["gap"],
        "fit_hard": fit["hard"],
        "undecided": undecided_fraction(model),
        "train_time_s": train_time,
        # after epochs 1, 11, 21, ..., 191 and 200, on the evaluation data (validation in tune, test in final):
        # for learning curves only, never used to pick a model
        "curve": {"eval_hard": history["val_hard"], "eval_soft": history["val_soft"], "undecided": history["undecided"]},
        "loss": history["loss"],  # training loss, every epoch
        **rule_measures(harden(model)),
    }


def summarize_runs(runs) -> dict:
    return {
        "eval_hard": summarize([r["eval_hard"] for r in runs]),
        "eval_soft": summarize([r["eval_soft"] for r in runs]),
        "gap_mean_pp": float(100 * np.mean([r["gap"] for r in runs])),
        "fit_hard": summarize([r["fit_hard"] for r in runs]),
        "runs_at_or_above_99": int(sum(r["eval_hard"] >= 0.99 for r in runs)),
        "xor_core_any_bank": int(sum(r["xor_core_any_bank"] for r in runs)),
        "xor_core_all_banks": int(sum(r["xor_core_all_banks"] for r in runs)),
        "exact_all_banks": int(sum(r["exact_all_banks"] for r in runs)),
        "mean_noise_bit_clauses": float(np.mean([r["noise_bit_clauses"] for r in runs])),
        "runs_without_noise_bits": int(sum(r["noise_bit_clauses"] == 0 for r in runs)),
        "mean_undecided": float(np.mean([r["undecided"] for r in runs])),
        "mean_train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--structure", choices=["2x10", "1x20"], required=True)
    parser.add_argument("--mode", choices=["tune", "final"], required=True)
    parser.add_argument("--init-mean", type=float, default=None, help="final mode: chosen start")
    parser.add_argument("--tau", type=float, default=None, help="final mode: chosen tau")
    parser.add_argument("--seeds", type=int, default=None, help="default: 5 (tune), 100 (final)")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="final mode: first seed; runs are appended to an existing results file "
                             "for the same setting, so 100 seeds can be run in parts")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tune-init-means", type=float, nargs="+", default=TUNE_INIT_MEANS)
    parser.add_argument("--tune-taus", type=float, nargs="+", default=TUNE_TAUS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_noisy_xor()
    start = time.perf_counter()
    if args.mode == "tune":
        seeds = args.seeds or 5
        perm = np.random.default_rng(0).permutation(len(X_train))  # fixed split
        fit_idx, val_idx = perm[1000:], perm[:1000]
        data = (X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx])
        settings = list(product(args.tune_init_means, args.tune_taus))
    else:
        if args.init_mean is None or args.tau is None:
            parser.error("final mode needs --init-mean and --tau (chosen in the tuning pass)")
        seeds = args.seeds or 100
        data = (X_train, y_train, X_test, y_test)
        settings = [(args.init_mean, args.tau)]

    out = {"structure": args.structure, "mode": args.mode, "args": vars(args),
           "train_config": asdict(TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, log_every=10)),
           "settings": []}
    what = "validation (noisy labels)" if args.mode == "tune" else "test"
    print(f"{args.structure}, {args.mode}: {seeds} seeds per setting; accuracy on {what}", flush=True)
    for init_mean, tau in settings:
        seed_list = range(args.seed_start, args.seed_start + seeds) if args.mode == "final" else range(seeds)
        jobs = [(args.structure, init_mean, tau, s, args.epochs, args.lr, args.batch_size, data) for s in seed_list]
        runs = []
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(one_run, jobs):
                runs.append(r)
                if args.mode == "final":
                    print(f"    run {len(runs):3d}/{len(jobs)} (seed {r['seed']:3d}): test hard {r['eval_hard']:.4f}  "
                          f"soft {r['eval_soft']:.4f}  XOR clauses {'yes' if r['xor_core_any_bank'] else 'no '}  "
                          f"noise-bit clauses {r['noise_bit_clauses']}  {r['train_time_s']:.0f}s", flush=True)
        runs.sort(key=lambda r: r["seed"])
        final_path = RESULTS_DIR / f"e1_relaxed_final_{args.structure}.json"
        if args.mode == "final" and not args.no_save and final_path.exists():
            old = json.loads(final_path.read_text())["settings"][0]
            if (old["init_mean"], old["tau"]) == (init_mean, tau):
                new_seeds = {r["seed"] for r in runs}
                runs = sorted([r for r in old["runs"] if r["seed"] not in new_seeds] + runs, key=lambda r: r["seed"])
        summary = summarize_runs(runs)
        out["settings"].append({"init_mean": init_mean, "tau": tau, "summary": summary, "runs": runs})
        h = summary["eval_hard"]
        print(f"  [{len(runs)} runs] start {init_mean:>4}, tau {tau:>4}:  hard {h['mean']:.2f}% ± {h['ci95']:.2f} (min {h['min']:.2f})  "
              f"soft {summary['eval_soft']['mean']:.2f}%  gap {summary['gap_mean_pp']:.2f} pp  "
              f"XOR clauses {summary['xor_core_any_bank']}/{seeds}  noise-bit clauses {summary['mean_noise_bit_clauses']:.1f}",
              flush=True)
    print(f"({time.perf_counter() - start:.0f}s)")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        name = f"e1_relaxed_{args.mode}_{args.structure}.json"
        path = RESULTS_DIR / name
        if args.mode == "tune" and path.exists():
            # merge: keep settings run earlier, replace those run now
            done = {(st["init_mean"], st["tau"]) for st in out["settings"]}
            old = [st for st in json.loads(path.read_text())["settings"] if (st["init_mean"], st["tau"]) not in done]
            out["settings"] = sorted(old + out["settings"], key=lambda st: (-st["init_mean"], st["tau"]))
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Saved {name}")


if __name__ == "__main__":
    main()
