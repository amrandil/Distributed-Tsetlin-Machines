"""
E4 - Relaxed Tsetlin Machine on Iris (Granmo's Binary Iris protocol).

Model: MultiClassRelaxedTM, 3 classes x 100 clauses (50 positive, 50
negative) = Granmo's 300 clauses; predicted class = largest hard vote sum.
Adam, mini-batches of 32, 500 epochs, test accuracy after the final epoch.
Splits: seed k -> iris.iris_split(encoding, k), the same flowers as the
standard-TM runs (run_e4_baseline.py).

Two modes:
  tune   Settings chosen on validation data only: for each of 20 splits, 24 of
         the 120 training flowers are held out and the model trains on the
         other 96; the 30 test flowers are not used. Grid: tau x start (w mean)
         x learning rate. The setting with the highest mean validation
         accuracy (hard) is chosen.
  final  Chosen setting (read from the tuning results unless given), all 120
         training flowers, 1,000 seeds, hard test accuracy after the final epoch.

Usage:
    python -m experiments.relaxed_tm.classification.run_e4_relaxed --encoding binary --mode tune
    python -m experiments.relaxed_tm.classification.run_e4_relaxed --encoding binary --mode final
    python -m experiments.relaxed_tm.classification.run_e4_relaxed --encoding binary --mode final --runs 4 --epochs 20 --no-save
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from cakit.tm import MultiClassRelaxedTM, format_multiclass_rules, harden, undecided_fraction
from experiments.relaxed_tm.classification.data import IRIS_CLASSES
from experiments.relaxed_tm.classification.iris import ENCODINGS, iris_split
from experiments.relaxed_tm.common.train import TrainConfig, evaluate, set_seed, summarize, train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_INPUTS, N_CLASSES, CLAUSES_PER_CLASS = 16, 3, 100
TUNE_TAUS = [1.0, 3.0, 10.0, 30.0]
TUNE_INIT_MEANS = [-3.0, -1.5]
TUNE_LRS = [0.01, 0.05]
KEEP_RULES_FOR_SEEDS = 20  # final mode: rules of the first seeds are stored


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def one_run(job):
    torch.set_num_threads(1)
    encoding, init_mean, tau, lr, seed, epochs, batch, validation = job
    X_fit, y_fit, X_eval, y_eval, names = iris_split(encoding, seed, validation=validation)
    set_seed(seed)
    model = MultiClassRelaxedTM(N_INPUTS, N_CLASSES, CLAUSES_PER_CLASS, tau=tau, init_mean=init_mean)
    config = TrainConfig(epochs=epochs, batch_size=batch, lr=lr, log_every=10)
    start = time.perf_counter()
    history = train(model, X_fit, y_fit, config, X_val=X_eval, y_val=y_eval)
    train_time = time.perf_counter() - start
    ev = evaluate(model, X_eval, y_eval)
    fit = evaluate(model, X_fit, y_fit)
    hard = harden(model)
    out = {
        "seed": seed,
        "eval_hard": ev["hard"], "eval_soft": ev["soft"], "gap": ev["gap"],
        "fit_hard": fit["hard"], "fit_soft": fit["soft"],
        "undecided": undecided_fraction(model),
        "train_time_s": train_time,
        # after epochs 1, 11, 21, ... and the last epoch, on the evaluation data (validation in tune,
        # test in final): for learning curves only, never used to pick a model
        "curve": {"eval_hard": history["val_hard"], "eval_soft": history["val_soft"],
                  "fit_hard": history["train_hard"], "undecided": history["undecided"]},
        "loss": history["loss"],
        **hard.stats(),
    }
    if not validation and seed < KEEP_RULES_FOR_SEEDS:
        out["rules"] = format_multiclass_rules(hard, IRIS_CLASSES, names)
    return out


def summarize_runs(runs) -> dict:
    return {
        "eval_hard": summarize([r["eval_hard"] for r in runs]),
        "eval_soft": summarize([r["eval_soft"] for r in runs]),
        "gap_mean_pp": float(100 * np.mean([r["gap"] for r in runs])),
        "fit_hard": summarize([r["fit_hard"] for r in runs]),
        "runs_at_100": int(sum(r["eval_hard"] == 1.0 for r in runs)),
        "mean_undecided": float(np.mean([r["undecided"] for r in runs])),
        "mean_clause_length": float(np.mean([r["mean_clause_length"] for r in runs])),
        "mean_empty_clauses": float(np.mean([r["n_empty"] for r in runs])),
        "mean_train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
    }


def choose_setting(tune_path: Path) -> dict:
    """Highest mean validation accuracy (hard); ties broken by soft accuracy."""
    settings = json.loads(tune_path.read_text())["settings"]
    best = max(settings, key=lambda st: (round(st["summary"]["eval_hard"]["mean"], 6),
                                         st["summary"]["eval_soft"]["mean"]))
    return {k: best[k] for k in ("init_mean", "tau", "lr")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoding", choices=ENCODINGS, required=True)
    parser.add_argument("--mode", choices=["tune", "final"], required=True)
    parser.add_argument("--init-mean", type=float, default=None, help="final mode; default: from the tuning results")
    parser.add_argument("--tau", type=float, default=None, help="final mode; default: from the tuning results")
    parser.add_argument("--lr", type=float, default=None, help="final mode; default: from the tuning results")
    parser.add_argument("--runs", type=int, default=1000, help="final mode: seeds 0..runs-1")
    parser.add_argument("--splits", type=int, default=20, help="tune mode: splits per setting")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tune-taus", type=float, nargs="+", default=TUNE_TAUS)
    parser.add_argument("--tune-init-means", type=float, nargs="+", default=TUNE_INIT_MEANS)
    parser.add_argument("--tune-lrs", type=float, nargs="+", default=TUNE_LRS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    tune_path = RESULTS_DIR / f"e4_relaxed_tune_{args.encoding}.json"
    if args.mode == "tune":
        settings = [dict(init_mean=m, tau=t, lr=l)
                    for m, t, l in product(args.tune_init_means, args.tune_taus, args.tune_lrs)]
        seeds, validation = list(range(args.splits)), True
    else:
        given = {"init_mean": args.init_mean, "tau": args.tau, "lr": args.lr}
        if None in given.values():
            if not tune_path.exists():
                parser.error(f"final mode needs --init-mean, --tau and --lr, or the tuning results {tune_path.name}")
            chosen = choose_setting(tune_path)
            given = {k: v if v is not None else chosen[k] for k, v in given.items()}
            print(f"Setting from the tuning pass ({tune_path.name}): {given}", flush=True)
        settings = [given]
        seeds, validation = list(range(args.runs)), False

    what = "validation (24 of the training flowers)" if validation else "test (30 flowers)"
    n_total = len(settings) * len(seeds)
    print(f"Relaxed TM, {args.encoding} encoding, {args.mode}: {len(settings)} setting(s) x {len(seeds)} "
          f"{'splits' if validation else 'runs'} x {args.epochs} epochs, accuracy on {what}, "
          f"{args.workers} workers", flush=True)

    out = {"encoding": args.encoding, "mode": args.mode, "args": vars(args),
           "model": {"n_inputs": N_INPUTS, "n_classes": N_CLASSES, "clauses_per_class": CLAUSES_PER_CLASS},
           "train_config": asdict(TrainConfig(epochs=args.epochs, batch_size=args.batch_size, log_every=10)),
           "settings": []}
    start = time.perf_counter()
    done = 0
    with Pool(args.workers) as pool:
        for st in settings:
            jobs = [(args.encoding, st["init_mean"], st["tau"], st["lr"], s, args.epochs, args.batch_size, validation)
                    for s in seeds]
            runs = []
            for r in pool.imap_unordered(one_run, jobs):
                runs.append(r)
                done += 1
                if not validation:
                    elapsed = time.perf_counter() - start
                    mean = 100 * np.mean([x["eval_hard"] for x in runs])
                    print(f"  run {done:4d}/{n_total} (seed {r['seed']:3d}): test hard {r['eval_hard']:.4f}  "
                          f"soft {r['eval_soft']:.4f}  running mean {mean:.2f}%  {r['train_time_s']:.1f}s  "
                          f"elapsed {fmt_time(elapsed)}  left ~{fmt_time(elapsed / done * (n_total - done))}",
                          flush=True)
            runs.sort(key=lambda r: r["seed"])
            summary = summarize_runs(runs)
            out["settings"].append({**st, "summary": summary, "runs": runs})
            h = summary["eval_hard"]
            elapsed = time.perf_counter() - start
            print(f"  [{len(out['settings'])}/{len(settings)}] start {st['init_mean']:>4}, tau {st['tau']:>4}, "
                  f"lr {st['lr']:<5}: hard {h['mean']:.2f}% ± {h['ci95']:.2f}  soft {summary['eval_soft']['mean']:.2f}%  "
                  f"gap {summary['gap_mean_pp']:.2f} pp  elapsed {fmt_time(elapsed)}  "
                  f"left ~{fmt_time(elapsed / done * (n_total - done))}", flush=True)

    if args.mode == "tune":
        best = max(out["settings"], key=lambda st: (round(st["summary"]["eval_hard"]["mean"], 6),
                                                    st["summary"]["eval_soft"]["mean"]))
        print(f"\nBest on validation: start {best['init_mean']}, tau {best['tau']}, lr {best['lr']} "
              f"({best['summary']['eval_hard']['mean']:.2f}%)", flush=True)
    print(f"Wall time {fmt_time(time.perf_counter() - start)}", flush=True)
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = tune_path if args.mode == "tune" else RESULTS_DIR / f"e4_relaxed_final_{args.encoding}.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
