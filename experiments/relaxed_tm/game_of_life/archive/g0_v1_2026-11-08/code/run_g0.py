"""
G0 - Learn Game of Life: relaxed TM vs standard TM on the 512-row table.

Data: all 512 neighbourhoods (9 bits) -> next state; training and evaluation
use the same 512 rows (the question is whether the rule can be represented
exactly). One epoch = one pass over the 512 rows in a new random order.

Relaxed TM: single-output RelaxedTM, 256 clauses (128 +, 128 -), learned
threshold b, Adam, mini-batches of 32, 1,000 epochs, model after the final
epoch. Standard TM: Granmo's 2018 Multi-Class TM with 2 classes (dead, alive),
the same 256 clauses in total (128 per class), 100 states per action,
1,000 epochs. His code has no single-output form; E1 used it the same way.

Modes:
  tune-relaxed   start mean x tau x learning rate (2 x 5 x 2), tuning seeds 1000-1019
  tune-standard  T x s (3 x 3), tuning seeds 1000-1019
  relaxed        chosen setting (from the tuning results unless given), seeds 0-99;
                 --loss squared for the squared-error variant
  standard       chosen T and s, seeds 0-99

A setting is chosen by its exact-rule rate (share of seeds with 512 of 512
rows right, hard); ties go to the setting that is exact after fewer epochs
(median), then to the higher mean hard accuracy.

Usage:
    python -m experiments.relaxed_tm.game_of_life.run_g0 --mode tune-relaxed
    python -m experiments.relaxed_tm.game_of_life.run_g0 --mode tune-standard
    python -m experiments.relaxed_tm.game_of_life.run_g0 --mode relaxed
    python -m experiments.relaxed_tm.game_of_life.run_g0 --mode relaxed --loss squared
    python -m experiments.relaxed_tm.game_of_life.run_g0 --mode standard
    python -m experiments.relaxed_tm.game_of_life.run_g0 --mode relaxed --runs 2 --epochs 50 --no-save
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cakit.tm import RelaxedTM, harden, undecided_fraction
from experiments.relaxed_tm.common.train import TrainConfig, evaluate, set_seed, summarize, train
from experiments.relaxed_tm.game_of_life.clauses import analyze_bank, merged_rules, pack_bank, wrong_rows
from experiments.relaxed_tm.game_of_life.data import N_INPUTS, gol_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CLAUSES = 256
EPOCHS = 1000
BATCH = 32
TUNE_SEEDS = list(range(1000, 1020))
TUNE_INIT_MEANS = [-3.0, -1.5]
TUNE_TAUS = [1.0, 3.0, 10.0, 30.0, 100.0]
TUNE_LRS = [0.01, 0.05]
TUNE_TS = [10, 30, 100]
TUNE_SS = [3.0, 5.0, 10.0]
STATES = 100
KEEP_RULES_FOR_SEEDS = 10


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def first_exact(curve: List[float]) -> Dict:
    """First epoch (1-based) with hard accuracy 1.0, and whether it stays exact afterwards."""
    hits = [i for i, a in enumerate(curve) if a == 1.0]
    if not hits:
        return {"first_exact_epoch": None, "stays_exact": False}
    first = hits[0]
    return {"first_exact_epoch": first + 1, "stays_exact": all(a == 1.0 for a in curve[first:])}


def exact_last(curve: List[float], n: int = 100) -> float:
    """Share of the last n logged epochs at which the hard model was exact."""
    tail = curve[-n:]
    return float(np.mean([a == 1.0 for a in tail])) if tail else 0.0


# ----------------------------------------------------------------------
# One run of each model
# ----------------------------------------------------------------------
def relaxed_run(job: Dict) -> Dict:
    torch.set_num_threads(1)
    X, y = gol_table()
    set_seed(job["seed"])
    model = RelaxedTM(N_INPUTS, job["clauses"], tau=job["tau"], learn_bias=True, init_mean=job["init_mean"])
    cfg = TrainConfig(epochs=job["epochs"], batch_size=BATCH, lr=job["lr"], log_every=1, loss=job["loss"])
    start = time.perf_counter()
    hist = train(model, X, y, cfg)
    train_time = time.perf_counter() - start

    ev = evaluate(model, X, y)
    hard = harden(model)
    wrong = wrong_rows(hard.predict)
    out = {
        "seed": job["seed"],
        "hard": ev["hard"], "soft": ev["soft"], "gap": ev["gap"],
        "exact": len(wrong) == 0,
        **first_exact(hist["train_hard"]),
        "exact_last100": exact_last(hist["train_hard"]),
        "wrong_rows": wrong,
        "undecided": undecided_fraction(model),
        "bias": float(model.bias),
        "train_time_s": train_time,
        "clauses": analyze_bank(hard, target=1),
    }
    if job.get("keep_model"):
        out["model"] = pack_bank(hard)
        out["hard_curve"] = hist["train_hard"]
        out["soft_curve"] = hist["train_soft"]
        out["loss_curve"] = hist["loss"]            # mean training loss per epoch
        out["undecided_curve"] = hist["undecided"]  # share of literals with 0.1 < p < 0.9, per epoch
    if job.get("keep_rules"):
        out["rules"] = merged_rules(hard)
    return out


def standard_run(job: Dict) -> Dict:
    from experiments.relaxed_tm.classification.baselines import Granmo2018Config, run_granmo2018_tm

    X, y = gol_table()
    cfg = Granmo2018Config(clauses_total=job["clauses"], T=job["T"], s=job["s"], states=STATES,
                           epochs=job["epochs"], eval_every=1, version="current")
    r = run_granmo2018_tm(X, y, X, y, cfg, job["seed"])
    hard = r["hard"]  # HardMultiClassTM, banks [dead, alive]; ties -> dead
    wrong = wrong_rows(hard.predict)
    out = {
        "seed": job["seed"],
        "hard": r["train"],   # his own evaluation on the 512 rows (train = test here)
        "exact": len(wrong) == 0,
        **first_exact(r["test_curve"]),
        "exact_last100": exact_last(r["test_curve"]),
        "wrong_rows": wrong,
        "hard_matches_original": r["hard_matches_original"],
        "train_time_s": r["train_time_s"],
        "clauses_dead_bank": analyze_bank(hard.banks[0], target=0),
        "clauses_alive_bank": analyze_bank(hard.banks[1], target=1),
    }
    if job.get("keep_model"):
        out["model"] = [pack_bank(b) for b in hard.banks]
        out["hard_curve"] = r["test_curve"]
    if job.get("keep_rules"):
        out["rules"] = {"dead": merged_rules(hard.banks[0]), "alive": merged_rules(hard.banks[1])}
    return out


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------
def _clause_means(runs: List[Dict], key: str) -> Dict:
    totals = [r[key]["total"] for r in runs]
    fields = ["active", "distinct", "duplicate_fraction", "pure_fraction", "distinct_in_reference",
              "mean_literals", "empty", "contradictory", "included_literals"]
    return {f: float(np.mean([t[f] for t in totals])) for f in fields}


def summarize_runs(runs: List[Dict], kind: str) -> Dict:
    exact = [r["exact"] for r in runs]
    epochs = [r["first_exact_epoch"] for r in runs if r["first_exact_epoch"] is not None]
    s = {
        "exact_rate": float(np.mean(exact)),
        "n_exact": int(sum(exact)),
        "n_runs": len(runs),
        "stays_exact": int(sum(r["stays_exact"] for r in runs)),
        "exact_last100_mean": float(np.mean([r["exact_last100"] for r in runs])),
        "ever_exact": int(sum(r["first_exact_epoch"] is not None for r in runs)),
        "hard": summarize([r["hard"] for r in runs]),
        "first_exact_epoch_median": float(np.median(epochs)) if epochs else None,
        "first_exact_epoch_max": int(max(epochs)) if epochs else None,
        "wrong_rows_mean": float(np.mean([len(r["wrong_rows"]) for r in runs])),
        "train_time_s_mean": float(np.mean([r["train_time_s"] for r in runs])),
    }
    if kind == "relaxed":
        s["soft"] = summarize([r["soft"] for r in runs])
        s["gap_mean"] = float(np.mean([r["gap"] for r in runs]))
        s["undecided_mean"] = float(np.mean([r["undecided"] for r in runs]))
        s["clauses"] = _clause_means(runs, "clauses")
    else:
        s["hard_matches_original"] = int(sum(r["hard_matches_original"] for r in runs))
        s["clauses_dead_bank"] = _clause_means(runs, "clauses_dead_bank")
        s["clauses_alive_bank"] = _clause_means(runs, "clauses_alive_bank")
    return s


def choose(settings: List[Dict]) -> Dict:
    def key(st):
        med = st["summary"]["first_exact_epoch_median"]
        return (-st["summary"]["exact_rate"], med if med is not None else float("inf"), -st["summary"]["hard"]["mean"])
    return min(settings, key=key)


def short(s: Dict) -> str:
    med = s["first_exact_epoch_median"]
    return (f"exact {s['n_exact']}/{s['n_runs']} (ever {s['ever_exact']})  hard {s['hard']['mean']:.2f}%  "
            f"median epochs to exact {med if med is not None else '-'}  "
            f"exact in last 100 epochs {100 * s['exact_last100_mean']:.0f}%")


# ----------------------------------------------------------------------
def run_pool(fn, jobs, workers):
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        out = []
        for i, r in enumerate(pool.imap(fn, jobs), 1):
            out.append(r)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)} runs, {fmt_time(el)} elapsed, ~{fmt_time(el / i * (len(jobs) - i))} left", flush=True)
    return out


def load_choice(name: str) -> Optional[Dict]:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())["chosen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=["tune-relaxed", "tune-standard", "relaxed", "standard"])
    ap.add_argument("--loss", choices=["bce", "squared"], default="bce")
    ap.add_argument("--clauses", type=int, default=CLAUSES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--runs", type=int, default=None, help="seeds 0..runs-1 (default 100); in tune modes: number of tuning seeds")
    ap.add_argument("--tau", type=float); ap.add_argument("--init-mean", type=float); ap.add_argument("--lr", type=float)
    ap.add_argument("--T", type=int); ap.add_argument("--s", type=float)
    ap.add_argument("--taus", type=str, help="tune-relaxed grid, e.g. 0.3,0.5,1")
    ap.add_argument("--lrs", type=str, help="tune-relaxed grid, e.g. 0.05,0.1,0.2")
    ap.add_argument("--init-means", type=str, help="tune-relaxed grid, e.g. -1.5,-0.5")
    ap.add_argument("--Ts", type=str, help="tune-standard grid, e.g. 100,300")
    ap.add_argument("--ss", type=str, help="tune-standard grid, e.g. 10,20,30")
    ap.add_argument("--out", type=str, help="results file name (default depends on the mode)")
    ap.add_argument("--tune-file", type=str, help="tuning results to take the setting from (main modes)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    if a.loss == "squared" and a.mode != "relaxed":
        ap.error("--loss squared is only for --mode relaxed")

    base = {"clauses": a.clauses, "epochs": a.epochs}
    t0 = time.perf_counter()

    if a.mode.startswith("tune"):
        seeds = TUNE_SEEDS[: a.runs] if a.runs else TUNE_SEEDS
        if a.mode == "tune-relaxed":
            lst = lambda v, d: [float(x) for x in v.split(",")] if v else d
            grid = [{"init_mean": m, "tau": t, "lr": l, "loss": "bce"}
                    for m, t, l in product(lst(a.init_means, TUNE_INIT_MEANS), lst(a.taus, TUNE_TAUS), lst(a.lrs, TUNE_LRS))]
            fn, kind, name = relaxed_run, "relaxed", "g0_tune_relaxed.json"
        else:
            grid = [{"T": int(T), "s": float(sv)} for T, sv in product(
                [int(x) for x in a.Ts.split(",")] if a.Ts else TUNE_TS,
                [float(x) for x in a.ss.split(",")] if a.ss else TUNE_SS)]
            fn, kind, name = standard_run, "standard", "g0_tune_standard.json"
        jobs = [{**base, **g, "seed": sd} for g in grid for sd in seeds]
        print(f"{a.mode}: {len(grid)} settings x {len(seeds)} seeds = {len(jobs)} runs, {a.workers} workers")
        runs = run_pool(fn, jobs, a.workers)
        settings = []
        for i, g in enumerate(grid):
            rs = runs[i * len(seeds):(i + 1) * len(seeds)]
            st = {"setting": g, "summary": summarize_runs(rs, kind), "runs": rs}
            settings.append(st)
            print(f"  {g}: {short(st['summary'])}")
        best = choose(settings)
        print(f"Chosen: {best['setting']}  ({short(best['summary'])})")
        result = {"mode": a.mode, **base, "batch": BATCH, "seeds": seeds,
                  "chosen": {**best["setting"]}, "settings": settings}
    else:
        n = a.runs or 100
        if a.mode == "relaxed":
            ch = load_choice(a.tune_file or "g0_tune_relaxed.json") or {}
            setting = {"init_mean": a.init_mean if a.init_mean is not None else ch.get("init_mean"),
                       "tau": a.tau if a.tau is not None else ch.get("tau"),
                       "lr": a.lr if a.lr is not None else ch.get("lr"), "loss": a.loss}
            fn, kind = relaxed_run, "relaxed"
            name = "g0_relaxed.json" if a.loss == "bce" else "g0_relaxed_squared.json"
        else:
            ch = load_choice(a.tune_file or "g0_tune_standard.json") or {}
            setting = {"T": a.T if a.T is not None else ch.get("T"), "s": a.s if a.s is not None else ch.get("s")}
            fn, kind, name = standard_run, "standard", "g0_standard.json"
        if any(v is None for v in setting.values()):
            ap.error(f"setting incomplete {setting}: run the tuning mode first or give the values")
        jobs = [{**base, **setting, "seed": sd, "keep_model": True, "keep_rules": sd < KEEP_RULES_FOR_SEEDS}
                for sd in range(n)]
        print(f"{a.mode} ({setting}): {n} seeds, {a.workers} workers")
        runs = run_pool(fn, jobs, a.workers)
        summary = summarize_runs(runs, kind)
        print(short(summary))
        result = {"mode": a.mode, **base, "batch": BATCH, "setting": setting, "summary": summary, "runs": runs}

    result["wall_time_s"] = time.perf_counter() - t0
    print(f"Done in {fmt_time(result['wall_time_s'])}")
    if a.out:
        name = a.out
    if not a.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / name).write_text(json.dumps(result, indent=1))
        print(f"Saved results/{name}")



if __name__ == "__main__":
    main()
