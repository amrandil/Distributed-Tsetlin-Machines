"""
G0 - Architectures at 744 clauses per TM (PLAN.md, sections 4-6).

Four models, the same data, epochs, seeds and clauses per TM:

    --model relaxed  --arch single   RelaxedTM, 744 clauses, threshold b = 0
    --model relaxed  --arch multi    MultiClassRelaxedTM, 2 classes x 744, no threshold
    --model standard --arch single   Granmo 2018 TsetlinMachine.pyx, 744 clauses
    --model standard --arch multi    Granmo 2018 MultiClassTsetlinMachine.pyx, 2 x 744 (1,488 total)

Data: all 512 neighbourhoods -> next state; training and evaluation on the
same 512 rows. One epoch = one pass over the 512 rows in a new random order;
1,000 epochs; the model after the final epoch is reported. Multi-class class
order [dead, alive]; ties go to dead (as Granmo's code).

Modes:
  tune      the grid of PLAN.md section 5, tuning seeds 1000-1019
  extend    one extension pass, only if the chosen setting is on a grid edge:
            one step beyond that edge, on that axis (crossed with all values of
            the other axes)
  main      the chosen setting (best over tune + extend), seeds 0-99
  ablation  --ablation bias (relaxed: learned threshold), squared (relaxed:
            squared error), boost (standard multi: boost_true_positive_feedback,
            which needs Granmo's April 2018 Multi-Class code); seeds 0-99

Selection rule (all models): highest exact-rule rate on the tuning seeds;
ties -> fewer median epochs to exact -> higher mean hard accuracy.

Usage:
    python -m experiments.relaxed_tm.game_of_life.run_g0 --model relaxed --arch single --mode tune
    python -m experiments.relaxed_tm.game_of_life.run_g0 --model relaxed --arch single --mode extend
    python -m experiments.relaxed_tm.game_of_life.run_g0 --model relaxed --arch single --mode main
    python -m experiments.relaxed_tm.game_of_life.run_g0 --model relaxed --arch single --mode ablation --ablation bias
    python -m experiments.relaxed_tm.game_of_life.run_g0 --model standard --arch multi --mode main --runs 2 --epochs 20 --no-save
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

from cakit.tm import MultiClassRelaxedTM, RelaxedTM, harden, undecided_fraction
from experiments.relaxed_tm.common.train import TrainConfig, evaluate, set_seed, summarize, train
from experiments.relaxed_tm.game_of_life.clauses import (
    analyze_bank, analyze_multiclass, merged_rules, pack_bank, wrong_rows,
)
from experiments.relaxed_tm.game_of_life.data import N_INPUTS, gol_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CLAUSES_PER_TM = 744
EPOCHS = 1000
BATCH = 32
STATES = 100
TUNE_SEEDS = list(range(1000, 1020))
MAIN_SEEDS = 100
KEEP_RULES_FOR_SEEDS = 10

# PLAN.md section 5: grids fixed before any run
GRIDS = {
    "relaxed": {"init_mean": [-3.0, -1.5, -0.5], "tau": [0.3, 1.0, 3.0, 10.0], "lr": [0.01, 0.05, 0.2]},
    "standard": {"T": [10, 30, 100, 300], "s": [3.0, 10.0, 30.0, 100.0]},
}
# one step beyond each edge, used only by --mode extend
EXTEND = {
    "init_mean": {"low": -4.5, "high": 0.0},
    "tau": {"low": 0.1, "high": 30.0},
    "lr": {"low": 0.002, "high": 0.5},
    "T": {"low": 3, "high": 1000},
    "s": {"low": 1.5, "high": 300.0},
}


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def first_exact(curve: List[float]) -> Dict:
    hits = [i for i, a in enumerate(curve) if a == 1.0]
    if not hits:
        return {"first_exact_epoch": None, "stays_exact": False}
    return {"first_exact_epoch": hits[0] + 1, "stays_exact": all(a == 1.0 for a in curve[hits[0]:])}


def exact_last(curve: List[float], n: int = 100) -> float:
    tail = curve[-n:]
    return float(np.mean([a == 1.0 for a in tail])) if tail else 0.0


# ----------------------------------------------------------------------
# One run of each model
# ----------------------------------------------------------------------
def relaxed_run(job: Dict) -> Dict:
    torch.set_num_threads(1)
    X, y = gol_table()
    set_seed(job["seed"])
    learn_bias = job.get("learn_bias", False)
    if job["arch"] == "single":
        model = RelaxedTM(N_INPUTS, job["clauses"], tau=job["tau"], learn_bias=learn_bias, init_mean=job["init_mean"])
    else:
        model = MultiClassRelaxedTM(N_INPUTS, 2, job["clauses"], tau=job["tau"], learn_bias=learn_bias,
                                    init_mean=job["init_mean"])
    cfg = TrainConfig(epochs=job["epochs"], batch_size=BATCH, lr=job["lr"], log_every=1, loss=job.get("loss", "bce"))
    start = time.perf_counter()
    hist = train(model, X, y, cfg)
    train_time = time.perf_counter() - start

    ev = evaluate(model, X, y)
    hard = harden(model)
    wrong = wrong_rows(hard.predict)
    if job["arch"] == "single":
        clauses = analyze_bank(hard, target=1)
        packed = pack_bank(hard)
        bias = float(model.bias)
    else:
        clauses = analyze_multiclass(hard.banks)
        packed = [pack_bank(b) for b in hard.banks]
        bias = [float(b.bias) for b in model.banks]
    out = {
        "seed": job["seed"],
        "hard": ev["hard"], "soft": ev["soft"], "gap": ev["gap"],
        "exact": len(wrong) == 0,
        **first_exact(hist["train_hard"]),
        "exact_last100": exact_last(hist["train_hard"]),
        "wrong_rows": wrong,
        "undecided": undecided_fraction(model),
        "bias": bias,
        "train_time_s": train_time,
        "clauses": clauses,
    }
    if job.get("keep_model"):
        out.update(model=packed, hard_curve=hist["train_hard"], soft_curve=hist["train_soft"],
                   loss_curve=hist["loss"], undecided_curve=hist["undecided"])
    if job.get("keep_rules"):
        out["rules"] = merged_rules(hard) if job["arch"] == "single" else {
            "dead": merged_rules(hard.banks[0]), "alive": merged_rules(hard.banks[1])}
    return out


def standard_run(job: Dict) -> Dict:
    from experiments.relaxed_tm.classification.baselines import (
        Granmo2018Config, run_granmo2018_single_tm, run_granmo2018_tm,
    )

    X, y = gol_table()
    boost = int(job.get("boost", 0))
    if job["arch"] == "single":
        cfg = Granmo2018Config(clauses_total=job["clauses"], T=job["T"], s=job["s"], states=STATES,
                               epochs=job["epochs"], eval_every=1)
        r = run_granmo2018_single_tm(X, y, X, y, cfg, job["seed"])
        hard = r["hard"]
        wrong = [int(i) for i in np.flatnonzero(r["own_pred_test"] != y)]  # his own prediction rule
        clauses = analyze_bank(hard, target=1)
        packed = pack_bank(hard)
    else:
        # his code takes the total and splits it over the classes
        cfg = Granmo2018Config(clauses_total=2 * job["clauses"], T=job["T"], s=job["s"], states=STATES,
                               epochs=job["epochs"], eval_every=1,
                               version="april2018" if boost else "current", boost_true_positive_feedback=boost)
        r = run_granmo2018_tm(X, y, X, y, cfg, job["seed"])
        hard = r["hard"]   # HardMultiClassTM [dead, alive]; ties -> dead, as his code
        wrong = wrong_rows(hard.predict)
        clauses = analyze_multiclass(hard.banks)
        packed = [pack_bank(b) for b in hard.banks]
    out = {
        "seed": job["seed"],
        "hard": r["train"],
        "exact": len(wrong) == 0,
        **first_exact(r["test_curve"]),
        "exact_last100": exact_last(r["test_curve"]),
        "wrong_rows": wrong,
        "hard_matches_original": r["hard_matches_original"],
        "train_time_s": r["train_time_s"],
        "clauses": clauses,
    }
    if job.get("keep_model"):
        out.update(model=packed, hard_curve=r["test_curve"])
    if job.get("keep_rules"):
        out["rules"] = merged_rules(hard) if job["arch"] == "single" else {
            "dead": merged_rules(hard.banks[0]), "alive": merged_rules(hard.banks[1])}
    return out


# ----------------------------------------------------------------------
# Summaries and selection
# ----------------------------------------------------------------------
CLAUSE_FIELDS = ["active", "distinct", "duplicate_fraction", "pure_fraction", "distinct_in_reference",
                 "mean_literals", "empty", "contradictory", "included_literals"]


def summarize_runs(runs: List[Dict], model: str) -> Dict:
    epochs = [r["first_exact_epoch"] for r in runs if r["first_exact_epoch"] is not None]
    s = {
        "n_runs": len(runs),
        "n_exact": int(sum(r["exact"] for r in runs)),
        "exact_rate": float(np.mean([r["exact"] for r in runs])),
        "ever_exact": int(len(epochs)),
        "stays_exact": int(sum(r["stays_exact"] for r in runs)),
        "exact_last100_mean": float(np.mean([r["exact_last100"] for r in runs])),
        "first_exact_epoch_median": float(np.median(epochs)) if epochs else None,
        "first_exact_epoch_max": int(max(epochs)) if epochs else None,
        "hard": summarize([r["hard"] for r in runs]),
        "wrong_rows_mean": float(np.mean([len(r["wrong_rows"]) for r in runs])),
        "train_time_s_mean": float(np.mean([r["train_time_s"] for r in runs])),
        "clauses": {f: float(np.mean([r["clauses"]["total"][f] for r in runs])) for f in CLAUSE_FIELDS},
    }
    if "shared_across" in runs[0]["clauses"]["total"]:
        s["clauses"]["shared_across"] = float(np.mean([r["clauses"]["total"]["shared_across"] for r in runs]))
    if model == "relaxed":
        s["soft"] = summarize([r["soft"] for r in runs])
        s["gap_mean"] = float(np.mean([r["gap"] for r in runs]))
        s["undecided_mean"] = float(np.mean([r["undecided"] for r in runs]))
    else:
        s["hard_matches_original"] = int(sum(r["hard_matches_original"] for r in runs))
    return s


def sort_key(st: Dict):
    sm = st["summary"]
    med = sm["first_exact_epoch_median"]
    return (-sm["exact_rate"], med if med is not None else float("inf"), -sm["hard"]["mean"])


def short(s: Dict) -> str:
    med = s["first_exact_epoch_median"]
    return (f"exact {s['n_exact']}/{s['n_runs']} (ever {s['ever_exact']})  hard {s['hard']['mean']:.2f}%  "
            f"median epochs to exact {med if med is not None else '-'}  "
            f"exact in last 100 epochs {100 * s['exact_last100_mean']:.0f}%")


def run_pool(fn, jobs, workers):
    t0 = time.perf_counter()
    out = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(fn, jobs), 1):
            out.append(r)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)} runs, {fmt_time(el)} elapsed, ~{fmt_time(el / i * (len(jobs) - i))} left",
                      flush=True)
    return out


def tune_name(model: str, arch: str, ext: bool = False) -> str:
    return f"g0_tune_{model}_{arch}{'_ext' if ext else ''}.json"


def load(name: str) -> Optional[Dict]:
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def all_tuned_settings(model: str, arch: str) -> List[Dict]:
    out = []
    for ext in (False, True):
        d = load(tune_name(model, arch, ext))
        if d:
            out.extend(d["settings"])
    return out


def edges(setting: Dict, model: str) -> Dict[str, str]:
    """Axes on which the setting lies on the edge of the original grid: {axis: 'low'|'high'}."""
    out = {}
    for axis, values in GRIDS[model].items():
        if setting[axis] == min(values):
            out[axis] = "low"
        elif setting[axis] == max(values):
            out[axis] = "high"
    return out


def run_settings(grid: List[Dict], seeds: List[int], base: Dict, fn, model: str, workers: int) -> List[Dict]:
    jobs = [{**base, **g, "seed": sd} for g in grid for sd in seeds]
    print(f"{len(grid)} settings x {len(seeds)} seeds = {len(jobs)} runs, {workers} workers")
    runs = run_pool(fn, jobs, workers)
    settings = []
    for i, g in enumerate(grid):
        rs = runs[i * len(seeds):(i + 1) * len(seeds)]
        st = {"setting": g, "summary": summarize_runs(rs, model), "runs": rs}
        settings.append(st)
        print(f"  {g}: {short(st['summary'])}")
    return settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=["relaxed", "standard"])
    ap.add_argument("--arch", required=True, choices=["single", "multi"])
    ap.add_argument("--mode", required=True, choices=["tune", "extend", "main", "ablation"])
    ap.add_argument("--ablation", choices=["bias", "squared", "boost"])
    ap.add_argument("--clauses", type=int, default=CLAUSES_PER_TM, help="clauses per TM")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tune-seeds", type=int, default=len(TUNE_SEEDS), help="number of tuning seeds (from 1000)")
    ap.add_argument("--runs", type=int, default=MAIN_SEEDS, help="seeds 0..runs-1 for main/ablation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    fn = relaxed_run if a.model == "relaxed" else standard_run
    base = {"arch": a.arch, "clauses": a.clauses, "epochs": a.epochs}
    t0 = time.perf_counter()
    print(f"{a.model} TM, {a.arch}, {a.clauses} clauses per TM, {a.epochs} epochs, mode {a.mode}")

    if a.mode == "tune":
        seeds = TUNE_SEEDS[: a.tune_seeds]
        axes = GRIDS[a.model]
        grid = [dict(zip(axes, vals)) for vals in product(*axes.values())]
        settings = run_settings(grid, seeds, base, fn, a.model, a.workers)
        best = min(settings, key=sort_key)
        e = edges(best["setting"], a.model)
        print(f"Chosen: {best['setting']}  ({short(best['summary'])})")
        print(f"On a grid edge: {e if e else 'no'}" + ("  -> run --mode extend" if e else ""))
        result = {"seeds": seeds, "chosen": best["setting"], "edges": e, "settings": settings}
        name = tune_name(a.model, a.arch)

    elif a.mode == "extend":
        tuned = load(tune_name(a.model, a.arch))
        if not tuned:
            ap.error("run --mode tune first")
        e = tuned["edges"]
        if not e:
            print("The chosen setting is not on a grid edge: no extension (PLAN.md section 5).")
            return
        seeds = tuned["seeds"]
        axes = GRIDS[a.model]
        grid = []
        for axis, side in e.items():
            other = {k: v for k, v in axes.items() if k != axis}
            for vals in product(*other.values()):
                g = dict(zip(other, vals))
                g[axis] = EXTEND[axis][side]
                grid.append({k: g[k] for k in axes})
        settings = run_settings(grid, seeds, base, fn, a.model, a.workers)
        best = min(tuned["settings"] + settings, key=sort_key)
        print(f"Chosen over tune + extend: {best['setting']}  ({short(best['summary'])})")
        result = {"seeds": seeds, "extended": e, "chosen": best["setting"], "settings": settings}
        name = tune_name(a.model, a.arch, ext=True)

    else:
        settings = all_tuned_settings(a.model, a.arch)
        if not settings:
            ap.error("run --mode tune first")
        setting = dict(min(settings, key=sort_key)["setting"])
        suffix = ""
        if a.mode == "ablation":
            if a.ablation is None:
                ap.error("--ablation is required")
            if a.ablation in ("bias", "squared") and a.model != "relaxed":
                ap.error(f"--ablation {a.ablation} is for the relaxed TMs")
            if a.ablation == "boost" and (a.model, a.arch) != ("standard", "multi"):
                ap.error("--ablation boost is for the standard multi-class TM")
            setting.update({"bias": {"learn_bias": True}, "squared": {"loss": "squared"},
                            "boost": {"boost": 1}}[a.ablation])
            suffix = f"_ablation_{a.ablation}"
        jobs = [{**base, **setting, "seed": sd, "keep_model": True, "keep_rules": sd < KEEP_RULES_FOR_SEEDS}
                for sd in range(a.runs)]
        print(f"setting {setting}: {a.runs} seeds, {a.workers} workers")
        runs = run_pool(fn, jobs, a.workers)
        summary = summarize_runs(runs, a.model)
        print(short(summary))
        result = {"setting": setting, "summary": summary, "runs": runs}
        name = f"g0_{a.model}_{a.arch}{suffix}.json"

    result = {"model": a.model, "arch": a.arch, "mode": a.mode, **base, "batch": BATCH, **result,
              "wall_time_s": time.perf_counter() - t0}
    print(f"Done in {fmt_time(result['wall_time_s'])}")
    if not a.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / name).write_text(json.dumps(result, indent=1))
        print(f"Saved results/{name}")


if __name__ == "__main__":
    main()
