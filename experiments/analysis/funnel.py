"""
Apply the four-stage filter (the "funnel") to both feedbacks and document it.

Run after axes_metrics.py:

    python experiments/analysis/funnel.py                 # writes experiments/funnel.md + funnel.json
    python experiments/analysis/funnel.py --as-highlights # also makes the survivors the catalog highlights

Stages, in order (each removes runs from what the previous stage kept):
  1. complete domination   domination == dominate_complete
  2. neutral               domination == neutral (rules almost never competed)
  3. exact orbit           label in {fixed, cycle, drift}
  4. early settling        settle == early

Set aside, not discarded: runs removed at stage 3 that settled late
(label exact AND settle == late). These are the clean patterns that took time
to self-organise and are listed as their own group.

Outputs
  experiments/funnel.md                      human-readable, for the thesis
  experiments/results_<fb>/funnel.json       machine-readable (per stage lists)
  experiments/highlights.md (--as-highlights) catalog highlights in the format
                                              generate.py parses; the previous
                                              hand-written file is moved to
                                              experiments/highlights_manual.md
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
FEEDBACKS = ("majority", "minority")
EXACT_LABELS = ("fixed", "cycle", "drift")
GENERATIONS = 500
CLASS_ORDER = ["I_x_II", "I_x_III", "I_x_IV", "II_x_III", "II_x_IV", "III_x_IV"]
PRETTY_CLASS = {k: k.replace("_x_", " × ") for k in CLASS_ORDER}

STAGES = [
    ("complete_domination", "Complete domination",
     "One rule holds every cell of the TA-status plot over the last 100 generations.",
     lambda r: r["domination"] == "dominate_complete"),
    ("neutral", "Neutral coexistence",
     "Both rules are present on the TA-status plot, but fewer than 2% of cell-steps are "
     "decisive: the two rules give the same output on almost every neighbourhood the cells "
     "actually see, so the arm choice has no effect on the B&W plot.",
     lambda r: r["domination"] == "neutral"),
    ("exact_orbit", "Exact orbit",
     "The B&W plot ends in a fixed, cyclic or drifting pattern (lag-Hamming label).",
     lambda r: r["label"] in EXACT_LABELS),
    ("early_settling", "Early settling",
     "The B&W plot already shows its final pattern by generation 100.",
     lambda r: r["settle"] == "early"),
]


def _load(feedback: str) -> List[Dict[str, object]]:
    path = EXPERIMENTS / f"results_{feedback}" / "summary.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "decisive_frac" not in rows[0]:
        raise SystemExit(f"{path} has no decisive_frac column: run axes_metrics.py first.")
    for r in rows:
        r["rule_a"], r["rule_b"] = int(r["rule_a"]), int(r["rule_b"])
        r["decisive_frac"] = float(r["decisive_frac"])
        r["decisive_share_a"] = float(r["decisive_share_a"]) if r["decisive_share_a"] not in ("", None) else None
        r["settled_at"] = int(float(r["settled_at"])) if r["settled_at"] not in ("", None) else None
        r["freeze_gen"] = int(float(r["freeze_gen"]))
    return rows


def _key(r) -> str:
    return f"{r['rule_a']} vs {r['rule_b']}"


def _freeze_text(r) -> str:
    """freeze_gen == GENERATIONS means some TA switched arm in the very last generation."""
    return "still switching at the end" if r["freeze_gen"] >= GENERATIONS else f"frozen at {r['freeze_gen']}"


def _class_rank(r) -> int:
    return CLASS_ORDER.index(r["class_pair"]) if r["class_pair"] in CLASS_ORDER else len(CLASS_ORDER)


def _balance(r) -> float:
    """0 = perfectly even competition on decisive cell-steps, 0.5 = one-sided."""
    s = r["decisive_share_a"]
    return 0.5 if s is None else abs(s - 0.5)


def run_funnel(rows: List[Dict[str, object]]) -> Dict[str, object]:
    remaining = list(rows)
    stages = []
    side_late_exact: List[Dict[str, object]] = []
    for sid, title, desc, pred in STAGES:
        removed = [r for r in remaining if pred(r)]
        remaining = [r for r in remaining if not pred(r)]
        if sid == "exact_orbit":
            side_late_exact = [r for r in removed if r["settle"] == "late"]
        stages.append({"id": sid, "title": title, "description": desc,
                       "removed": removed, "remaining": len(remaining)})
    return {"total": len(rows), "stages": stages, "survivors": remaining,
            "late_exact": side_late_exact}


def _count(rows, field) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r[field])] = out.get(str(r[field]), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _examples(removed, n=6) -> List[str]:
    """One pair per class pair (most decisive-balanced first), up to n."""
    picked, seen = [], set()
    for r in sorted(removed, key=lambda r: (_balance(r), r["rule_a"], r["rule_b"])):
        if r["class_pair"] not in seen:
            seen.add(r["class_pair"])
            picked.append(f"{_key(r)} ({PRETTY_CLASS.get(r['class_pair'], r['class_pair'])}, {r['label']})")
        if len(picked) >= n:
            break
    return picked


def _crosstab_md(rows, row_field, col_field, row_order=None, col_order=None) -> str:
    rows_v = row_order or sorted({str(r[row_field]) for r in rows})
    cols_v = col_order or sorted({str(r[col_field]) for r in rows})
    lines = ["| | " + " | ".join(cols_v) + " | total |", "|---" * (len(cols_v) + 2) + "|"]
    for rv in rows_v:
        cells = [sum(1 for r in rows if str(r[row_field]) == rv and str(r[col_field]) == cv) for cv in cols_v]
        if sum(cells):
            lines.append(f"| {rv} | " + " | ".join(str(c) for c in cells) + f" | {sum(cells)} |")
    return "\n".join(lines)


def write_markdown(results: Dict[str, Dict[str, object]]) -> Path:
    L: List[str] = []
    L.append("# Filtering funnel\n")
    L.append("_Generated by `experiments/analysis/funnel.py` from `results_<feedback>/summary.csv`. "
             "Do not edit by hand; re-run the script._\n")
    L.append("Each stage removes one kind of run from what the previous stage kept. "
             "Survivors of all four stages are the runs where the two rules genuinely compete "
             "and the B&W plot neither ends in an exact orbit nor settles early.\n")

    L.append("## Counts\n")
    L.append("| Stage | " + " | ".join(f"{fb.capitalize()} remaining" for fb in results) + " |")
    L.append("|---" * (len(results) + 1) + "|")
    L.append("| All runs | " + " | ".join(str(res["total"]) for res in results.values()) + " |")
    for i, (sid, title, _, _) in enumerate(STAGES):
        L.append(f"| − {title.lower()} | " + " | ".join(
            str(res["stages"][i]["remaining"]) for res in results.values()) + " |")
    L.append("")
    L.append("Set aside at stage 3 (exact orbit reached late, kept as their own group): " + ", ".join(
        f"{fb} {len(res['late_exact'])}" for fb, res in results.items()) + ".\n")
    dom_pass = {fb: sum(1 for r in res["survivors"] if r["domination"] == "dominate")
                for fb, res in results.items()}
    L.append("Runs tagged `dominate` (one rule ≥ 90% but not 100%) are not removed by any stage; "
             "among the survivors: " + ", ".join(f"{fb} {n}" for fb, n in dom_pass.items()) + ".\n")

    for i, (sid, title, desc, _) in enumerate(STAGES):
        L.append(f"## Stage {i + 1}: {title}\n")
        L.append(desc + "\n")
        for fb, res in results.items():
            removed = res["stages"][i]["removed"]
            L.append(f"**{fb.capitalize()}**: {len(removed)} removed.\n")
            if not removed:
                continue
            L.append("By class pair: " + ", ".join(
                f"{PRETTY_CLASS.get(k, k)} {v}" for k, v in _count(removed, "class_pair").items()) + ".\n")
            if sid != "exact_orbit":
                L.append("By Hamming label: " + ", ".join(
                    f"{k} {v}" for k, v in _count(removed, "label").items()) + ".\n")
            L.append("Example candidates: " + "; ".join(_examples(removed)) + ".\n")

    L.append("## Set aside: clean patterns that settled late\n")
    L.append("Removed at stage 3 because the final orbit is exact, but the B&W plot only reached "
             "it after generation 100. The TAs column gives the last generation any TA switched arm: "
             "if it is before `settled_at`, the B&W pattern organised itself after the rule map "
             "had already stopped changing.\n")
    for fb, res in results.items():
        items = sorted(res["late_exact"], key=lambda r: (_class_rank(r), r["rule_a"], r["rule_b"]))
        L.append(f"**{fb.capitalize()}** ({len(items)})\n")
        if items:
            L.append("| Pair | Classes | Label | settled_at | TAs |")
            L.append("|---|---|---|---:|---|")
            for r in items:
                L.append(f"| {_key(r)} | {PRETTY_CLASS.get(r['class_pair'])} | {r['label']} | "
                         f"{r['settled_at']} | {_freeze_text(r)} |")
            L.append("")

    L.append("## Survivors\n")
    for fb, res in results.items():
        surv = res["survivors"]
        L.append(f"### {fb.capitalize()} ({len(surv)})\n")
        L.append("Hamming label × settle:\n")
        L.append(_crosstab_md(surv, "label", "settle",
                              col_order=[c for c in ("late", "tail_only", "never") if any(r["settle"] == c for r in surv)]))
        L.append("")
        L.append("By class pair: " + ", ".join(
            f"{PRETTY_CLASS.get(k, k)} {v}" for k, v in _count(surv, "class_pair").items()) + ".\n")

    out = EXPERIMENTS / "funnel.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def write_json(feedback: str, res: Dict[str, object]) -> Path:
    def slim(r):
        return {k: r[k] for k in ("rule_a", "rule_b", "class_pair", "label", "settle", "settled_at",
                                  "domination", "decisive_frac", "decisive_share_a", "freeze_gen")}
    data = {
        "feedback": feedback,
        "total": res["total"],
        "stages": [{"id": s["id"], "title": s["title"], "remaining": s["remaining"],
                    "removed": [slim(r) for r in s["removed"]]} for s in res["stages"]],
        "late_exact": [slim(r) for r in res["late_exact"]],
        "survivors": [slim(r) for r in res["survivors"]],
    }
    out = EXPERIMENTS / f"results_{feedback}" / "funnel.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


def write_highlights(results: Dict[str, Dict[str, object]]) -> Path:
    """Survivors (+ the late-exact group) in the format catalog/generate.py parses."""
    target = EXPERIMENTS / "highlights.md"
    backup = EXPERIMENTS / "highlights_manual.md"
    if target.exists() and not backup.exists():
        head = target.read_text(encoding="utf-8").splitlines()[:1]
        if not (head and head[0].startswith("# Funnel")):
            shutil.move(str(target), str(backup))
            print(f"  previous highlights moved to {backup.name}")

    L: List[str] = []
    for fb, res in results.items():
        L.append(f"# Funnel survivors: {fb} feedback\n")
        L.append("## Clean patterns that settled late\n")
        L.append("exact orbit reached after generation 100\n")
        for r in sorted(res["late_exact"], key=lambda r: (_class_rank(r), r["rule_a"], r["rule_b"])):
            L.append(f"{_key(r)} ⇒ {r['label']}, settled at {r['settled_at']}, TAs {_freeze_text(r)}\n")
        for cp in CLASS_ORDER:
            group = [r for r in res["survivors"] if r["class_pair"] == cp]
            if not group:
                continue
            L.append(f"## {PRETTY_CLASS[cp]}\n")
            for label in ("aperiodic", "near_cycle"):
                for settle in ("never", "late", "tail_only"):
                    sub = [r for r in group if r["label"] == label and r["settle"] == settle]
                    if not sub:
                        continue
                    L.append(f"{label} · {settle}\n")
                    for r in sorted(sub, key=_balance):
                        share = r["decisive_share_a"]
                        note = f"decisive share A {share:.2f}" if share is not None else ""
                        L.append(f"{_key(r)} ⇒ {note}\n")
    target.write_text("\n".join(L) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the filtering funnel and document it.")
    parser.add_argument("--as-highlights", action="store_true",
                        help="Write the survivors to experiments/highlights.md for the catalog "
                             "(the hand-written file is kept as highlights_manual.md).")
    args = parser.parse_args()

    results = {fb: run_funnel(_load(fb)) for fb in FEEDBACKS
               if (EXPERIMENTS / f"results_{fb}" / "summary.csv").exists()}
    for fb, res in results.items():
        counts = " → ".join([str(res["total"])] + [str(s["remaining"]) for s in res["stages"]])
        print(f"  {fb:<8} {counts}   (late exact set aside: {len(res['late_exact'])})")
        print(f"           wrote {write_json(fb, res).relative_to(EXPERIMENTS.parent)}")
    print(f"  wrote {write_markdown(results).relative_to(EXPERIMENTS.parent)}")
    if args.as_highlights:
        print(f"  wrote {write_highlights(results).relative_to(EXPERIMENTS.parent)}")


if __name__ == "__main__":
    main()
