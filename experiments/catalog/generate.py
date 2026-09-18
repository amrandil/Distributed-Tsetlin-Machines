#!/usr/bin/env python3
"""
Build a static catalog of cross-class CLA pair results.

Pre-renders Wolfram-style elementary-CA plates (rule icon, single-cell seed,
random initial condition) for the 88 unique ECA rules, then writes one HTML
page per pair under majority and minority feedback.

Usage (from the repo root):

    python experiments/catalog/generate.py
    python -m http.server -d experiments 8000

Then open http://127.0.0.1:8000/catalog/site/index.html

Highlights stay a stub until experiments/catalog/highlights.json is filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dtm_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from cakit.rules import (
    CLASS_SHORT_LABELS,
    UNIQUE_RULES,
    get_class,
    get_label,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
CATALOG_SRC = Path(__file__).resolve().parent
SITE = CATALOG_SRC / "site"
RULE_DIR = SITE / "rules"
HIGHLIGHTS_PATH = CATALOG_SRC / "highlights.json"
AUDIT_PATH = EXPERIMENTS / "analysis" / "visual_taxonomy_audit.csv"

CLASS_PAIRS = ["I_x_II", "I_x_III", "I_x_IV", "II_x_III", "II_x_IV", "III_x_IV"]
FEEDBACKS = ("majority", "minority")

NEIGHBORHOODS = (
    (1, 1, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, 0, 0),
    (0, 1, 1),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
)

# Same canvas for both Wolfram-style plates so they sit side by side at equal size.
PLATE_SIZE = 201
PLATE_GENS = 120
RANDOM_SEED = 42


# ---- Elementary CA -----------------------------------------------------------

def _rule_table(rule: int) -> np.ndarray:
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


def eca_evolve(
    initial: np.ndarray,
    rule: int,
    generations: int,
    boundary: str,
) -> np.ndarray:
    table = _rule_table(rule)
    hist = np.zeros((generations + 1, initial.size), dtype=np.uint8)
    hist[0] = initial.astype(np.uint8)
    for t in range(generations):
        row = hist[t]
        if boundary == "periodic":
            left = np.roll(row, 1)
            right = np.roll(row, -1)
        else:
            left = np.empty_like(row)
            right = np.empty_like(row)
            left[0] = 0
            left[1:] = row[:-1]
            right[-1] = 0
            right[:-1] = row[1:]
        hist[t + 1] = table[(left << 2) | (row << 1) | right]
    return hist


def _save_spacetime(history: np.ndarray, path: Path) -> None:
    rows, cols = history.shape
    cell = 0.028
    fig, ax = plt.subplots(figsize=(cols * cell, rows * cell))
    ax.imshow(history, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1, aspect="equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def _save_rule_icon(rule: int, path: Path) -> None:
    table = _rule_table(rule)
    fig, ax = plt.subplots(figsize=(9.4, 1.85))
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 1.85)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cell = 0.20
    gap = 0.06
    for i, nbhd in enumerate(NEIGHBORHOODS):
        x0 = 0.22 + i * 1.15
        for j, bit in enumerate(nbhd):
            ax.add_patch(
                Rectangle(
                    (x0 + j * (cell + gap), 1.28),
                    cell,
                    cell,
                    facecolor="black" if bit else "white",
                    edgecolor="black",
                    linewidth=0.6,
                )
            )
        out = int(table[nbhd[0] * 4 + nbhd[1] * 2 + nbhd[2]])
        ax.add_patch(
            Rectangle(
                (x0 + cell + gap, 0.48),
                cell,
                cell,
                facecolor="black" if out else "white",
                edgecolor="black",
                linewidth=0.6,
            )
        )
        ax.text(
            x0 + cell + gap + cell / 2,
            0.18,
            "".join(str(b) for b in nbhd),
            ha="center",
            va="center",
            fontsize=6,
            color="#444444",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_rule_plates(force: bool = False) -> None:
    RULE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    random_seed = rng.integers(0, 2, size=PLATE_SIZE, dtype=np.uint8)
    for rule in UNIQUE_RULES:
        icon = RULE_DIR / f"rule_{rule}_icon.png"
        single = RULE_DIR / f"rule_{rule}_single.png"
        random_path = RULE_DIR / f"rule_{rule}_random.png"
        if force or not icon.exists():
            _save_rule_icon(rule, icon)
        if force or not single.exists():
            center = np.zeros(PLATE_SIZE, dtype=np.uint8)
            center[PLATE_SIZE // 2] = 1
            _save_spacetime(
                eca_evolve(center, rule, PLATE_GENS, boundary="periodic"),
                single,
            )
        if force or not random_path.exists():
            _save_spacetime(
                eca_evolve(random_seed, rule, PLATE_GENS, boundary="periodic"),
                random_path,
            )


# ---- Catalog scan ------------------------------------------------------------

def _pair_dir_name(rule_a: int, rule_b: int) -> str:
    return f"rule{rule_a}_vs_rule{rule_b}"


def _load_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_audit() -> Dict[Tuple[str, int, int], Dict[str, str]]:
    out: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    if not AUDIT_PATH.exists():
        return out
    with AUDIT_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (row["feedback"], int(row["rule_a"]), int(row["rule_b"]))
            except (KeyError, ValueError):
                continue
            out[key] = row
    return out


def _load_highlights() -> Dict[str, object]:
    if not HIGHLIGHTS_PATH.exists():
        return {"status": "pending", "items": []}
    with HIGHLIGHTS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def scan_pairs() -> List[Dict[str, object]]:
    audit = _load_audit()
    pairs: List[Dict[str, object]] = []
    for feedback in FEEDBACKS:
        root = EXPERIMENTS / f"results_{feedback}"
        if not root.exists():
            continue
        for class_pair in CLASS_PAIRS:
            class_dir = root / class_pair
            if not class_dir.is_dir():
                continue
            for pair_dir in sorted(class_dir.glob("rule*_vs_rule*")):
                if not pair_dir.is_dir():
                    continue
                try:
                    left, right = pair_dir.name.split("_vs_")
                    rule_a = int(left.replace("rule", ""))
                    rule_b = int(right.replace("rule", ""))
                except ValueError:
                    continue
                metrics = _load_metrics(pair_dir / "metrics.json")
                tag = audit.get((feedback, rule_a, rule_b), {})
                pairs.append(
                    {
                        "feedback": feedback,
                        "class_pair": class_pair,
                        "rule_a": rule_a,
                        "rule_b": rule_b,
                        "dir": pair_dir,
                        "has_bw": (pair_dir / "spacetime_bw.png").exists(),
                        "has_color": (pair_dir / "spacetime_color.png").exists(),
                        "has_hamming": (pair_dir / "hamming_distance.png").exists(),
                        "metrics": metrics,
                        "label": str(metrics.get("label") or ""),
                        "taxonomy": tag.get("proposed_primary", ""),
                        "motif": tag.get("motif", ""),
                        "your_tag": tag.get("your_tag", ""),
                    }
                )
    pairs.sort(key=lambda p: (p["feedback"], p["class_pair"], p["rule_a"], p["rule_b"]))
    return pairs


# ---- HTML --------------------------------------------------------------------

def _asset(depth: int, name: str) -> str:
    return f"{'../' * depth}assets/{name}"


def _rule_img(depth: int, rule: int, kind: str) -> str:
    return f"{'../' * depth}rules/rule_{rule}_{kind}.png"


def _result_img(depth: int, pair: Dict[str, object], name: str) -> str:
    # Pages live under experiments/catalog/site/; results live under experiments/.
    up = "/".join([".."] * (depth + 2))
    return (
        f"{up}/results_{pair['feedback']}/{pair['class_pair']}/"
        f"{_pair_dir_name(int(pair['rule_a']), int(pair['rule_b']))}/{name}"
    )


def _pair_href(from_depth: int, pair: Dict[str, object]) -> str:
    path = f"{pair['feedback']}/{pair['class_pair']}/{_pair_dir_name(int(pair['rule_a']), int(pair['rule_b']))}.html"
    return f"{'../' * from_depth}{path}"


def _page(title: str, body: str, depth: int, active: str = "") -> str:
    def nav(label: str, href: str, key: str) -> str:
        cls = ' class="active"' if active == key else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    prefix = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{_asset(depth, 'style.css')}">
</head>
<body>
  <div class="wrap">
    <header class="site">
      <h1>CLA rule-pair catalog</h1>
      <nav class="site">
        {nav("Home", prefix + "index.html", "home")}
        {nav("Majority", prefix + "majority/index.html", "majority")}
        {nav("Minority", prefix + "minority/index.html", "minority")}
        {nav("Rules", prefix + "rules/index.html", "rules")}
        {nav("Highlights", prefix + "highlights.html", "highlights")}
      </nav>
    </header>
    {body}
  </div>
  <script src="{_asset(depth, 'app.js')}"></script>
</body>
</html>
"""


def _view_switch(*, item: bool = False) -> str:
    attr = "data-item-view" if item else "data-gallery-view"
    cls = "view-switch item-view" if item else "view-switch"
    label = "This pair diagram view" if item else "All gallery diagrams"
    return f"""<div class="{cls}" role="group" aria-label="{label}">
      <button type="button" {attr}="bw">B&amp;W</button>
      <button type="button" {attr}="color">Color</button>
    </div>"""


def _img(src: str, alt: str, exists: bool = True) -> str:
    if not exists:
        return f'<p class="missing">Missing: {escape(alt)}</p>'
    return f'<img src="{escape(src)}" alt="{escape(alt)}" loading="lazy">'


def _gallery_thumb(pair: Dict[str, object], depth: int) -> str:
    bw = _result_img(depth, pair, "spacetime_bw.png") if pair["has_bw"] else ""
    color = _result_img(depth, pair, "spacetime_color.png") if pair["has_color"] else ""
    src = bw or color
    if not src:
        return '<p class="missing">Missing diagrams</p>'
    return (
        f'<img class="gallery-thumb" src="{escape(src)}" '
        f'data-bw="{escape(bw)}" data-color="{escape(color)}" '
        f'alt="rule {pair["rule_a"]} vs {pair["rule_b"]}" loading="lazy">'
    )


def _rule_section(pair: Dict[str, object], which: str, depth: int) -> str:
    rule = int(pair[which])
    return f"""
    <section class="block">
      <div class="rule-head">
        <div>
          <h2>{escape(get_label(rule))}</h2>
          <p class="lede" style="margin:0.2rem 0 0">Elementary CA lookup (111→000) and two standard evolutions.</p>
        </div>
        {_img(_rule_img(depth, rule, "icon"), f"Rule {rule} neighborhood map")}
      </div>
      <div class="plot-grid plates">
        <div class="plot-card">
          <h3>From a single cell</h3>
          {_img(_rule_img(depth, rule, "single"), f"Rule {rule} from a single centered 1")}
        </div>
        <div class="plot-card">
          <h3>From a random grid</h3>
          {_img(_rule_img(depth, rule, "random"), f"Rule {rule} from a random initial row")}
        </div>
      </div>
    </section>
    """


METRIC_KEYS = (
    "label",
    "mean_hamming",
    "std_hamming",
    "mean_hamming_after_burn_in",
    "mean_hamming_final_window",
    "std_hamming_final_window",
    "max_hamming_final_window",
    "detected_grid_period",
    "detected_grid_period_after_burn_in",
    "detected_hamming_period",
    "burn_in",
    "final_window",
    "max_period",
    "grid_size",
    "generations",
    "feedback",
    "feedback_radius",
    "seed",
)


def _metrics_table(metrics: Dict[str, object]) -> str:
    if not metrics:
        return (
            '<p class="missing">No metrics.json for this run '
            "(the minority batch did not write Hamming metrics).</p>"
        )
    rows = []
    for key in METRIC_KEYS:
        if key not in metrics:
            continue
        val = metrics[key]
        if val is None:
            val_s = "none"
        elif isinstance(val, float):
            val_s = f"{val:.6g}"
        else:
            val_s = str(val)
        rows.append(f"<tr><th>{escape(key)}</th><td>{escape(val_s)}</td></tr>")
    return "<table class='metrics'>" + "".join(rows) + "</table>"


def _write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_pair_page(
    pair: Dict[str, object],
    prev_pair: Optional[Dict[str, object]],
    next_pair: Optional[Dict[str, object]],
    other: Optional[Dict[str, object]],
) -> None:
    depth = 2
    a = int(pair["rule_a"])
    b = int(pair["rule_b"])
    title = f"Rule {a} vs {b} · {pair['feedback']}"
    crumbs = (
        f'<div class="crumbs"><a href="../../index.html">Home</a> · '
        f'<a href="../index.html">{escape(str(pair["feedback"]))}</a> · '
        f'<a href="index.html">{escape(str(pair["class_pair"]))}</a></div>'
    )
    chips = [
        f'<span class="chip">{escape(str(pair["feedback"]))} feedback</span>',
        f'<span class="chip">{escape(str(pair["class_pair"]))}</span>',
        f'<span class="chip blue">Rule {a} · {CLASS_SHORT_LABELS[get_class(a)]}</span>',
        f'<span class="chip red">Rule {b} · {CLASS_SHORT_LABELS[get_class(b)]}</span>',
    ]
    if pair["label"]:
        chips.append(f'<span class="chip">{escape(str(pair["label"]))}</span>')
    if pair["taxonomy"]:
        chips.append(f'<span class="chip">{escape(str(pair["taxonomy"]))}</span>')

    def pager_link(item: Optional[Dict[str, object]], label: str) -> str:
        if item is None:
            return f"<span class='missing'>{label}</span>"
        if item["feedback"] == pair["feedback"] and item["class_pair"] == pair["class_pair"]:
            href = f"{_pair_dir_name(int(item['rule_a']), int(item['rule_b']))}.html"
        else:
            href = _pair_href(2, item)
        return (
            f'<a href="{href}">{label}: rule {item["rule_a"]} vs {item["rule_b"]}</a>'
        )

    other_html = ""
    if other is not None:
        other_html = (
            f'<a href="{_pair_href(2, other)}">Same pair under {other["feedback"]} feedback</a>'
        )
    else:
        other_fb = "minority" if pair["feedback"] == "majority" else "majority"
        other_html = f"<span class='missing'>No {other_fb} run for this pair</span>"

    pair_plots = f"""
    <section class="block">
      <h2>The pair, from a random grid</h2>
      <p class="lede">CLA rule selection on the same 201-cell random initial condition used in the batch ({escape(str(pair['feedback']))} neighborhood feedback).</p>
      <div class="plot-grid">
        <div class="plot-card">
          <h3>Cell state (B&amp;W)</h3>
          {_img(_result_img(depth, pair, "spacetime_bw.png"), "pair spacetime B&W", bool(pair["has_bw"]))}
        </div>
        <div class="plot-card">
          <h3>TA rule choice (color)</h3>
          {_img(_result_img(depth, pair, "spacetime_color.png"), "pair spacetime color", bool(pair["has_color"]))}
        </div>
      </div>
    </section>
    <section class="block">
      <h2>Hamming distance and metrics</h2>
      <p class="lede">Row-to-row fraction of cells that change. Burn-in is generation 100; the label uses the last 100 steps.</p>
      <div class="plot-grid">
        <div class="plot-card">
          <h3>Hamming time series</h3>
          {_img(_result_img(depth, pair, "hamming_distance.png"), "Hamming distance", bool(pair["has_hamming"]))}
        </div>
        <div class="plot-card">
          <h3>Metrics</h3>
          {_metrics_table(pair.get("metrics") or {})}
        </div>
      </div>
    </section>
    """

    notes = ""
    if pair["your_tag"] or pair["motif"]:
        bits = []
        if pair["your_tag"]:
            bits.append(escape(str(pair["your_tag"])))
        if pair["motif"]:
            bits.append("motif: " + escape(str(pair["motif"])))
        notes = f'<p class="lede">Hand label (audit): {" · ".join(bits)}</p>'

    body = f"""
    {crumbs}
    <p class="pair-title">Rule {a} vs Rule {b}</p>
    <div class="meta-row">{''.join(chips)}</div>
    <div class="pager">{pager_link(prev_pair, "Previous")}{other_html}{pager_link(next_pair, "Next")}</div>
    {notes}
    {_rule_section(pair, "rule_a", depth)}
    {_rule_section(pair, "rule_b", depth)}
    {pair_plots}
    """
    _write(
        SITE / str(pair["feedback"]) / str(pair["class_pair"]) / f"{_pair_dir_name(a, b)}.html",
        _page(title, body, depth, active=str(pair["feedback"])),
    )


def _group(pairs: Sequence[Dict[str, object]], feedback: str, class_pair: Optional[str] = None) -> List[Dict[str, object]]:
    out = [p for p in pairs if p["feedback"] == feedback]
    if class_pair is not None:
        out = [p for p in out if p["class_pair"] == class_pair]
    return out


def write_listing(feedback: str, class_pair: str, items: Sequence[Dict[str, object]]) -> None:
    depth = 2
    labels = sorted({str(p["label"]) for p in items if p["label"]})
    options = "".join(f'<option value="{escape(lab)}">{escape(lab)}</option>' for lab in labels)
    label_filter = ""
    if options:
        label_filter = f'<select id="list-label"><option value="">All Hamming labels</option>{options}</select>'
    cards = []
    for p in items:
        img = _gallery_thumb(p, depth)
        href = f"{_pair_dir_name(int(p['rule_a']), int(p['rule_b']))}.html"
        extra = p["label"] or p["taxonomy"] or ""
        hay = f"{p['rule_a']} {p['rule_b']} {p['label']} {p['taxonomy']}".lower()
        cards.append(
            f'<article class="card" data-pair="{escape(hay)}" data-label="{escape(str(p["label"]))}">'
            f'<a class="thumb-link" href="{href}">{img}</a>'
            f"{_view_switch(item=True)}"
            f'<a class="card-meta" href="{href}"><div class="k">Rule {p["rule_a"]} vs {p["rule_b"]}</div>'
            f'<div class="s">{escape(str(extra))}</div></a></article>'
        )
    body = f"""
    <div class="crumbs"><a href="../../index.html">Home</a> · <a href="../index.html">{escape(feedback)}</a></div>
    <p class="pair-title">{escape(class_pair.replace('_', ' '))}</p>
    <p class="lede">{len(items)} pairs. Click a card for the Wolfram-style rule plates plus the CLA fight.</p>
    <div class="search">
      <input id="list-filter" type="search" placeholder="Filter by rule number…">
      {label_filter}
      {_view_switch()}
    </div>
    <div class="grid-cards">{''.join(cards)}</div>
    """
    _write(
        SITE / feedback / class_pair / "index.html",
        _page(f"{feedback} · {class_pair}", body, depth, active=feedback),
    )


def write_feedback_index(feedback: str, items: Sequence[Dict[str, object]]) -> None:
    counts = {cp: len(_group(items, feedback, cp)) for cp in CLASS_PAIRS}
    cards = []
    for cp in CLASS_PAIRS:
        n = counts[cp]
        cards.append(
            f'<a class="card" href="{cp}/index.html"><div class="k">{cp.replace("_", " ")}</div>'
            f'<div class="s">{n} pairs</div></a>'
        )
    body = f"""
    <div class="crumbs"><a href="../index.html">Home</a></div>
    <p class="pair-title">{escape(feedback).title()} feedback</p>
    <p class="lede">{len(items)} cross-class pairs. Each pair page shows both elementary rules, then the CLA spacetime diagrams.</p>
    <div class="grid-cards">{''.join(cards)}</div>
    """
    _write(SITE / feedback / "index.html", _page(f"{feedback} feedback", body, 1, active=feedback))


def write_rules_index() -> None:
    cards = []
    for rule in UNIQUE_RULES:
        cards.append(
            f"""<div class="card">
              <div class="k">{escape(get_label(rule))}</div>
              <img src="rule_{rule}_icon.png" alt="Rule {rule} icon" loading="lazy">
              <div class="plot-grid plates">
                <div class="plot-card"><h3>Single cell</h3><img src="rule_{rule}_single.png" alt="" loading="lazy"></div>
                <div class="plot-card"><h3>Random</h3><img src="rule_{rule}_random.png" alt="" loading="lazy"></div>
              </div>
            </div>"""
        )
    body = f"""
    <p class="pair-title">88 unique elementary rules</p>
    <p class="lede">Both plates use a periodic 201-cell ring and 120 steps. Single-cell plots start from one centered 1; random plots share one initial row (seed {RANDOM_SEED}).</p>
    <div class="grid-cards">{''.join(cards)}</div>
    """
    _write(SITE / "rules" / "index.html", _page("Elementary rules", body, 1, active="rules"))


def write_highlights(pairs: Sequence[Dict[str, object]]) -> None:
    data = _load_highlights()
    items = data.get("items") or []
    if data.get("status") == "pending" or not items:
        body = f"""
        <p class="pair-title">Highlights</p>
        <div class="note">
          <strong>Waiting on comments.</strong>
          {escape(str(data.get("note") or "Fill experiments/catalog/highlights.json, then re-run generate.py."))}
        </div>
        <p class="lede">This page will list the main interesting pairs, a short comment on each, and a link into the catalog.</p>
        """
    else:
        lookup = {
            (p["feedback"], int(p["rule_a"]), int(p["rule_b"])): p
            for p in pairs
        }
        cards = []
        for item in items:
            key = (item.get("feedback"), int(item["rule_a"]), int(item["rule_b"]))
            pair = lookup.get(key)
            title = item.get("title") or f"Rule {item['rule_a']} vs {item['rule_b']}"
            comment = item.get("comment") or ""
            link = _pair_href(0, pair) if pair else "#"
            thumb = ""
            if pair and pair["has_bw"]:
                thumb = _img(_result_img(0, pair, "spacetime_bw.png"), title)
            cards.append(
                f'<article class="block"><h2>{escape(str(title))}</h2>'
                f'<p>{escape(str(comment))}</p>{thumb}'
                f'<p><a href="{link}">Open pair page</a></p></article>'
            )
        body = f'<p class="pair-title">Highlights</p>{"".join(cards)}'
    _write(SITE / "highlights.html", _page("Highlights", body, 0, active="highlights"))


def write_home(pairs: Sequence[Dict[str, object]]) -> None:
    catalog = [
        {
            "feedback": p["feedback"],
            "class_pair": p["class_pair"],
            "rule_a": p["rule_a"],
            "rule_b": p["rule_b"],
            "label": p["label"],
            "taxonomy": p["taxonomy"],
        }
        for p in pairs
    ]
    n_maj = len(_group(pairs, "majority"))
    n_min = len(_group(pairs, "minority"))
    body = f"""
    <p class="pair-title">Cross-class Tsetlin CLA fights</p>
    <p class="lede">
      Each pair page shows how both elementary rules look on their own
      (single cell, then a shared random initial row), then the CLA spacetime
      diagrams from the batch runs. Majority and minority feedback are separate catalogs.
    </p>
    <div class="grid-cards">
      <a class="card" href="majority/index.html"><div class="k">Majority feedback</div><div class="s">{n_maj} pairs</div></a>
      <a class="card" href="minority/index.html"><div class="k">Minority feedback</div><div class="s">{n_min} pairs</div></a>
      <a class="card" href="rules/index.html"><div class="k">Elementary rules</div><div class="s">{len(UNIQUE_RULES)} unique ECAs</div></a>
      <a class="card" href="highlights.html"><div class="k">Highlights</div><div class="s">Commentary pending</div></a>
    </div>
    <section class="block" style="margin-top:1.2rem">
      <h2>Find a pair</h2>
      <div class="search">
        <input id="catalog-search" type="search" placeholder="e.g. 29 105 or coexist_sharp">
        <select id="catalog-feedback">
          <option value="">Both feedbacks</option>
          <option value="majority">majority</option>
          <option value="minority">minority</option>
        </select>
        <select id="catalog-class">
          <option value="">All class pairs</option>
          {''.join(f'<option value="{cp}">{cp}</option>' for cp in CLASS_PAIRS)}
        </select>
      </div>
      <div id="catalog-results" class="results-list"></div>
    </section>
    <script>const CATALOG = {json.dumps(catalog, separators=(',', ':'))};</script>
    """
    _write(SITE / "index.html", _page("CLA rule-pair catalog", body, 0, active="home"))


def copy_assets() -> None:
    dest = SITE / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("style.css", "app.js"):
        shutil.copy2(CATALOG_SRC / "assets" / name, dest / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the static CLA pair catalog.")
    parser.add_argument("--force-plates", action="store_true", help="Regenerate ECA reference plates.")
    parser.add_argument("--plates-only", action="store_true", help="Only render rule plates.")
    args = parser.parse_args()

    print("Rendering elementary-CA plates…")
    render_rule_plates(force=args.force_plates)
    if args.plates_only:
        print(f"Wrote plates to {RULE_DIR}")
        return

    print("Scanning pair results…")
    pairs = scan_pairs()
    if not pairs:
        raise SystemExit("No pair result directories found under experiments/results_majority or results_minority.")

    if SITE.exists():
        for child in SITE.iterdir():
            if child.name == "rules":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    copy_assets()
    write_home(pairs)
    write_highlights(pairs)
    write_rules_index()

    index = {(p["feedback"], int(p["rule_a"]), int(p["rule_b"])): p for p in pairs}
    for feedback in FEEDBACKS:
        items = _group(pairs, feedback)
        if not items:
            continue
        write_feedback_index(feedback, items)
        for class_pair in CLASS_PAIRS:
            group = _group(pairs, feedback, class_pair)
            if not group:
                continue
            write_listing(feedback, class_pair, group)
            for i, pair in enumerate(group):
                prev_pair = group[i - 1] if i else None
                next_pair = group[i + 1] if i + 1 < len(group) else None
                other_fb = "minority" if feedback == "majority" else "majority"
                other = index.get((other_fb, int(pair["rule_a"]), int(pair["rule_b"])))
                write_pair_page(pair, prev_pair, next_pair, other)

    print(f"Wrote {len(pairs)} pair pages → {SITE}")
    print("Serve with:  python -m http.server -d experiments 8000")
    print("Open:        http://127.0.0.1:8000/catalog/site/index.html")


if __name__ == "__main__":
    main()
