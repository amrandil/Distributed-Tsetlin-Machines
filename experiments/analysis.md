# Cross-class pair analysis

How we score the 1,659 × 2 CLA runs, and how those scores are meant to
be **filtered independently** (in the catalog or in
`summary.csv`). Hamming is one axis, not the whole story.

Implementation: `experiments/cla/run_cross_class_pairs.py`.
Rebuild scores from existing PNGs (no CLA re-run):

```bash
python experiments/cla/run_cross_class_pairs.py --feedback majority --backfill-hamming
python experiments/cla/run_cross_class_pairs.py --feedback minority --backfill-hamming
python experiments/catalog/generate.py
```

Each pair directory contains:

- `spacetime_bw.png` — cell state (black / white)
- `spacetime_color.png` — which rule each TA chose (blue = rule A, red = rule B)
- `hamming_distance.png` — activity (fraction of cells that flip each step)
- `lag_hamming.png` — lag Hamming with and without a spatial roll
- `metrics.json` — all filter fields below

The same numbers are in `experiments/results_{majority,minority}/summary.csv`.
The catalog listing pages filter on Hamming label × domination × settle.

---

## What one run is

A pair is a 1D ring of **201** cells, evolved for **500** generations
(plus the initial row), Tsetlin **10** states per arm, seed **42**.
Each cell picks one of two elementary CA rules (the two arms of the
pair). Majority and minority neighborhood feedback are separate
batches.

Time is **down** the space-time plot. Two rows are the same if every
cell agrees. **Hamming distance** is the fraction of cells that
disagree (`0` = identical, `1` = every cell different).

---

## Filter axes

These are **product tags**, not a single taxonomy. You can look at one
axis alone, or intersect (e.g. coexist ∩ fixed ∩ late).

Suggested pass through the catalog:

1. **Rule domination** — look at complete one-rule lock, then ≥ 90%
   dominate, then hide those and look at two-rule fights.
2. **Hamming class** — what the whole B&W row is doing after burn-in.
3. **Settle time** — how soon the B&W pattern at the *bottom* of the
   plot is already in place.
4. **Borders** — not computed. After the filters above, label striking
   B&W domain walls by eye.

Confidence (shade of blue/red) is not scored.

| Axis | Source | Catalog field |
|---|---|---|
| Domination | color PNG, last 100 generations | `domination`, `dominant_rule`, `rule_share_a/b` |
| Hamming class | B&W PNG, after generation 100 | `label`, `period`, `shift`, `residual` |
| Settle | B&W PNG, from the end of the run | `settle`, `settled_at`, `late_label` |
| Borders | eye | — |

---

## 1. Rule domination

Decoded from `spacetime_color.png` the same way B&W Hamming is decoded:
each cell is a 4×4 pixel block; one interior pixel is sampled. Blue
shades (B > R) are rule A; red shades (R > B) are rule B. Only **which
arm**, not which of the 10 confidence states.

Measured on the **last 100 generations**.

| Tag | Meaning |
|---|---|
| `coexist` | Neither rule occupies ≥ 90% of cell-steps |
| `dominate` | One rule occupies ≥ 90%, but not all |
| `dominate_complete` | One rule occupies **100%** of cell-steps |

`dominant_rule` is `a` or `b` when one rule wins, else empty.
`rule_share_a` / `rule_share_b` are the raw fractions.

This is occupancy of the **color** plot, not “does the B&W texture look
like elementary rule A.” A frozen white field can still be coexist if
TAs stay split between arms.

---

## 2. Lag Hamming (B&W orbit)

The first **100** generations are burn-in. The Hamming **label** uses
only the rows after that.

### Consecutive Hamming (activity)

Compare each row only to the **next** row. That is how much changed in
one step. It does not tell you whether the movie repeats after 12
steps: a blinking pattern can flip many cells every frame and still be
periodic. Blue series in `hamming_distance.png`.

### Lag Hamming (does the whole line come back?)

**Lag** `τ` means “this many generations later.” For each `τ = 1, 2, …`
up to one third of the post-burn-in length (here **133**):

1. Compare row `t` to row `t + τ`.
2. Average the disagreement over `t`.

**Unshifted** lag Hamming: cells must match **in the same place**.

A pattern that **slides** along the ring looks new in those seats even
if the shape is unchanged. For the same lags we also try every
**circular roll** of the later row and keep the best (smallest)
disagreement. That is **shift-aware** lag Hamming. The winning roll is
a signed **shift** (`0` = no slide). A roll that only improves the
match by less than one cell is ignored, so noise does not invent a
slide; a single on-cell that actually moves still counts as drift (it
disagrees in two seats).

`lag_hamming.png` plots both curves. A valley at lag `p` means the
configuration (possibly after a slide) comes back after `p` steps. The
**period** is the **smallest** such valley. Harmonics (`2p`, `3p`, …)
are the same loop counted twice.

### Hamming labels

Only the shift-aware curve after burn-in is used.

| Label | Meaning |
|---|---|
| `fixed` | Period 1, shift 0, leftover disagreement ≈ 0. The line has stopped. |
| `cycle` | Period ≥ 2, shift 0, leftover ≈ 0. Repeats **in place**. |
| `drift` | Leftover ≈ 0 after a **nonzero** slide. Repeats while travelling. Period 1 is a still picture that scoots every step. |
| `near_cycle` | There is a real valley, but it is not a clean zero. Mostly periodic (standing or sliding). |
| `aperiodic` | No valley in range. The **whole** 201-cell row does not come back. |

**Exact** means leftover Hamming ≤ `1e-6`. **Near** means a local
minimum that is at most half the median of the shift-aware curve **and**
at most `0.25` leftover. Otherwise `aperiodic`.

Stored with the name:

- `period` — lag of the first accepted valley (`none` if aperiodic)
- `shift` — best roll at that lag
- `residual` — leftover disagreement **with** the best roll
- `residual_unshifted` — same lag, no roll

These numbers describe the **entire row**. A periodic patch on a messy
background often looks `aperiodic` or `near_cycle`.

---

## 3. Settle time (when the B&W pattern locks)

Independent of the official Hamming label, which averages **all** of
the post-burn-in window.

1. Classify the **last 167 rows** (about the last third of the plot)
   with the same lag-Hamming rules. That is `late_label` / `late_period`
   — what the **bottom** of the diagram is doing.
2. Walk down from generation 0 in steps of 50. Ask: from here to the
   end, is lag Hamming at that period at least as clean as the bottom?
3. `settled_at` is the first such generation.

| Tag | Meaning |
|---|---|
| `early` | `settled_at` ≤ 100. The end-pattern is already there by burn-in. |
| `late` | `settled_at` > 100. The top is messier; it organizes later. |
| `never` | The bottom itself is `aperiodic` (no period to lock onto). |

A pattern can flicker or move and still be settled: the *rule of the
movie* has started, even if many cells flip every step. Frozen is only
one kind of settled.

`never` should line up with Hamming `aperiodic` on the full window, but
not exactly: a tail can look aperiodic while the long window found a
weak valley (`near_cycle` + `never`), or a long window can look
aperiodic while the last third has a valley (`aperiodic` + `late`).

---

## 4. Borders (not computed)

Borders here means **B&W texture** interfaces (fixed next to periodic,
two different cycles, …), not TA-arm walls in the color plot. Those two
can disagree.

No automatic sharp / wavy / irregular tag. After filtering, label the
clear examples by hand.

---

## Batch stats

1,659 cross-class pairs per feedback. From `summary.csv` after the
domination + settle backfill.

### Domination

| Tag | Majority | Majority % | Minority | Minority % |
|---|---:|---:|---:|---:|
| `coexist` | 1609 | 97.0% | 1645 | 99.2% |
| `dominate` | 14 | 0.8% | 10 | 0.6% |
| `dominate_complete` | 36 | 2.2% | 4 | 0.2% |

True one-rule lock is rare. Minority almost never gives up the second
arm (`dominant_rule` is always `a` on the 14 minority dominate /
complete rows). Majority complete lock: 24× rule A, 12× rule B.

### Domination by class pair

**Majority**

| Class pair | n | coexist | dominate | complete |
|---|---:|---:|---:|---:|
| I × II | 520 | 505 | 3 | 12 |
| I × III | 88 | 82 | 0 | 6 |
| I × IV | 32 | 32 | 0 | 0 |
| II × III | 715 | 696 | 7 | 12 |
| II × IV | 260 | 251 | 4 | 5 |
| III × IV | 44 | 43 | 0 | 1 |

**Minority** — 14 non-coexist rows, all in II × III (10) or II × IV (4).
Every I × _ pair coexists.

### Hamming labels

| Label | Majority | Majority % | Minority | Minority % |
|---|---:|---:|---:|---:|
| `cycle` | 650 | 39.2% | 598 | 36.0% |
| `aperiodic` | 371 | 22.4% | 337 | 20.3% |
| `near_cycle` | 339 | 20.4% | 461 | 27.8% |
| `fixed` | 291 | 17.5% | 254 | 15.3% |
| `drift` | 8 | 0.5% | 9 | 0.5% |
| **Total** | **1659** | | **1659** | |

Global travelling orbits are rare: the whole line has to move as a unit.

Hamming class is mostly independent of domination: almost all `cycle` /
`fixed` rows are still `coexist` (TAs split, B&W orbit regular). The
50 majority non-coexist rows scatter across `fixed`, `near_cycle`,
`aperiodic`, and a few `drift`. All 14 minority non-coexist rows are
`near_cycle`.

### Labels by Wolfram class pair (majority)

| Class pair | n | fixed | cycle | drift | near_cycle | aperiodic |
|---|---:|---:|---:|---:|---:|---:|
| I × II | 520 | 276 | 231 | 0 | 13 | 0 |
| I × III | 88 | 0 | 76 | 0 | 12 | 0 |
| I × IV | 32 | 6 | 20 | 0 | 6 | 0 |
| II × III | 715 | 1 | 239 | 3 | 202 | 270 |
| II × IV | 260 | 8 | 81 | 5 | 99 | 67 |
| III × IV | 44 | 0 | 3 | 0 | 7 | 34 |

Class I in the pair still produces many frozen or exact cycles. II × III
and III × IV hold almost all of the `aperiodic` mass.

### Labels by Wolfram class pair (minority)

| Class pair | n | fixed | cycle | drift | near_cycle | aperiodic |
|---|---:|---:|---:|---:|---:|---:|
| I × II | 520 | 237 | 192 | 1 | 84 | 6 |
| I × III | 88 | 0 | 50 | 0 | 31 | 7 |
| I × IV | 32 | 8 | 10 | 0 | 9 | 5 |
| II × III | 715 | 1 | 269 | 0 | 231 | 214 |
| II × IV | 260 | 8 | 74 | 8 | 95 | 75 |
| III × IV | 44 | 0 | 3 | 0 | 11 | 30 |

Minority feedback yields more `near_cycle` and fewer clean `fixed` /
`cycle` labels than majority, especially in I × II.

### Settle time

| Tag | Majority | Majority % | Minority | Minority % |
|---|---:|---:|---:|---:|
| `early` | 1025 | 61.8% | 916 | 55.2% |
| `late` | 243 | 14.6% | 402 | 24.2% |
| `never` | 391 | 23.6% | 341 | 20.6% |

Minority organizes later more often. Almost every exact `fixed` /
`cycle` / `drift` is `early` (the official Hamming window is already a
clean orbit, so the bottom cannot have appeared only at gen 350).
**Late** is mostly `near_cycle` (215 majority, 339 minority): the long
window is a bit dirty, the last third is cleaner. **Never** is almost
all `aperiodic`.

**Majority settle by class pair**

| Class pair | n | early | late | never |
|---|---:|---:|---:|---:|
| I × II | 520 | 511 | 9 | 0 |
| I × III | 88 | 76 | 12 | 0 |
| I × IV | 32 | 28 | 4 | 0 |
| II × III | 715 | 289 | 143 | 283 |
| II × IV | 260 | 115 | 71 | 74 |
| III × IV | 44 | 6 | 4 | 34 |

Class I pairs lock early. Chaos-involving pairs hold the late / never
mass.

### Leftover disagreement (`residual`)

Median leftover Hamming at the chosen lag (or at the best lag if
`aperiodic`):

| Label | Majority median | Minority median |
|---|---:|---:|
| `fixed` / `cycle` / `drift` | ~0 (float noise) | ~0 |
| `near_cycle` | 0.114 | 0.103 |
| `aperiodic` | 0.311 | 0.311 |

### Activity after burn-in

Median consecutive Hamming (fraction of cells that flip per step) after
generation 100:

| Label | Majority | Minority |
|---|---:|---:|
| `fixed` | 0 | 0 |
| `cycle` | 0.176 | 0.251 |
| `near_cycle` | 0.437 | 0.454 |
| `aperiodic` | 0.518 | 0.505 |
| `drift` | 0.597 | 0.826 |

Exact cycles can still be busy frame-to-frame. Drift is the most active
because the whole pattern is moving.

### Exact periods (`fixed` + `cycle` + `drift`)

Smallest lag with leftover ≈ 0. Period 1 is almost all `fixed`, plus the
period-1 `drift` rows.

**Majority** (28 distinct periods; max 128):

| Period | Count | Period | Count |
|---:|---:|---:|---:|
| 1 | 298 | 32 | 4 |
| 2 | 159 | 36 | 6 |
| 3 | 5 | 40 | 2 |
| 4 | 139 | 42 | 1 |
| 6 | 72 | 48 | 3 |
| 8 | 40 | 56 | 1 |
| 10 | 1 | 60 | 20 |
| 12 | 86 | 66 | 2 |
| 15 | 1 | 72 | 1 |
| 16 | 14 | 80 | 1 |
| 18 | 5 | 84 | 14 |
| 22 | 1 | 120 | 15 |
| 24 | 46 | 128 | 1 |
| 28 | 4 | 30 | 7 |

Most common **cycles** (period ≥ 2): 2, 4, 12, 6, 24, 8.

**Minority** (25 distinct periods; max 120). Most common exact cycles:
4 (126), 2 (118), 12 (102), 24 (51), 84 (40), 8 (33).

The old detector only searched periods ≤ 20. Many exact repeats here are
longer than that (24, 42, 60, 84, 120, …).

### Drift pairs

Majority (8):

| Pair | Period | Shift |
|---|---:|---:|
| 9 vs 45 (II × III) | 2 | −2 |
| 24 vs 30 (II × III) | 1 | −1 |
| 152 vs 30 (II × III) | 1 | −1 |
| 2 vs 106 (II × IV) | 1 | +1 |
| 34 vs 106 (II × IV) | 1 | +1 |
| 35 vs 106 (II × IV) | 1 | +1 |
| 130 vs 106 (II × IV) | 1 | +1 |
| 162 vs 106 (II × IV) | 1 | +1 |

Minority (9): 40 vs 57 (I × II), period 1, shift +1; the other eight are
II × IV pairs, all period 1, shift +1 (several again involve rule 106 or
41).

---

## What is not computed

- Periodic **regions** on a mixed background (unused cells keep global
  Hamming high).
- Two different periods on different parts of the line.
- B&W border geometry (sharp / wavy / irregular).
- Whether a coexist domain’s texture matches the solo elementary rule.
- TA confidence (shade).

Those stay visual, after the filters above.
