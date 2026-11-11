# Experiment Plan: Relaxed TM on Game of Life

Nov 8, 2026 · Amr Kandil

> Progress on each experiment is tracked in [README.md](README.md); results
> will go in `reports/`.

This experiment moves the relaxed TM from plain classification to a cellular
automaton rule. It asks three things: can the relaxed TM learn Conway's Game
of Life exactly, how many clauses does it need and how many of them end up as
duplicates, and does the learned rule behave like Game of Life when it runs
on a grid.

## 1. Goal and scope

**In scope**

- One single-output relaxed TM (the same `RelaxedTM` as in the
  classification experiments) that maps a cell's 3×3 neighbourhood to the
  cell's next state.
- One TM shared by every cell, trained on one step of Game of Life. This is
  the same setup as DiffLogic CA's Game of Life experiment (section 7).
- The standard TM (Granmo 2018 code, as in E1 and E4) next to the relaxed TM
  in every experiment, on the same data and seeds.
- Stefano's question from E0: how the number of clauses relates to duplicate
  clauses, now on a task whose required number of clauses is known (about
  190, section 3).
- A comparison of the single-output TM (one clause bank voting alive vs dead)
  with a two-class multi-class TM (one clause bank per class, G5).

**Out of scope for now**

- A held-out test (G2, kept as a note for future work, section 6).
- Per-cell TMs that learn locally; other CA tasks.

**What a successful result looks like:** the hardened relaxed TM gets all 512
neighbourhoods right in most seeds; we can say how the number of distinct and
duplicate clauses changes with the clause count, for both TMs; and the learned
rule, run on a large grid, produces exactly the same states as Game of Life.

## 2. Task and data

### The rule

Each cell is alive (1) or dead (0). Its next state depends on itself and its
8 neighbours (the Moore neighbourhood). Let k be the number of live
neighbours:

- a dead cell with exactly 3 live neighbours becomes alive (birth);
- a live cell with 2 or 3 live neighbours stays alive (survival);
- every other cell is dead in the next step.

In short: next = 1 if k = 3, or if the cell is alive and k = 2.

### Inputs and literals

```text
3×3 neighbourhood          input bits (row by row)
NW  N  NE                  x = [NW, N, NE, W, C, E, SW, S, SE]
 W  C   E                  9 bits, C (the cell itself) at index 4
SW  S  SE                  literals: 9 bits + 9 negations = 18
```

The bit order is the same as DiffLogic CA's (row by row, centre at index 4).
In printed rules the literals are named by position, e.g. `N ∧ ¬SE`.

### Data: the full table

There are 2⁹ = 512 possible neighbourhoods. The data set is all of them, each
once, with its next state as the label.

- 140 of the 512 lead to an alive cell (27%): 112 with exactly 3 live
  neighbours (centre alive or dead), and 28 with the centre alive and exactly
  2 live neighbours. The other 372 lead to a dead cell.
- **Training and evaluation use the same 512 rows.** The question in G0 and
  G1 is whether the TM can *represent* the rule exactly, not whether it
  generalises to unseen neighbourhoods (that would be G2). This is also what
  DiffLogic CA tested.
- Sampling 3×3 patches from random grids at density 0.5 gives every
  neighbourhood with equal probability, so training on random grids would be
  the same data as the table. DiffLogic CA's tiny 3×3 grids with wrap-around
  edges also contain only rows of this table.

## 3. Reference: how many clauses does Game of Life need?

A clause is an AND of literals, so one clause covers a whole group of
neighbourhoods. The fewer literals it has, the more neighbourhoods it covers
(each free bit doubles its coverage).

This clause set implements Game of Life exactly in the "any clause fires"
style: a positive clause fires only on neighbourhoods that lead to alive, a
negative clause only on neighbourhoods that lead to dead, and every
neighbourhood is covered by at least one clause of the right sign. It was
checked against all 512 neighbourhoods.

| Team | Case | Clauses | Literals each |
| --- | --- | --- | --- |
| Positive (alive next) | exactly 3 live neighbours, any centre | 56 | 8 |
| | centre alive and exactly 2 live neighbours | 28 | 9 |
| Negative (dead next) | 4 or more live neighbours | 70 | 4 |
| | at most 1 live neighbour | 8 | 7 |
| | centre dead and exactly 2 live neighbours | 28 | 9 |
| **Total** | | **190** (84 + 106) | |

- "Exactly" cases need long clauses: to be sure there are exactly 3 live
  neighbours, all 8 neighbours must be fixed. "At least" cases can be short:
  `N ∧ NE ∧ E ∧ SE` already guarantees death.
- 190 is not proven to be the minimum for this "any clause fires" style.
- **A TM can need fewer**, because it adds up votes instead of checking
  whether any clause fired. Example: "4 or more live neighbours → dead" takes
  70 clauses above, but 18 one-literal clauses do it with votes: `N`, `NE`, …
  (8 clauses) vote dead, `¬N`, `¬NE`, … (8 clauses) vote alive, and `C` and
  `¬C` both vote dead (exactly one of them always fires, adding +1). The score
  for dead is k + 1 − (8 − k) = 2k − 7, positive exactly when k ≥ 4. The full
  rule needs a window (k = 2 or 3), which one sum of single literals cannot
  express, so how few clauses a TM really needs is an open question that G1
  measures.

The reference set is built in code as a hard TM (section 9), used as a unit
test and printed next to the learned rules.

## 4. Model and training

### Relaxed TM

The model is the `RelaxedTM` from `cakit/tm/` without changes: 18 literals,
soft AND clauses with include-probability p = sigmoid(w), fixed positive and
negative teams of equal size, vote sum s, output ŷ = sigmoid((s − b)/τ),
hardened at the end (literal included iff p > 0.5; empty clauses do not vote).
See the classification [PLAN.md](../classification/PLAN.md), section 3.

| Setting | Value | Why |
| --- | --- | --- |
| Clauses (G0) | 256 (128 positive, 128 negative) | The reference needs 106 negative clauses; equal teams of 128 leave some spare |
| Loss | Binary cross-entropy | Same as E0–E4; squared error as one variant in G0 |
| Threshold b | Learned | Only 27% of the rows are alive, so a learned threshold helps |
| Optimizer | Adam, mini-batches of 32 | As in E4 |
| Epochs | 1,000 passes over the 512 rows (16,000 updates) | No benchmark exists; DiffLogic CA's training sees about 1,000 passes too (3,000 steps × 20 grids × 9 cells). The tuning pass checks whether fewer are enough |
| Reported model | The model after the final epoch | As in Granmo's protocol; no picking the best epoch |
| Seeds | 0–99 for reported runs; 1000–1019 for tuning | Tuning and reporting never share seeds |
| Tuned | Start mean of w (−3, −1.5), τ, learning rate (0.01, 0.05) | Section 5 |

**τ needs a new tuning pass.** With 256 clauses the vote sums can be much
larger than with 20 (E1) or 100 per class (E4), so the E0b and E4 values do
not carry over. The τ grid is 1, 3, 10, 30, 100.

**Tuning without a validation split.** All 512 rows are the data, so there is
nothing to hold out. Since G0 and G1 ask whether the rule can be
represented, tuning on the table is acceptable, but it is done on separate
seeds (1000–1019) and the criterion is the exact-rule rate on those seeds.

### Standard TM

Granmo's 2018 code, with the same total clause count, epochs and seeds. His
code has no single-output form, only the Multi-Class TM, so the standard TM
is a Multi-Class TM with 2 classes (dead, alive) and the total split evenly:
256 clauses = 128 per class. E1 used it the same way for noisy XOR. T and s
are tuned on the same tuning seeds: T ∈ {10, 30, 100}, s ∈ {3, 5, 10}.

### DiffLogic CA's setup, for comparison

From their notebook (`diffLogic_CA.ipynb`, Game of Life section):

| | DiffLogic CA | This plan |
| --- | --- | --- |
| Data | All 512 3×3 patterns, each as a tiny grid with wrap-around edges | All 512 rows directly |
| CA steps | 1 | 1 |
| Model | 16 perception kernels (first layer hand-wired: each neighbour paired with the centre) + update network of many 128-gate layers | One relaxed TM on the 9 bits, no hand-wired structure |
| Loss | Sum of squared errors | Binary cross-entropy (squared error as a variant) |
| Optimizer | AdamW, lr 0.05, weight decay 0.01, gradients clipped at 100 | Adam, tuned lr |
| Start | Every gate starts as a pass-through | w at a negative mean (most literals excluded) |
| Training | 3,000 steps of 20 grids, one seed | 1,000 epochs, 100 seeds |
| Evaluation | Hard loss on the training batch; a 128×128 grid run for 300 steps, judged by eye | Exact check over all 512 rows; grid comparison step by step (G3) |
| Result | Rule learned "perfectly"; 336 active gates | — |

## 5. Experiments

Run in order; each starts once the previous one works.

| # | Experiment | What varies | Question it answers |
| --- | --- | --- | --- |
| G0 | Learn the rule | Relaxed TM vs standard TM; tuning pass first; one squared-error run | Can the relaxed TM learn all 512 rows exactly? How often, and how fast? |
| G1 | Clause count | Total clauses 16, 32, 64, 128, 192, 256, 384, 512 | How does the exact-rule rate, the number of distinct clauses and the number of duplicates change with the clause count? Does the TM use voting to get below 190? |
| G3 | Run on a grid | Models from G0 and G1 | Does the learned rule produce the same states as Game of Life over many steps? |
| G5 | Single-output vs multi-class | One bank (single-output) vs two banks (one per class), same total clauses | Does splitting the clauses into an "alive" and a "dead" bank change the exact-rule rate, the duplicates or the rules learned? |

### G0: learn the rule

1. Tuning pass (relaxed TM): start mean × τ × learning rate = 2 × 5 × 2 =
   20 settings, 20 tuning seeds each. Choose the setting with the highest
   exact-rule rate; on ties, the one that reaches it in fewer epochs.
   Tuning pass (standard TM): T × s = 3 × 3 = 9 settings, same seeds.
2. Main runs: both TMs with their chosen settings, seeds 0–99.
3. Squared-error variant: the relaxed TM with the same settings but squared
   error instead of cross-entropy, seeds 0–99. Squared error pushes less
   once a prediction is nearly right, which works like a soft "enough" (as
   τ did in E0b), so it may leave fewer duplicates. It may also leave
   confidently wrong rows stuck. This run shows which.

### G1: clause count

- Both TMs, total clauses 16, 32, 64, 128, 192, 256, 384, 512 (equal teams),
  50 seeds each (0–49).
- Settings from G0. τ is re-tuned per clause count on the tuning seeds
  (vote sums grow with the clause count); start mean and learning rate stay.
- Expected picture: below about 190 clauses the "any clause fires" style is
  impossible, so an exact rule there means the TM uses voting. Above 190,
  extra clauses can only become duplicates or junk. The main plot is the
  number of distinct clauses against the total: if it levels off while the
  total keeps growing, every extra clause is a duplicate.
- Also report the clause count each TM needs to reach an exact rule in at
  least 90% of seeds.

### G3: run on a grid

- Models: the relaxed and standard TM from G0 (several seeds), the smallest
  exact model from G1, and a few models that are *not* exact (wrong on a
  few rows), to see what a few wrong rows do to the dynamics.
- Grids, all with wrap-around edges:
  - random 128×128 grid at density 0.5, 300 steps (as DiffLogic CA);
  - known patterns on a small grid: block (still), blinker (period 2),
    glider (moves one cell diagonally every 4 steps).
- Run the hardened TM and true Game of Life side by side from the same start.
  Record at every step the number of cells that differ, and the first step
  at which any cell differs.
- For an exact model the difference is always 0, so for those G3 is a check
  and a figure. The non-exact models are where it says something: which
  wrong rows matter (rows that occur often on real grids) and how fast an
  error spreads.
- Figures: side-by-side animations (learned vs true), and differing cells
  over time.

### G5: single-output vs multi-class

So far the TM has one clause bank: positive clauses vote alive, negative
clauses vote dead, and the sign of the vote sum decides. A multi-class TM, as
in E4 and Granmo's Multi-Class TM, has one bank per class instead:

```text
single-output:  one bank, + team votes alive, − team votes dead
                → alive if s ≥ b

multi-class:    bank "dead":  + team votes for dead,  − team against dead   → s_dead
                bank "alive": + team votes for alive, − team against alive  → s_alive
                → class with the larger vote sum
```

- **Models:** the relaxed TM as `MultiClassRelaxedTM` with 2 classes
  (softmax cross-entropy over the two vote sums, as in E4), compared with the
  single-output relaxed TM from G0/G1. The standard TM is already a two-class
  Multi-Class TM in G0/G1 (section 4), so its G0/G1 results are the
  reference here and it is not run again.
- **Same total clauses:** the multi-class TM gets the same total as the
  single-output TM, split evenly over the two banks: 128, 256 and 512 in
  total (so 64, 128 and 256 per bank). 50 seeds each (0–49).
- **Settings:** start mean and learning rate from G0; τ re-tuned on the
  tuning seeds. Ties between the two vote sums are predicted as dead, the
  majority class; the class order is fixed as [dead, alive] so argmax does
  this.
- **What to look at:**
  - exact-rule rate and epochs to exact, against the single-output TM with
    the same total;
  - distinct clauses and duplicates, per bank and across banks. In the
    multi-class TM, evidence for dead can be learned twice: by the dead
    bank's + team and by the alive bank's − team. If both banks learn the
    same clauses, the second bank adds duplicates, not new rules;
  - pure clauses (section 7) and overlap with the reference set, per bank.
- **Why it matters later:** a CA with more than two states, or more than one
  output bit per cell, needs more than one output, so this shows what the
  multi-class form costs or gains on a task where the right answer is known.

## 6. Not planned now

- **G2: held-out test (future investigation).** Train on part of the table
  (e.g. 400 rows) and test on the rest (112). A TM that learned compact
  rules (clauses covering many rows, or voting) should still get the held-out
  rows right; one that memorised one row per clause cannot. This would
  separate "represents the rule" from "generalises the rule", which neither
  G0 nor DiffLogic CA can do.
- **G4: training through several unrolled steps.** Dropped. For Game of Life
  one step already holds all the information.

## 7. Metrics

All accuracies are labelled hard or soft.

**Correctness**

- Hard accuracy over the 512 rows, and soft accuracy; soft–hard gap.
- **Exact-rule rate:** the fraction of seeds whose hardened model gets all
  512 rows right. The headline number.
- Wrong rows: for non-exact models, which neighbourhoods are wrong, grouped by
  the centre and the number of live neighbours k.

**Clauses (the clause-count question)**

- Non-empty, empty and contradictory clauses.
- Distinct clauses per team, and the duplicate fraction = 1 − distinct /
  non-empty.
- Literals per clause (mean and distribution).
- **Pure clauses:** a positive clause is pure if it fires only on rows that
  lead to alive (a negative one: only on rows that lead to dead). If every
  clause is pure, the model works in the "any clause fires" style of the
  reference set; impure clauses mean the model relies on voting.
- How many learned clauses appear in the reference set (section 3).
- The hardened rules printed as text, merged duplicates shown with a count,
  e.g. `+ : N ∧ NE ∧ E ∧ ¬SE ∧ … (×3)`.

**Training behaviour**

- First epoch at which the hard model is exact, and whether it stays exact.
- Share of undecided literals (0.1 < p < 0.9) at the end.
- Training time.

**Grid (G3)**

- Differing cells per step and the first differing step, per model and grid.

**Size, for the comparison with DiffLogic CA**

- Number of clauses and included literals of the hardened model, reported
  next to DiffLogic CA's 336 active gates. They are not the same unit, so
  both are reported side by side rather than one called smaller.

## 8. Success criteria and decision rules

**Counts as working if:**

- G0: the relaxed TM is exact (512 of 512, hard) in at least 90% of seeds,
  with a soft–hard gap of at most 1 point.
- G1: clear curves of exact-rule rate, distinct clauses and duplicates
  against the clause count, for both TMs.
- G3: every exact model matches Game of Life on every grid for all steps.

**What to do if not:**

| Symptom | Likely cause | Next step |
| --- | --- | --- |
| Hard accuracy high but rarely 512/512 | A few long "exactly" clauses (8–9 literals) never finish | More epochs; entropy penalty late in training; check which rows are wrong |
| Soft good, hard much lower | Many undecided p | Entropy penalty, lower τ; gradient grafting as fallback |
| Neither improves | Clauses dead or vote saturated | Change start mean and τ (larger τ with many clauses) |
| Exact only with many more clauses than 190 | Many duplicates or junk clauses | Report as a G1 result; compare with the standard TM |
| Standard TM fails at all settings | T or s outside the grid | Widen the grid before comparing |

## 9. Implementation checklist

New stage folder `experiments/relaxed_tm/game_of_life/`, same layout as
`classification/`. The model, hardening and rule printer are reused from
`cakit/tm/`, the training loop from `common/train.py`.

- [ ] `data.py`: the 512-row table (bit order as in section 2), a true Game of
  Life step on a grid with wrap-around edges, 3×3 patch extraction from a
  grid, and the known patterns (block, blinker, glider).
- [ ] `reference.py`: the 190-clause reference set as a hard TM, with
  position names for the rule printer.
- [ ] `common/train.py`: an optional squared-error loss, off by default, so
  existing runs do not change.
- [ ] Clause analysis: distinct clauses, duplicate fraction, pure clauses,
  overlap with the reference set, merged rule printing.
- [ ] `run_g0.py` (tuning pass and main runs, both TMs, `--loss` option),
  `run_g1.py` (clause sweep), `run_g3.py` (grid runs), one `plot_gN.py` per
  experiment.
- [ ] `run_g5.py`: two-class `MultiClassRelaxedTM` and Granmo's Multi-Class TM
  on the table, with the per-bank and cross-bank clause analysis.
- [ ] Hard TM applied to a whole grid at once (all 3×3 patches as one batch).
- [ ] Unit tests: the table has 140 alive rows; the reference set gets 512
  of 512; the true step keeps a block, flips a blinker and moves a glider by
  one cell every 4 steps; patches from a random grid match the table.
- [ ] Reports in `reports/`, results as JSON in `results/`; README status
  table updated after each experiment.

## Sources

- [Miotti et al. — Differentiable Logic Cellular Automata (arXiv:2506.04912)](https://arxiv.org/html/2506.04912v1)
- [DiffLogic CA notebook (google-research/self-organising-systems)](https://github.com/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb)
- [Granmo — The Tsetlin Machine (arXiv:1804.01508)](https://arxiv.org/pdf/1804.01508)
- Classification plan and results: [../classification/PLAN.md](../classification/PLAN.md), E0b and E4 reports in [../classification/reports/](../classification/reports/)
