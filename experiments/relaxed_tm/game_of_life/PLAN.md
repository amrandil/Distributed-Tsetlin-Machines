# Experiment Plan: Relaxed TM on Game of Life

Nov 8, 2026 · Amr Kandil · revised the same day

> Progress on each experiment is tracked in [README.md](README.md); results
> go in `reports/`. A first version of G0 (different architectures and clause
> budgets) is archived in [archive/g0_v1_2026-11-08/](archive/g0_v1_2026-11-08/);
> it is not used for any choice in this plan.

This experiment moves the relaxed TM from plain classification to a cellular
automaton rule. It asks: can the relaxed TM learn Conway's Game of Life
exactly; how does it compare with the standard TM **in the same
architecture**; and how many clauses does each need.

## 1. Goal and scope

**In scope**

- A fair comparison of the relaxed TM and the standard TM in two
  architectures, with the same number of clauses per TM:

  | | Single-output: one TM votes for or against "alive" | Multi-class: one TM per class (dead, alive), argmax |
  | --- | --- | --- |
  | Relaxed | `RelaxedTM` | `MultiClassRelaxedTM`, 2 classes |
  | Standard | Granmo's 2018 `TsetlinMachine.pyx` | Granmo's 2018 `MultiClassTsetlinMachine.pyx`, 2 classes |

- First with 744 clauses per TM, enough for a full lookup table (G0), then with fewer (G1).
- One TM shared by every cell, trained on one step of Game of Life (the same
  setup as DiffLogic CA's Game of Life experiment, section 8).
- Stefano's question from E0: how the number of clauses relates to duplicate
  clauses (G1).
- Running the learned rules on a grid (G3).

**Out of scope for now**

- A held-out test (G2, future work, section 7).
- Per-cell TMs that learn locally; other CA tasks.

**What a successful result looks like:** for each of the four models, we know
how often it learns all 512 neighbourhoods exactly at 744 clauses per TM, and
how that changes as the clause count shrinks; and the learned rules, run on a
large grid, produce exactly the same states as Game of Life.

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
  G1 is whether a TM can *represent* the rule exactly, not whether it
  generalises to unseen neighbourhoods (that would be G2). This is also what
  DiffLogic CA tested.
- One epoch is one pass over all 512 rows, in a new random order each time.
- Sampling 3×3 patches from random grids at density 0.5 gives every
  neighbourhood with equal probability, so training on random grids would be
  the same data as the table. DiffLogic CA's tiny 3×3 grids with wrap-around
  edges also contain only rows of this table, each one 9 times.

## 3. How many clauses can Game of Life need?

### 744: the most it can ever need

A clause that fixes all 9 bits (9 literals) fires on exactly one
neighbourhood: a one-row clause. With enough one-row clauses a TM becomes a
complete lookup table. How many it needs depends on which class a model
predicts **when no clause fires**, because rows of that class need no clause:

- **Single-output TM:** predicts alive when the vote sum is ≥ 0. A row where
  no clause fires has sum 0 and comes out alive for free; each of the
  **372 dead rows** needs a negative one-row clause.
- **Multi-class TM:** predicts the class with the larger vote sum; on a tie,
  Granmo's code keeps class 0, which is dead here. A row where no clause fires
  in either TM is a tie (0 against 0) and comes out dead for free; each of the
  **140 alive rows** needs a positive one-row clause in the alive TM.

A TM splits its clauses into two equal teams, so a lookup table needs 372 per
team in the single-output TM: **744 clauses per TM** (372 + 372). With 744
per TM, both architectures can hold a full lookup table of Game of Life, so
no solution, including pure memorisation, can need more. This is the maximum
for Game of Life, not for any rule on 9 inputs (a rule with more than 372
rows of one class would need more). G0 uses 744 clauses per TM; in the
multi-class architecture each class has its own TM of 744, so the model has
1,488 in total.

In practice a TM should need far fewer, because clauses can cover many rows
(`N ∧ NE ∧ E ∧ SE`, "4 or more live neighbours", covers 32 rows) and votes
can add up.

### 190: a hand-built exact clause set

A clause is an AND of literals, so one clause covers a whole group of
neighbourhoods; the fewer literals, the more it covers (each free bit doubles
its coverage). This clause set implements Game of Life exactly in the "any
clause fires" style: a positive clause fires only on rows that lead to alive,
a negative clause only on rows that lead to dead, and every row is covered by
a clause of the right sign. It was checked against all 512 rows.

| Team | Case | Clauses | Literals each |
| --- | --- | --- | --- |
| Positive (alive next) | exactly 3 live neighbours, any centre | 56 | 8 |
| | centre alive and exactly 2 live neighbours | 28 | 9 |
| Negative (dead next) | 4 or more live neighbours | 70 | 4 |
| | at most 1 live neighbour | 8 | 7 |
| | centre dead and exactly 2 live neighbours | 28 | 9 |
| **Total** | | **190** (84 + 106) | 1,288 literals in total |

- 190 is not proven to be the minimum for this style.
- **A TM can need fewer**, because it adds up votes instead of checking
  whether any clause fired. Example: "4 or more live neighbours → dead" takes
  70 clauses above, but 18 one-literal clauses do it with votes: `N`, `NE`, …
  (8 clauses) vote dead, `¬N`, `¬NE`, … (8 clauses) vote alive, and `C` and
  `¬C` both vote dead (exactly one of them always fires, adding +1). The score
  for dead is k + 1 − (8 − k) = 2k − 7, positive exactly when k ≥ 4. How few
  clauses a TM really needs is what G1 measures.

## 4. Models

All four models get the same data, the same epochs, the same seeds and the
same number of clauses per TM.

### Relaxed TM

The `RelaxedTM` from `cakit/tm/` (see the classification
[PLAN.md](../classification/PLAN.md), section 3): 18 literals, soft AND
clauses with include-probability p = sigmoid(w), equal positive and negative
teams, vote sum s, hardened at the end (literal included iff p > 0.5; empty
clauses do not vote).

- **Single-output:** output ŷ = sigmoid(s / τ), binary cross-entropy;
  prediction alive iff the hard vote sum ≥ 0.
- **Multi-class:** `MultiClassRelaxedTM`, one TM per class (dead, alive);
  softmax cross-entropy over the two vote sums divided by τ; prediction is the
  class with the larger hard vote sum, ties to dead (class order [dead,
  alive]).
- **No learned threshold (b = 0)** in either, because Granmo's TMs have none.
  A learned threshold is an ablation after G0 (section 6).
- Adam, mini-batches of 32 (16 updates per epoch), 1,000 epochs.

### Standard TM

Granmo's 2018 code from the cair/TsetlinMachine repository, 100 states per
action, 1,000 epochs:

- **Single-output:** `TsetlinMachine.pyx`, from the same commit as the
  Multi-Class file: one clause bank, even clauses positive and odd ones
  negative, prediction alive iff the vote sum ≥ 0. Added to
  `classification/granmo2018/` with an `.orig` copy and the same two changes
  as the Multi-Class file. Unlike the Multi-Class code, an empty clause votes
  (outputs 1) at prediction too; its predictions are taken from his own code.
- **Multi-class:** `MultiClassTsetlinMachine.pyx`, the version used in E1,
  with 2 classes. His code takes the total and splits it over the classes,
  so 744 per class means 1,488 in total.
- `boost_true_positive_feedback` is off: his single-output TM does not have
  it, so both standard TMs are run without it.

### Shared rules

| Setting | Value |
| --- | --- |
| Clauses per TM | G0: 744 (372 +, 372 −). G1: fewer |
| Epochs | 1,000; the model after the final epoch is reported (no picking the best epoch) |
| Seeds | Tuning: 1000–1019. Reported runs: 0–99 (G0), 0–49 (G1) |
| Exact | All 512 rows right, hard |

## 5. Tuning (G0)

Every model is tuned from scratch, with grids fixed here before any run.

| Model | Grid | Settings |
| --- | --- | --- |
| Relaxed, single-output and multi-class | start mean of w ∈ {−3, −1.5, −0.5} × τ ∈ {0.3, 1, 3, 10} × learning rate ∈ {0.01, 0.05, 0.2} | 36 each |
| Standard, single-output and multi-class | T ∈ {10, 30, 100, 300} × s ∈ {3, 10, 30, 100} | 16 each |

Why these ranges:

- **Relaxed:** start mean from mostly-included (−0.5, p ≈ 0.38) to mostly
  excluded (−3, p ≈ 0.05); τ over two orders of magnitude (with 744 clauses
  vote sums can reach the hundreds); learning rates around Adam's usual range.
- **Standard:** T is the target vote sum, up to about the size of one team
  (372). s sets how specific clauses become: a clause tends to settle on
  patterns that occur in roughly 1 in s of the examples it learns from, and
  the rarest Game of Life patterns are single births (2 of the 140 alive
  rows), so s goes up to 100.

**Selection rule**, the same for all four models: highest exact-rule rate on
the 20 tuning seeds; ties go to the setting that is first exact after fewer
epochs (median), then to the higher mean hard accuracy.

**One extension, decided in advance:** if a model's chosen setting lies on
the edge of its grid, that model gets one more pass that extends the grid one
step beyond that edge, on that axis only (e.g. s = 300 if s = 100 was chosen).
No other changes to the grids after seeing results.

**Time on the Mac (8 workers), roughly:** tuning about 5 hours in total (most
of it the multi-class models with 1,488 clauses), main runs about 1.5 hours. The
tuning seeds can be cut to 10 if needed, for all four models alike.

## 6. Experiments

Run in order; each starts once the previous one works.

| # | Experiment | What varies | Question it answers |
| --- | --- | --- | --- |
| G0 | Architectures at 744 clauses per TM | Relaxed vs standard × single-output vs multi-class; each tuned (section 5), then 100 seeds | Which models learn all 512 rows exactly, how often and how fast, and with what clauses? |
| G1 | Fewer clauses | Clauses per TM 744, 512, 256, 128, 64, 32, 16 | How does the exact-rule rate, the number of distinct clauses and the number of duplicates change with the clause count? Do the TMs use voting to need fewer clauses than the 190-clause set? |
| G3 | Run on a grid | Models from G0 and G1 | Do the learned rules produce the same states as Game of Life over many steps? |

### G0: architectures at 744 clauses per TM

1. Tuning passes for the four models (section 5).
2. Main runs: each model at its chosen setting, seeds 0–99.
3. Ablations, only after the main runs: learned threshold b for the relaxed
   single-output TM; squared error instead of cross-entropy for the relaxed
   TMs (DiffLogic CA's loss); `boost_true_positive_feedback` for the standard
   multi-class TM. Each at the model's chosen setting, 100 seeds. Boost
   exists only in Granmo's April 2018 Multi-Class code, so that ablation also
   switches to that version of his code.

### G1: fewer clauses

- All four models, 744, 512, 256, 128, 64, 32 and 16 clauses per TM, 50
  seeds each (0–49).
- Settings start from G0. Settings that depend on the clause count (τ for the
  relaxed TMs, T for the standard TMs) are re-tuned per clause count on the
  tuning seeds with a small grid of the same size for every model; the exact
  grids are fixed after G0 and before G1 runs.
- Main plots: exact-rule rate, distinct clauses and duplicate share against
  the clause count, one line per model. If distinct clauses level off while
  the total keeps growing, every extra clause is a duplicate.
- Also report the smallest clause count at which each model is exact in at
  least 90% of seeds.

### G3: run on a grid

- Models: several exact models from G0, the smallest exact ones from G1, and
  a few non-exact ones, to see what a few wrong rows do to the dynamics.
- Grids, all with wrap-around edges: a random 128×128 grid at density 0.5,
  300 steps (as DiffLogic CA); and known patterns on a small grid: block
  (still), blinker (period 2), glider (moves one cell diagonally every 4
  steps).
- Run the hardened TM and true Game of Life side by side from the same start;
  record the number of differing cells at every step and the first step with
  any difference. Figures: side-by-side animations and differing cells over
  time.

## 7. Not planned now

- **G2: held-out test (future investigation).** Train on part of the table
  (e.g. 400 rows) and test on the rest (112). A TM that learned compact rules
  (clauses covering many rows, or voting) should still get the held-out rows
  right; one that memorised one row per clause cannot. This would separate
  "represents the rule" from "generalises the rule", which neither G0 nor
  DiffLogic CA can do.
- **G4: training through several unrolled steps.** Dropped. For Game of Life
  one step already holds all the information.

## 8. DiffLogic CA's setup, for comparison

From their notebook (`diffLogic_CA.ipynb`, Game of Life section):

| | DiffLogic CA | This plan |
| --- | --- | --- |
| Data | All 512 3×3 patterns, each as a tiny grid with wrap-around edges (each of the 9 cells is one table row) | All 512 rows directly |
| CA steps | 1 | 1 |
| Model | 16 perception kernels (first layer hand-wired: each neighbour paired with the centre) + update network of many 128-gate layers | TMs on the 9 bits, no hand-wired structure |
| Loss | Sum of squared errors | Cross-entropy (squared error as an ablation) |
| Optimizer | AdamW, lr 0.05, weight decay 0.01, gradients clipped at 100 | Adam, tuned learning rate |
| Start | Every gate starts as a pass-through | w at a tuned negative mean |
| Training | 3,000 steps of 20 grids (≈ 117 passes over the 512 grids, ≈ 1,055 over the rows), one seed | 1,000 epochs, 100 seeds |
| Evaluation | Hard loss on the training batch; a 128×128 grid run for 300 steps, judged by eye | Exact check over all 512 rows; grid comparison step by step (G3) |
| Result | Rule learned "perfectly"; 336 active gates | — |

## 9. Metrics

All accuracies are labelled hard or soft.

**Correctness**

- Hard accuracy over the 512 rows; for the relaxed TMs also soft accuracy and
  the soft–hard gap.
- **Exact-rule rate:** the fraction of seeds whose final model gets all 512
  rows right. The headline number.
- **Ever exact** (at any epoch), first epoch at which the model is exact, and
  the share of the last 100 epochs at which it is exact (stability).
- Wrong rows, grouped by the centre state and the number of live neighbours k.

**Clauses** (per TM; for multi-class models also across the two TMs)

- Active, empty and contradictory clauses.
- Distinct clauses, and the duplicate share = 1 − distinct / active.
- Literals per clause (mean and distribution).
- **Pure clauses:** a clause that fires only on rows whose label is the one it
  votes for. All pure means the "any clause fires" style; impure clauses mean
  the model relies on voting.
- How many learned clauses are in the 190-clause set, per case.
- Hardened rules printed as text, duplicates merged with a count.

**Training**

- Hard accuracy per epoch (all models); loss and share of undecided literals
  per epoch (relaxed TMs); training time.

**Size, for the comparison with DiffLogic CA**

- Clauses and included literals of the hardened model, reported next to
  DiffLogic CA's 336 active gates (different units, so side by side only).

## 10. Implementation checklist

- [x] `data.py`: the 512-row table, true Game of Life step with wrap-around
  edges, 3×3 patches, block / blinker / glider.
- [x] `reference.py`: the 190-clause set as a hard TM.
- [x] `clauses.py`: distinct, duplicates, pure, overlap with the 190-clause
  set, merged rules, compact model saving.
- [x] `common/train.py`: optional squared-error loss (off by default).
- [x] Unit tests (`tests/tm/test_game_of_life.py`).
- [x] Granmo's `TsetlinMachine.pyx` in `classification/granmo2018/`, with a
  wrapper and conversion to `HardTM` checked against his own predictions.
- [x] `run_g0.py` for all four models (`--model relaxed|standard`,
  `--arch single|multi`), the grids of section 5, the selection rule, the
  edge extension, main runs and ablations.
- [x] Clause analysis for multi-class models: per TM and across TMs.
- [x] Squared error for the multi-class relaxed TM (softmax probabilities vs
  one-hot label) in `common/train.py`.
- [ ] `plot_g0.py`, report `reports/G0_architectures.md`.
- [ ] `run_g1.py`, `run_g3.py` and their plots and reports.

## Sources

- [Miotti et al. — Differentiable Logic Cellular Automata (arXiv:2506.04912)](https://arxiv.org/html/2506.04912v1)
- [DiffLogic CA notebook (google-research/self-organising-systems)](https://github.com/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb)
- [Granmo — The Tsetlin Machine (arXiv:1804.01508)](https://arxiv.org/pdf/1804.01508)
- [cair/TsetlinMachine — `TsetlinMachine.pyx` and `MultiClassTsetlinMachine.pyx`](https://github.com/cair/TsetlinMachine)
- Classification plan and results: [../classification/PLAN.md](../classification/PLAN.md), reports in [../classification/reports/](../classification/reports/)
