# Relaxed TM on Game of Life

The relaxed TM as a cellular automaton rule: one TM shared by every cell,
learning Conway's Game of Life from its 512-row table, compared with the
standard TM in the same architecture. Plan: [PLAN.md](PLAN.md).

| # | Experiment | Scripts | Status |
| --- | --- | --- | --- |
| G0 | Architectures at 744 clauses per TM: relaxed vs standard × single-output vs multi-class | `run_g0.py`, `plot_g0.py` | Done: relaxed TMs exact in 100/100 seeds (single-output and multi-class); standard TMs 0/100 (93.9% and 96.3%). [Report](reports/G0_architectures.md) |
| G1 | Fewer clauses: exact-rule rate, distinct clauses, duplicates | | To do |
| G2 | Held-out test | | Future work |
| G3 | Run the learned rules on a grid, compare with Game of Life | | To do |
| G4 | Training through several unrolled steps | | Dropped |

An earlier version of G0 is archived in
[archive/g0_v1_2026-11-08/](archive/g0_v1_2026-11-08/) and not used for any
choice in the current plan.
