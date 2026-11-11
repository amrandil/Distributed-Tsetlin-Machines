# Archive: first version of G0 (2026-11-08)

The first G0 runs, kept for reference only. They are **not** used for any
choice in the current plan (settings, grids, conclusions).

What was different from the current plan:

- Relaxed TM: single-output, 256 clauses, learned threshold b.
- Standard TM: Granmo's 2018 Multi-Class TM with 2 classes and 256 clauses in
  total (128 per class), so the two models did not have the same architecture
  or the same clauses per TM.
- Tuning was done in two passes per model, the second chosen after seeing the
  first.

Contents: the plan of that time (`PLAN_v1.md`), the code (`code/`), all
results (`results/`) and the report with its figures (`reports/`). The
report's figure links work from inside this folder.
