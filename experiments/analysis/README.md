# Analysis scripts

Post-processing for the cross-class pair batch in `experiments/cla/`. Nothing here
changes how the CLA runs; it adds metrics and applies the filtering funnel.

| Script | What it does | Writes |
|---|---|---|
| `fast_sim.py` | Vectorised, bit-identical replay of a pair run (grid + TA states + decisive mask). `--check` proves it matches cakit. | nothing (library + check) |
| `axes_metrics.py` | Adds the `neutral` domination tag, the `tail_only` settle tag, and the TA-layer descriptors to every pair. | `results_<fb>/**/metrics.json`, `results_<fb>/summary.csv` (plus `spacetime_decisive.png` with `--plots`, `history.npz` with `--save-histories`) |
| `decisive_plot.py` | Draws the decisive cell-step diagram in the same layout as the GUI exports. Used by `axes_metrics.py --plots`. | nothing on its own |
| `funnel.py` | Applies the four filter stages and documents each one. | `experiments/funnel.md`, `results_<fb>/funnel.json` (and `experiments/highlights.md` with `--as-highlights`) |

## Order of operations

From the repository root:

```bash
python experiments/analysis/fast_sim.py --check            # once, or after touching cakit
python experiments/analysis/axes_metrics.py --verify-png   # both feedbacks, ~2 min
python experiments/analysis/axes_metrics.py --plots        # decisive diagrams, ~10 min
python experiments/analysis/funnel.py --as-highlights
python experiments/catalog/generate.py                      # refresh the catalog
```

`--verify-png` decodes every `spacetime_bw.png` and checks it equals the simulated
grid; drop it on later runs once it has passed.

**Re-run `axes_metrics.py` after any `run_cross_class_pairs.py --backfill-hamming`.**
The backfill rewrites `metrics.json` and `summary.csv` from scratch and drops the
fields added here. `axes_metrics.py` is idempotent, so running it twice is safe.

## Tags added to the existing axes

- **Domination** gains `neutral`: occupancy says `coexist`, but fewer than 2% of
  the cell-steps in the last 100 generations are *decisive* (the two rules
  disagree on the neighbourhood the cell sees, so its arm choice matters). The
  original occupancy tag is kept as `domination_occupancy`.
- **Settle** gains `tail_only`: `_settle_metrics` found no start before the tail
  that reaches the tail's cleanliness and fell back to `settled_at = 334`. The
  original tag is kept as `settle_raw`.

The catalog builds its filter menus from the values it finds, so both new tags
appear there after `generate.py`.

## Funnel stages

1. remove `dominate_complete`
2. remove `neutral`
3. remove exact orbits (`fixed`, `cycle`, `drift`); the ones that settled `late`
   are set aside as their own group, not discarded
4. remove `early` settling

`--as-highlights` turns the survivors (plus the set-aside group) into
`experiments/highlights.md`, grouped by class pair and by label · settle, with the
most evenly contested pairs first. The previous hand-written file is kept as
`experiments/highlights_manual.md`.
