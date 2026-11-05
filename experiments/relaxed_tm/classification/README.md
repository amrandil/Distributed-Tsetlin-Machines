# Tier 0: Relaxed TM on Noisy XOR and Iris

Does a relaxed Tsetlin Machine, trained by gradient descent and then hardened
back to 0/1 logic, learn small classification tasks as well as a standard
Tsetlin Machine, and give readable clauses? No cellular automata yet.

Full plan (model, data, protocol, hyperparameters, success criteria):
[PLAN.md](PLAN.md).

## Experiments

| # | Experiment | Script | Status |
| --- | --- | --- | --- |
| E0 | Sanity check: clean 2-bit XOR | `run_e0.py` | Done: [report](reports/E0_clean_xor.md) |
| E0b | Why duplicate clauses form (4 clauses, 50 seeds per setting) | `run_e0b_duplicates.py` | Done: [report](reports/E0b_duplicate_clauses.md) |
| E1 | Main result: noisy XOR, benchmark protocol | `granmo2018/repeat_demo.py` (reference), `run_e1_baseline.py` | Part 1 done: Granmo's code, his setup, 99.92% ± 0.06 (published 99.3% ± 0.3), [report](reports/E1_noisy_xor.md). Part 2 (relaxed TM) to do |
| E2 | Clause count on noisy XOR | | To do |
| E3 | Noise robustness on noisy XOR | | To do |
| E4 | Main result: Iris, benchmark protocol | | To do |
| E5 | Ablations (τ, init, λ₁, λ₂, bias, Iris encoding) | | To do |
| E6 | Baselines (standard TM, dNL-style, logistic regression) | | To do |

New experiments follow the same pattern: `run_eN.py` here, a write-up in
`reports/EN_<name>.md`, raw output in `results/eN_<name>*.json`.

## Files

| File | Contents |
| --- | --- |
| `data.py` | Clean XOR; Granmo's noisy XOR files and a generator for other noise levels; Iris measurements; Granmo's Binary Iris file; 80/20 split |
| `run_e0.py` | E0: clean XOR, several seeds, records rules, epochs to 100% and per-epoch training curves |
| `plot_e0.py` | E0 figures (training curves, results per clause count, include-probabilities over epochs) |
| `run_e0b_duplicates.py`, `plot_e0b.py` | E0b: starting point, margin, τ and learning rate vs. duplicate clauses; figure |
| `baselines.py` | Standard TM via `tmu` (the only file that imports it); converts its clauses to our `HardMultiClassTM` |
| `install_tmu.sh` | Installs tmu at the pinned commit (see below) |
| `xor_rules.py` | Checks learned clauses against the XOR rule: all four XOR clauses present, number of extra clauses, exact match |
| `run_e1_baseline.py` | Standard TM on noisy XOR, Granmo's settings, 100 seeded runs in our harness: `--impl tmu` or `--impl granmo2018` |
| `granmo2018/` | Granmo's original 2018 code and noisy XOR demo (MIT licence), with `.orig` copies; `repeat_demo.py` runs his demo 100 times his way (E1 reference) |
| `reports/` | One write-up per experiment; figures in `reports/figures/` |
| `results/` | Raw results as JSON |

`data.py` downloads Granmo's files from `cair/TsetlinMachine` on first use
and caches them in `data/` (git-ignored).

## Implementation checklist (PLAN.md section 10)

- [x] Data loaders (noisy XOR files and generator, Iris, thermometer encoder, 80/20 split, Binary Iris file)
- [x] Model: `RelaxedTM`, `MultiClassRelaxedTM` (`cakit/tm/relaxed.py`)
- [x] Hardening with Granmo's empty-clause convention (`cakit/tm/harden.py`)
- [x] Rule printer (`cakit/tm/rules.py`)
- [x] Training loop with penalties and per-epoch soft/hard accuracy (`../common/train.py`)
- [ ] Logging and reporting of all section 7 metrics per run (summary statistics done; training accuracy, timing and size still to add)
- [x] Unit tests (`tests/tm/`)
- [ ] Baselines: standard TM via `tmu` (done for noisy XOR), logistic regression
- [ ] Energy measurement with CodeCarbon
- [ ] Run E0 → E6 (E0 done)

## Run

```bash
python -m experiments.relaxed_tm.classification.run_e0 --clauses 10
python -m experiments.relaxed_tm.classification.run_e1_baseline        # ~1 h on 2 cores
```

The standard-TM baseline needs `tmu`, which is not in `requirements.txt`.
Install it once with `bash experiments/relaxed_tm/classification/install_tmu.sh`:
the script builds a pinned commit of `cair/tmu`, because the PyPI release and
the latest tag fail under NumPy 2, and removes an Intel-only compiler flag that
breaks the build on Apple Silicon. Without tmu, everything else runs and its
tests are skipped.

## E0 summary

10/10 seeds recover the exact XOR rule with 6, 8, 10 or 20 clauses; 5/10 with
the minimum of 4 clauses, where two clauses can learn the same case. Details:
[reports/E0_clean_xor.md](reports/E0_clean_xor.md).
