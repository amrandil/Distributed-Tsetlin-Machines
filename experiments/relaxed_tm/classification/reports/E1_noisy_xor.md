# E1 — Main result: noisy XOR

E1 has two parts: (1) reproduce Granmo's (2018) standard Tsetlin Machine
result by running his original code in his own setup, as the reference;
(2) run the relaxed TM under the same protocol. This report covers part 1.
Part 2 will be added below.

Experiment plan: [PLAN.md](../PLAN.md), sections 2, 4 and 8.

## Part 1: Standard TM (reference)

**Result:** running Granmo's original code and noisy XOR demo, unchanged, 100
times gives **99.92% ± 0.06** mean test accuracy, against **99.3% ± 0.3**
published in his Table 15. Our reproduction is slightly better and much more
consistent: our worst run reaches 98.06%, his 91.6%.

Run on 2026-11-05, branch `relaxed-tm`. Raw results:
[`../results/e1_granmo2018_demo_noisy_xor.json`](../results/e1_granmo2018_demo_noisy_xor.json).

### Question

Does Granmo's own code, run his way on his data, reproduce his published
noisy XOR result? This sets the reference the relaxed TM is compared with.

### Setup

| Part | Granmo (2018) | Our run |
| --- | --- | --- |
| Data | 10,000 examples, 12 binary inputs; y = XOR of two inputs (x1, x2 in the released files), the other 10 random. Released as files in `cair/TsetlinMachine` | Same (his released files, unchanged since April 2018) |
| Train / test split | 50% for training: 5,000 / 5,000, the fixed split of the released files | Same |
| Label noise | 40% of outputs inverted; test outputs unperturbed | Same (39.7% of training labels inverted in the file; test labels clean) |
| TM | Multi-Class Tsetlin Machine (Section 6.1; footnote 7) | Same |
| Clauses | 20; his code splits them over the 2 classes: 10 per class, 5 positive and 5 negative | Same |
| s, T | s = 3.9, T = 15 | Same |
| Boosted true-positive feedback | Not used (footnote 7 does not mention it, unlike the Iris, digits and MNIST experiments) | Same (off) |
| Tsetlin Automaton states | "Each has 100 states"; his code uses 100 per action, 200 in total | Same |
| Epochs | 200; accuracy after the final epoch is reported | Same |
| Runs | 100 replications with different random number streams | Same: 100 fresh processes, each running his demo once, with no seeding by us |
| Code | His `MultiClassTsetlinMachine.pyx` and `NoisyXORDemo.py` (`cair/TsetlinMachine`) | Same files, commit `3cb8c89`. Changes needed for Python 3 only: `print(...)` syntax in the demo and the Cython setting `language_level=2` (see [`../granmo2018/`](../granmo2018/)) |
| Reported statistics | Test accuracy: mean ± 95% CI, 5th / 95th percentile, min, max | Same, plus training accuracy |

His code has changed since 2018, but every change after June 2018 leaves its
behaviour the same. The one earlier change (June 5, 2018) concerns empty
clauses: before it, an empty clause voted at prediction; since then it does
not. We ran the current version.

### Results

| Measure | Granmo (2018), published (Table 15) | Our reproduction |
| --- | --- | --- |
| Runs | 100 | 100 |
| Test accuracy, mean ± 95% CI | 99.3% ± 0.3 | **99.92% ± 0.06** |
| 5th / 95th percentile | 95.9 / 100.0 | 99.28 / 100.00 |
| Min / max | 91.6 / 100.0 | 98.06 / 100.00 |
| Runs ≥ 99% | — | 97 / 100 |
| Training accuracy, mean | — | 60.29% |

92 of the 100 runs reach 100% test accuracy; the other 8 lie between 98.06%
and 99.42%.

Training accuracy is about 60% in every run. That is expected: with 40% of
training labels inverted, a model that has learned XOR exactly agrees with
only the 60% of labels that were not inverted.

### Comments

**The reproduction differs a bit from the published result.** The mean is
0.6 percentage points higher, and the spread is much smaller: his worst 5% of
runs fall below 95.9% and his worst run reaches 91.6%, while ours stay above
99.28% and 98.06%. A possible reason, not tested: the paper first appeared in
April 2018, when empty clauses still voted at prediction (see Setup). An empty
clause fires on every input, and with only 10 clauses per class it can tip
close votes, which would cause occasional bad runs like his. For the relaxed
TM, the reference is our reproduction (99.92%), which uses the same data and
statistics as part 2 will.

**Decision needed for part 2.** Granmo's "20 clauses" are a 2-class
Multi-Class TM (2 × 10), while the plan's E1 uses a single-output relaxed TM
with 20 clauses. Both have 20 clauses, but the structure differs. Options:

- run the relaxed TM as a 2-class `MultiClassRelaxedTM` with 10 clauses per class: the same structure as the reference, so the cleanest comparison;
- run it as a single-output `RelaxedTM` with 20 clauses, as planned, and note the difference;
- run both.

### Reproduce

```bash
pip install cython   # compiles his .pyx on first use (needs a C compiler)
python -m experiments.relaxed_tm.classification.granmo2018.repeat_demo --runs 100
```

Runs are not seeded, as in his setup, so the numbers vary slightly from one
repetition of the 100 runs to the next. His C random-number generator
(`rand()`) also differs between operating systems; our runs used Linux.
