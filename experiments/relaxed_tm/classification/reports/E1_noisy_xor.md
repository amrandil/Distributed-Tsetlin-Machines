# E1 — Main result: noisy XOR

E1 has two parts: (1) run the standard Tsetlin Machine ourselves under
Granmo's (2018) protocol, as the reference; (2) run the relaxed TM under the
same protocol. This report covers part 1. Part 2 will be added below.

Experiment plan: [PLAN.md](../PLAN.md), sections 2, 4 and 8.

## Part 1: Standard TM (reference)

**Result:** the standard TM reaches **99.94% ± 0.07** mean test accuracy over
100 runs, at or above Granmo's published 99.3% ± 0.3. 98/100 runs reach at
least 99%. The four XOR clauses are learned in almost every run, but most
runs also keep a few extra clauses that use noise bits: on average 3.9 per
run, and only 1 run in 100 has no noise bit in any clause.

Run on 2026-10-26, branch `relaxed-tm`. Raw results:
[`../results/e1_standard_tm_noisy_xor.json`](../results/e1_standard_tm_noisy_xor.json).

### Question

Do our data loading, evaluation and statistics reproduce Granmo's noisy XOR
benchmark? And what do the standard TM's learned clauses look like, so the
relaxed TM can be compared with them on both accuracy and readability?

### Setup

| Part | Setting |
| --- | --- |
| Data | Granmo's released files (`cair/TsetlinMachine`): 12 input bits, label = x1 XOR x2, x3–x12 random |
| Train / test | 5,000 / 5,000, fixed split. 40% of training labels inverted; test labels clean |
| Model | Multi-Class TM with 2 classes ("y = 0", "y = 1"), 10 clauses per class (5 positive, 5 negative): Granmo's "20 clauses" |
| Settings | T = 15, s = 3.9, no boosted true-positive feedback (as in Granmo's noisy XOR demo) |
| Tsetlin Automata | 8 state bits = 256 states (tmu default); Granmo (2018) used 200 (100 per action) |
| Epochs | 200; test accuracy of the model after the final epoch is reported |
| Runs | 100 (seeds 0–99) |
| Prediction | Class with the larger vote sum; empty clauses do not vote |
| Library | `tmu`, pinned commit `acff83c` of `cair/tmu` |

Test accuracy was also logged every 10 epochs for a learning curve. It is not
used to choose a model.

**How the clauses were read out.** The TA include/exclude actions of each
class were converted into our own `HardMultiClassTM` (`baselines.py`), so the
standard TM's rules are printed and checked with the same code as the relaxed
TM's. In all 100 runs the converted model predicts exactly like tmu.

**Rule measures**, per class bank (`xor_rules.py`):

- *XOR clauses present:* all four XOR clauses appear with the right polarity. For the "y = 1" bank: `+ x1∧¬x2`, `+ ¬x1∧x2`, `− x1∧x2`, `− ¬x1∧¬x2`. The "y = 0" bank should learn the mirror image.
- *Extra clauses:* clauses that are not one of those four (duplicates of an XOR clause do not count as extra).
- *Exact rule:* XOR clauses present and no extra clauses.

### Results

| Measure | Standard TM (ours) | Granmo (2018) |
| --- | --- | --- |
| Test accuracy, mean ± 95% CI | **99.94% ± 0.07** | 99.3% ± 0.3 (Table 15) |
| 5th / 95th percentile | 100.00 / 100.00 | — |
| Min / max | 97.02 / 100.00 | — |
| Runs ≥ 99% | 98 / 100 | — |
| Training accuracy, mean | 60.30% | — |
| Training time per run | 67 s (CPU, 1 core) | — |

Training accuracy is about 60% in every run. That is expected: with 40% of
training labels inverted, a model that has learned XOR exactly agrees with
only the 60% of labels that were not inverted.

Mean test accuracy over epochs (100 runs):

| Epoch | 10 | 20 | 30 | 50 | 100 | 150 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test accuracy | 95.80% | 98.36% | 99.49% | 99.86% | 99.86% | 99.99% | 99.94% |

**Learned clauses:**

| Measure | Result |
| --- | --- |
| XOR clauses present in the "y = 1" bank | 92 / 100 runs |
| XOR clauses present in at least one of the two banks | 99 / 100 runs |
| XOR clauses present in both banks | 86 / 100 runs |
| Exact rule in the "y = 1" bank | 7 / 100 runs |
| Extra clauses in the "y = 1" bank, mean | 2.0 (of 10) |
| Clauses using a noise bit (both banks), mean per run | 3.9 (of 20) |
| Runs where no clause uses a noise bit | 1 / 100 |
| Empty clauses / contradictory clauses (all runs) | 0 / 1 |
| Mean clause length | 2.07 literals |

What the 2,000 learned clauses (100 runs × 20) are:

| Kind | Share | Example |
| --- | --- | --- |
| An XOR clause | 79.6% | `+ : x1 ∧ ¬x2` |
| An XOR clause with a noise bit added | 7.4% | `+ : ¬x1 ∧ x2 ∧ x8` |
| One of x1, x2 plus noise bits | 7.3% | `− : ¬x1 ∧ ¬x11` |
| Noise bits only | 5.7% | `+ : ¬x9` |

Example, seed 0 ("y = 1" bank, 100% test accuracy):

```
+ : x1 ∧ ¬x2
+ : ¬x1 ∧ x2
...
− : x1 ∧ x2
− : ¬x1 ∧ ¬x2
```

All rules for every run are in the JSON file (`rule_y1.rules`, `rule_y0.rules`).

### Comments

**The reproduction works.** Our pipeline (data files, final-epoch
evaluation, statistics) gives a standard-TM result consistent with Granmo's,
so it can be used unchanged for the relaxed TM.

**Our standard TM scores higher than the published 99.3%.** We use tmu, a
later implementation, not Granmo's 2018 code. The differences we know of are
256 instead of 200 TA states and a different random-number generator; we
have not tested which, if any, explains the gap. For the relaxed TM, the
comparison that matters is with **our** standard-TM runs (same data, same
seeds, same statistics), so the bar is 99.94%, not 99.3%.

**The standard TM learns XOR but does not cleanly exclude the noise bits.**
About 80% of clauses are exactly XOR clauses; the rest carry noise bits.
These extra clauses do not hurt accuracy: they are few, they often duplicate
an XOR clause with one more condition, and the XOR clauses outvote them.
But only 1 run in 100 has no noise bit anywhere, and only 7 in 100 learn the
exact rule in the "y = 1" bank. Plan section 9 asked for "the XOR rule with
x3–x12 excluded in most runs"; the standard TM itself does not meet that.
This is an opportunity: the relaxed TM's L1 penalty pushes unneeded literals
out, so it may give cleaner rules than the standard TM at the same accuracy.

**With two classes, one bank can cover for the other.** The prediction
compares the two banks' vote sums. In 7 runs the "y = 1" bank misses an XOR
clause, but the "y = 0" bank has the mirrored clause, and the test accuracy
is still 100% (e.g. seed 32: "y = 1" lacks `− ¬x1∧¬x2`, but "y = 0" has
`+ ¬x1∧¬x2` twice). So "learned the rule" should be judged on both banks
together, not on one.

**Decision needed for part 2.** Granmo's "20 clauses" are a 2-class
Multi-Class TM (2 × 10), while the plan's E1 uses a single-output relaxed TM
with 20 clauses. Both have 20 clauses, but the structure differs. Options:

- run the relaxed TM as a 2-class `MultiClassRelaxedTM` with 10 clauses per class: the same structure as the reference, so the cleanest comparison;
- run it as a single-output `RelaxedTM` with 20 clauses, as planned, and note the difference;
- run both.

### Reproduce

```bash
pip install -r requirements.txt     # installs tmu from the pinned commit
python -m experiments.relaxed_tm.classification.run_e1_baseline   # ~1 h on 2 cores
```

Seeds fix all randomness: seed k is passed to tmu as k + 1, because tmu's
random generator never returns when seeded with 0.
