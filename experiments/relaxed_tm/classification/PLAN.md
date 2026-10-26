# Experiment Plan: Relaxed TM on Noisy XOR and Iris

Oct 18, 2026 · Amr Kandil

> Copied from the Claude Doc of the same name (revision 20). Progress on each
> experiment is tracked in [README.md](README.md); results are in [reports/](reports/).

This experiment checks one thing: does a relaxed (differentiable) Tsetlin Machine, trained by gradient descent and then rounded back to hard logic, learn small classification tasks as well as a standard TM, and give readable clauses?

## 1. Goal and scope

**In scope**

- A single-output relaxed TM: soft AND clauses in which every literal has a learned include-probability, trained end to end by gradient descent (backpropagation).
- Soft forward pass: the model is trained in its smooth form and rounded (hardened) to 0/1 only at the end. If rounding loses too much accuracy, switch to gradient grafting (section 4).
- Two tasks: noisy XOR and 3-class Binary Iris, both following Granmo's (2018) benchmark.
- One output bit per clause bank, with fixed positive and negative clause teams, as in a basic TM. For 3-class Iris: one bank per class plus an argmax, as in Granmo's Multi-Class TM. No clauses are shared between banks, so this is not a Coalesced TM.

**Out of scope for now**

- Cellular automata, recurrence, multiple output bits (Coalesced TM).
- Tsetlin feedback, probabilistic TMs, hybrid learning rules.

**What a successful result looks like:** under the same benchmark protocol as Granmo (2018), the hardened model's test accuracy matches the standard TM's published results; the soft and hard accuracies are close; and on noisy XOR the learned clauses are exactly the XOR rule, with the irrelevant bits excluded.

## 2. Tasks and data

Both tasks give one output bit per example, so one single-output relaxed TM is enough.

### Task 1: Noisy XOR

The standard smallest test for a Tsetlin Machine (Granmo 2018). Each example has 12 bits: the first two decide the label by XOR, the other 10 are random noise. On the training set, a fraction of labels is flipped; the test set is clean.

```text
input  12 bits: x1 x2 | x3 … x12 (random)    output: y = x1 XOR x2

[0 1 | 1 0 0 1 1 0 1 0 0 1]  →  1
[1 1 | 0 0 1 0 1 1 0 0 1 0]  →  0

literals: 12 bits + 12 negations = 24
```

- Data, as in Granmo (2018): 10,000 examples, split 50/50 into 5,000 training and 5,000 test examples. 40% of the training labels are inverted; test labels are clean. Use the released data files in the cair/TsetlinMachine repository so results are directly comparable.
- Reference result (Granmo 2018, Table 15): the standard TM reaches 99.3% ± 0.3 mean test accuracy over 100 runs; a multilayer perceptron 95.4%; logistic regression and naive Bayes stay at chance (49.8%).
- What it tests: can the model find the two relevant bits among irrelevant ones, through heavy label noise? XOR is also **not linearly separable**: a problem is linearly separable if one straight line (or, with more inputs, one flat boundary) can put all the 1s on one side and all the 0s on the other. For XOR, the 1s sit at (0,1) and (1,0), on opposite corners of a square, and no single line separates them from (0,0) and (1,1). A model needs at least two clauses per side to solve it.
- Known correct rule: positive clauses x1 ∧ ¬x2 and ¬x1 ∧ x2; negative clauses x1 ∧ x2 and ¬x1 ∧ ¬x2; bits x3–x12 never included.

### Task 2: Iris (3 classes)

Granmo (2018) uses the full 3-class Iris problem: 150 flowers, 4 numeric measurements each (sepal length and width, petal length and width), 3 species with 50 flowers each. This is the main Iris task, following that benchmark.

- **Inputs: thermometer encoding, 16 bits (32 literals).** For each of the 4 measurements, pick k = 4 thresholds at the 20/40/60/80% quantiles of the training data. Each threshold gives one bit: "is the value above this threshold?". 4 measurements × 4 thresholds = 16 bits, the same count as Granmo's encoding. Thresholds are computed on the training split only, to avoid leaking test information.
- **Output:** 3 classes, handled as in Granmo's Multi-Class TM: one single-output relaxed TM (clause bank) per class, and the predicted class is the one with the highest vote sum (argmax). These are three independent single-output models side by side; no clauses are shared.
- **Split:** 80% training (120 flowers), 20% test (30 flowers), redrawn randomly for every run.
- **Reference result (Granmo 2018, Table 6):** standard TM 95.0% ± 0.2 mean test accuracy over 1,000 runs; multilayer perceptron 93.8%; SVM 93.6%; logistic regression 92.6%; naive Bayes 91.6%. These numbers use Granmo's encoding (below), so our own standard-TM runs on the thermometer-encoded data are the direct comparison.

```text
petal length 4.7, thresholds [1.6, 4.0, 4.6, 5.3]
  → [>1.6: 1, >4.0: 1, >4.6: 1, >5.3: 0]
```

**What Granmo did, and why we use thermometer encoding instead.** The paper only says "four bits per float". The released Binary Iris data file shows that each measurement was binned into a small whole number, and that number written as a 4-bit binary number, most significant bit first: a setosa row reads `0011 0010 0000 0000`, i.e. bins 3, 2, 0, 0. The bins are coarse (2–4 distinct values per measurement) and their boundaries are not documented.

We prefer thermometer encoding for interpretability. Each thermometer bit has a direct meaning ("petal length > 4.6"), so a learned clause reads as a plain rule, e.g. "petal length > 4.6 AND petal width ≤ 1.6 → versicolor". In Granmo's binary-number code a single bit has no meaning on its own: "bin = 2" needs a combination like "bit 3 = 0 AND bit 4 = 1", and the bin edges are unknown, so the clauses cannot be translated back into measurements. Granmo's released file is kept as an encoding ablation (E5), to check that the encoding choice does not cost accuracy.

```text
input  16 bits (4 per measurement) → 32 literals
  → clause bank "setosa"      → vote sum s1
  → clause bank "versicolor"  → vote sum s2
  → clause bank "virginica"   → vote sum s3
  → predicted class = argmax(s1, s2, s3)
```

## 3. Model: single-output relaxed TM

```text
input bits x (n)
  │  Step 1: literals  l = [x, 1 − x]                                   (fixed)
  ▼
literals l (2n)
  │  Step 2–3: soft clauses  c_j = ∏_k (1 − p_jk (1 − l_k)),  p = sigmoid(w)
  │            m clauses: first half positive (+), second half negative (−)
  │            ← the only learned stage: weights w (m × 2n)
  ▼
clause outputs c (m)
  │  Step 4: vote  s = Σ c⁺ − Σ c⁻                                       (fixed)
  ▼
vote sum s
  │  Step 5: output  ŷ = sigmoid((s − b) / τ)                            (fixed; b optional)
  ▼
ŷ  →  loss (binary cross-entropy); gradients flow back to w
```

The only learned numbers are one weight w per (clause, literal) pair, plus an optional bias b; every other step is a fixed formula.

**Step 1: literals.** From n input bits x, build 2n literals: l = [x1 … xn, 1 − x1 … 1 − xn]. Noisy XOR: 24 literals. Iris: 32.

**Step 2: include-probabilities.** Each clause j and literal k has a free real number w_jk, which gradient descent can move freely. Its include-probability is p_jk = sigmoid(w_jk), always between 0 and 1. Large positive w means "include", large negative w means "exclude", w = 0 means undecided (p = 0.5). In a standard TM this decision is made by a Tsetlin Automaton; here the TA is replaced by w.

**Step 3: soft clauses.** A hard clause is an AND: an excluded literal never blocks it (contributes 1), an included literal must be 1 (contributes its value l). With an include-probability p in between, each literal contributes a blend of the two cases, (1 − p)·1 + p·l = 1 − p·(1 − l). It pulls the clause down only when it is 0 and (partly) included. The clause is the product of all contributions (Payani & Fekri 2019):

```math
c_j = \prod_k \bigl(1 - p_{jk}(1 - l_k)\bigr)
```

Multiplying many numbers below 1 quickly underflows to 0 on a computer, so the clause is computed in log space, where the product becomes a sum: log c_j = Σ_k log(1 − p_jk(1 − l_k)). Two facts simplify each term:

1. The literals are exactly 0 or 1. If l_k = 1, the term is log(1) = 0, so it drops out. If l_k = 0, the term is log(1 − p_jk). Multiplying by (1 − l_k) does exactly this switching: it is 1 when the literal is 0, and 0 when the literal is 1.
2. 1 − sigmoid(w) = sigmoid(−w), so log(1 − p) = log sigmoid(−w), a standard, numerically stable function ("log-sigmoid").

Putting both together:

```math
\log c_j = \sum_k (1 - l_k)\,\log\sigma(-w_{jk})
```

For a batch of examples this is a single matrix product: the matrix of (1 − literal) values, examples × literals, times the matrix of log sigmoid(−w), literals × clauses. Taking exp gives every clause output for every example at once.

**Step 4: vote.** Half the clauses are positive, half negative (fixed teams, as in a basic TM). The vote sum is s = (sum of positive clause outputs) − (sum of negative clause outputs).

**Step 5: output.** ŷ = sigmoid((s − b) / τ). A basic TM outputs 1 if s ≥ 0, a hard step with no slope, so gradients cannot pass through it; the sigmoid is a smooth version of that step. Dividing by τ sets its steepness: small τ is close to the hard step but gives tiny gradients far from the threshold, large τ gives useful gradients over a wider range of vote sums. b is a learned threshold, or fixed at 0 to match a basic TM exactly.

τ is not the standard TM's T. In a standard TM, T is a target vote margin: once the vote sum reaches ±T, clauses stop receiving feedback. τ only shapes the smooth output. Starting with τ close to the benchmark's T keeps the two comparable: a vote sum of T gives an output of about 0.73, and the gradient keeps shrinking as the sum grows past it.

**Multi-class (Iris).** One clause bank per class gives one vote sum per class: s1, s2, s3. Training turns them into class probabilities with a softmax over s_c / τ and uses cross-entropy loss. Prediction is the class with the largest hard vote sum.

**Parameters:** m clauses × 2n literals, plus b. Noisy XOR with 10 clauses: 240 weights. Iris with 20 clauses: 640.

### Hardening (after training)

1. Include literal k in clause j if p_jk > 0.5, i.e. w_jk > 0.
2. Each hard clause is the AND of its included literals; the hard vote sum s_hard uses these 0/1 clause outputs.
3. Hard prediction: 1 if s_hard ≥ b, else 0.

Two edge cases to handle explicitly:

- **Empty clauses** (no literal included). Mathematically, an AND of nothing is 1, so an empty clause would vote on every input. Granmo's TM code treats an empty clause as 1 during training, so it keeps receiving feedback and can learn, but as 0 at prediction, so it does not vote. We follow the same rule for the hardened model: empty clauses output 0 at prediction. We also count them in every run, because many empty clauses mean wasted capacity (too many clauses, or too strong an L1 penalty).
- **Contradictory clauses** (both x and ¬x included) always output 0. Harmless but wasted; count them.

## 4. Training setup

Data, splits, epochs, repeats and reporting follow Granmo (2018) exactly, so the relaxed TM can be compared with the standard TM's published numbers; only the learning method differs.

### Benchmark protocol (Granmo 2018)

|  | Noisy XOR | Binary Iris |
| --- | --- | --- |
| Data | 10,000 examples, 12 input bits | 150 flowers, 16 input bits, 3 classes |
| Train / test split | 50 / 50 (5,000 / 5,000), fixed | 80 / 20 (120 / 30), redrawn every run |
| Label noise | 40% of training labels inverted; test labels clean | None |
| Epochs | 200 | 500 |
| Reported accuracy | Test accuracy after the final epoch | Test accuracy after the final epoch |
| Repeated runs | 100 | 1,000 |
| Reported statistics | Mean ± 95% confidence interval, 5th and 95th percentile, minimum, maximum | Same |
| Standard TM settings | 20 clauses, s = 3.9, T = 15, 100 states per TA | 300 clauses, s = 3.0, T = 10, 100 states per TA, Multi-Class TM, boosted true-positive feedback |

Two rules follow from this protocol:

- **No picking the best epoch.** Granmo reports the final epoch rather than the best one, to avoid tuning on the test set. We do the same. Any hyperparameter tuning uses a validation split carved out of the training data, never the test set.
- **Report training accuracy too.** Granmo reports it alongside test accuracy to show overfitting; we do the same.

### Relaxed TM training

Training is fully soft: the loss and gradients use the smooth model, and the hard (rounded) model is only evaluated, never trained directly.

- **Loss:** binary cross-entropy for noisy XOR; softmax cross-entropy over the 3 class vote sums for Iris.
- **Optimizer:** Adam, learning rate around 0.01–0.05; mini-batches of 128 (XOR) and 32 (Iris).
- **Clause count:** the same as the standard TM in the main comparison: 20 for noisy XOR; for Iris, 300 in total, i.e. 100 per class (Granmo's code splits the 300 clauses evenly over the 3 classes).
- **Initialization:** w drawn from a normal distribution with negative mean (e.g. mean −1.5, std 1). Most literals start mostly excluded (p ≈ 0.2), which keeps clauses short at the start, as DiffLogic CA biases its gates toward pass-through. The spread breaks symmetry between clauses.
- **Regularization:**
  - *L1 on include-probabilities:* add λ₁ · Σ p to the loss. Pushes unneeded literals out, giving shorter, more readable clauses.
  - *Entropy penalty:* add λ₂ · Σ H(p), where H(p) is largest at p = 0.5. Pushes each p toward 0 or 1, so rounding changes little. Start at 0 and switch on late in training.
- **Monitoring:** every epoch, log soft and hard accuracy on the training and validation data, to see how the soft–hard gap develops. This is for diagnosis only, not to choose which model is reported.

### Fallback: gradient grafting (Wang et al. 2021)

Used only if the soft–hard gap is too large (section 9). Each training step runs two versions of the same model: the soft one, and a hard one built by rounding the current p's. The loss is computed on the hard model's output, so training sees the real discrete model's error. That error's gradient is then passed back through the soft model, evaluated at the actual p's, so every literal still gets a non-zero signal. The hard model changes only when some p crosses 0.5.

## 5. Hyperparameters

Benchmark values come from Granmo (2018); the other starting values are guesses to be tuned on validation data. Swept settings belong to the experiments in section 8.

| Hyperparameter | Meaning | Noisy XOR | Iris | Swept? |
| --- | --- | --- | --- | --- |
| m | Number of clauses (half positive, half negative) | 20 (benchmark); also 4, 10 | 300 in total, 100 per class (benchmark); also 30, 60, 150 in total | Yes |
| τ | Output smoothness in sigmoid((s − b)/τ) | Start near the benchmark T = 15; also 1, 5 | Start near the benchmark T = 10; also 1, 5 | Yes |
| b | Threshold | 0 (fixed) or learned | Learned, one per class | Ablation |
| Init mean of w | Starting include-probability | −1.5 | −1.5 | Yes (−3, −1.5, 0) |
| Init std of w | Spread between clauses | 1 | 1 | No |
| Learning rate | Adam step size | 0.01–0.05 | 0.01–0.05 | Light tuning |
| Batch size | Examples per update | 128 | 32 | No |
| Epochs | Passes over training data | 200 (benchmark) | 500 (benchmark) | No; final epoch reported |
| λ₁ | L1 penalty on p | 0, 1e-4, 1e-3 | 0, 1e-4, 1e-3 | Yes |
| λ₂ | Entropy penalty on p (late in training) | 0, 1e-3 | 0, 1e-3 | Yes |
| Input encoding | How inputs become bits | Released noisy XOR data | Thermometer, k = 4 (16 bits); Granmo's Binary Iris file as ablation | Ablation |
| Label noise | Fraction of inverted training labels | 40% (benchmark); also 0, 10, 20% | n/a | Yes |
| Runs | Repeats per configuration | 100 (benchmark) | 1,000 (benchmark); 100 while developing | No |

## 6. Baselines

The main comparison is the standard TM under the identical protocol; the others place the result in context.

- **Standard TM, own runs:** an existing library such as `tmu` or `pyTsetlinMachine`, at the benchmark settings (section 4), on the same splits and seeds as the relaxed TM.
- **Published numbers (Granmo 2018), as a check that our own runs reproduce the benchmark:**

| Method | Noisy XOR test accuracy | Binary Iris test accuracy |
| --- | --- | --- |
| Standard TM | 99.3% ± 0.3 | 95.0% ± 0.2 |
| Multilayer perceptron | 95.4% ± 0.5 | 93.8% ± 0.2 |
| SVM | 58.0% ± 0.3 | 93.6% ± 0.3 |
| Logistic regression | 49.8% ± 0.3 | 92.6% ± 0.2 |
| Naive Bayes | 49.8% ± 0.2 | 91.6% ± 0.3 |

- **dNL-style soft-AND model** (Payani & Fekri 2019): the same soft clauses, combined by a soft OR instead of a vote. Isolates what the vote adds. Optional.
- **Logistic regression, own runs:** a simple non-logic reference on the same bits, to confirm it fails on XOR (not linearly separable).

## 7. Metrics

The hard accuracy on the test set is the headline number; the rest explain why it is or isn't good.

**Correctness**

- Hard test accuracy (after rounding): the main result.
- Soft test accuracy, and the **soft–hard gap** = soft minus hard accuracy. A large gap means rounding breaks the model.
- Success rate across seeds: e.g. on noisy XOR, the fraction of seeds reaching ≥ 99% hard test accuracy.

**Interpretability**

- Number of non-empty clauses, total included literals, average clause length.
- Empty and contradictory clause counts.
- On noisy XOR: does the learned rule equal the known XOR rule, and are x3–x12 never included? (Yes/no per seed.)
- The hardened rule printed as text, e.g. `+ : x1 ∧ ¬x2`, kept for every run.

**Training behavior**

- Epochs until the hard validation accuracy stops improving.
- Distribution of p at the end: the share of literals still undecided (0.1 < p < 0.9). Many undecided literals predict a large soft–hard gap.
- Wall-clock training time.

**Efficiency (for the sustainability question, recorded but not optimized yet)**

- Hardened model size in bits (one include bit per literal per clause).
- Logic operations per prediction.
- Training energy, measured with a tool such as CodeCarbon, for both the relaxed and the standard TM.

## 8. Experiments

Run in this order; each one only starts once the previous works.

| # | Experiment | Task | What varies | Question it answers |
| --- | --- | --- | --- | --- |
| E0 | Sanity check | Clean XOR, 2 input bits, no noise | Nothing | Does the code learn XOR at all? Expect 100% in a few epochs |
| E1 | Main result: noisy XOR | Noisy XOR, benchmark protocol (20 clauses, 200 epochs, 100 runs) | Nothing beyond the protocol | Does the hardened model match the standard TM's 99.3% and recover the XOR rule? |
| E2 | Clause count | Noisy XOR | m = 4, 10, 20 | How few clauses are enough? Do more clauses help or add junk? |
| E3 | Noise robustness | Noisy XOR | Noise 0, 10, 20, 40% | How does accuracy degrade with label noise, compared with the standard TM? |
| E4 | Main result: Iris | Binary Iris, 3 classes, benchmark protocol (100 clauses per class, 500 epochs, 1,000 runs) | Nothing beyond the protocol | Does it match the standard TM's 95.0%? |
| E5 | Ablations | Noisy XOR, Binary Iris | τ, init mean, λ₁, λ₂, b fixed vs. learned, Iris encoding | Which settings matter for accuracy and for the soft–hard gap? |
| E6 | Baselines | All tasks | Model type | Standard TM, dNL-style, logistic regression; same splits and seeds |

## 9. Success criteria and decision rules

**The relaxed TM counts as working if:**

- Noisy XOR (benchmark protocol): mean hard test accuracy over 100 runs close to the standard TM's 99.3% (at least 98%), and the hardened rule is the XOR rule with x3–x12 excluded in most runs.
- Binary Iris (benchmark protocol): mean hard test accuracy over 1,000 runs close to the standard TM's 95.0% (at least 94%).
- Soft–hard gap ≤ 1 percentage point on average.

**What to do if not:**

| Symptom | Likely cause | Next step |
| --- | --- | --- |
| Soft accuracy good, hard accuracy much lower | Many undecided p near 0.5 | Turn on the entropy penalty λ₂, or lower τ. If the gap stays, switch to gradient grafting (section 4) |
| Neither soft nor hard accuracy improves | Clauses dead (all near 0) or saturated vote | Change init mean of w, adjust τ, check gradients are non-zero |
| Accurate but rule unreadable (long clauses, irrelevant bits) | Too little sparsity pressure | Raise λ₁ |
| Many empty or duplicate clauses | Too many clauses for the task | Reduce m; report it as a result |
| Fails only on noisy XOR | Noise overwhelms the gradient | Larger batches, lower learning rate; compare with the standard TM's noise curve (E3) |

## 10. Implementation checklist

PyTorch is enough; the whole model is a few matrix products. Suggested place: a new module in the `cakit` repository, so it can later be dropped into the CA.

- [ ] Data: load the released noisy XOR files from cair/TsetlinMachine; Iris from the original measurements with a thermometer encoder fitted on each training split; 80/20 split redrawn per run; Granmo's Binary Iris file for the encoding ablation; optional noisy XOR generator (noise level as a parameter).
- [ ] Model: `RelaxedTM(n_inputs, n_clauses, tau, learn_bias)` with literals, log-space soft clauses, vote and sigmoid output; `MultiClassRelaxedTM` holding one bank per class with softmax training and argmax prediction.
- [ ] Hardening: `harden()` returning include masks, and a hard forward pass that uses them; explicit empty-clause convention.
- [ ] Rule printer: hardened clauses as text, e.g. `+ : x1 ∧ ¬x2`.
- [ ] Training loop: BCE or softmax cross-entropy, Adam, L1 and entropy penalties, fixed epoch count (200 / 500), final-epoch evaluation; soft and hard accuracy logged per epoch for diagnosis.
- [ ] Logging and reporting: config, seed and all metrics from section 7 per run; summary per configuration as mean ± 95% confidence interval, 5th and 95th percentile, minimum and maximum, for both test and training accuracy.
- [ ] Unit tests: a hand-set model that implements XOR exactly must give 100%; a clause with all p = 0 outputs 1; a clause including x and ¬x outputs 0.
- [ ] Baselines: standard TM via `tmu`; logistic regression via scikit-learn.
- [ ] Energy: wrap training runs with CodeCarbon.
- [ ] Run E0 → E6 in order; summarize in a results table per experiment.

## Sources

- [Granmo — The Tsetlin Machine (arXiv:1804.01508), noisy XOR benchmark](https://arxiv.org/pdf/1804.01508)
- [Payani & Fekri — Learning Algorithms via Neural Logic Networks (arXiv:1904.01554), soft AND clause](https://arxiv.org/pdf/1904.01554)
- [Wang et al. — RRL, Gradient Grafting (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/ffbd6cbb019a1413183c8d08f2929307-Abstract.html)
- [Miotti et al. — Differentiable Logic CA (soft and hard loss tracking, gate initialization)](https://google-research.github.io/self-organising-systems/difflogic-ca/)
- [CAIR — Tsetlin Machine code and datasets](https://github.com/cair/TsetlinMachine)
