# `cakit.tm` — relaxed Tsetlin Machine

A Tsetlin Machine whose include/exclude decisions are learned
include-probabilities, so it trains by gradient descent (PyTorch). After
training it is hardened to an ordinary 0/1 Tsetlin Machine. The model is
explained step by step in
[the Tier 0 plan, section 3](../../experiments/relaxed_tm/classification/PLAN.md#3-model-single-output-relaxed-tm).

## Forward pass and where it is in the code

```text
input bits x (n)
  │  literals  l = [x, 1 − x]                          make_literals()
  ▼
literals l (2n)
  │  soft clauses  c_j = ∏_k (1 − p_jk (1 − l_k))       RelaxedTM.clause_outputs()
  │  p = sigmoid(w), computed in log space              RelaxedTM.include_probs()
  │  learned: w, shape (clauses, 2n)
  ▼
clause outputs c
  │  vote  s = Σ c⁺ − Σ c⁻   (first half +, second half −)   RelaxedTM.vote_sum()
  ▼
vote sum s
  │  ŷ = sigmoid((s − b) / τ)                           RelaxedTM.forward()
  ▼
ŷ
```

Multi-class: `MultiClassRelaxedTM` holds one `RelaxedTM` bank per class;
softmax over `(s_c − b_c) / τ` for training, argmax for prediction.

## Files

| File | Contents |
| --- | --- |
| `relaxed.py` | `make_literals`, `RelaxedTM`, `MultiClassRelaxedTM`; L1 and entropy penalties on p |
| `harden.py` | `harden()`: include a literal iff p > 0.5. `HardTM` / `HardMultiClassTM` with a hard forward pass; empty clauses output 0 at prediction (Granmo's convention); `stats()` counts empty and contradictory clauses; `undecided_fraction()` |
| `rules.py` | `format_rules()`: hard clauses as text, e.g. `+ : x1 ∧ ¬x2` |
| `encoding.py` | `ThermometerEncoder`: continuous features to bits, thresholds at training-split quantiles |

Tests: `tests/tm/`. Experiments using this package: `experiments/relaxed_tm/`.
