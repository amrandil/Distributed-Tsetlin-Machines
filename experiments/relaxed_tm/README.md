# Relaxed (differentiable) Tsetlin Machine experiments

Verifies that a relaxed Tsetlin Machine, trained by gradient descent and then
hardened back to 0/1 logic, learns small classification tasks as well as a
standard Tsetlin Machine and gives readable clauses. No cellular automata yet.

The model lives in `cakit/tm/`:

| File | Contents |
| --- | --- |
| `relaxed.py` | `RelaxedTM` (soft AND clauses in log space, signed vote, `sigmoid((s - b) / tau)`) and `MultiClassRelaxedTM` (one bank per class, softmax training, argmax prediction) |
| `harden.py` | `harden()`: include iff p > 0.5; empty clauses output 0 at prediction (Granmo's convention); clause statistics |
| `rules.py` | `format_rules()`: hard clauses as text, e.g. `+ : x1 ∧ ¬x2` |
| `encoding.py` | `ThermometerEncoder`: continuous features to bits, fitted on the training split |

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt   # adds torch and scikit-learn
pip install -e .
pytest tests/tm                    # unit tests for cakit.tm
```

## Experiments

| # | Script | Status |
| --- | --- | --- |
| E0 | `python -m experiments.relaxed_tm.run_e0` | Clean 2-bit XOR, 10 seeds |
| E1-E6 | to come | Noisy XOR and Iris under Granmo's benchmark protocol |

`data.py` downloads Granmo's noisy XOR and Binary Iris files from
`cair/TsetlinMachine` on first use and caches them in `data/` (git-ignored).
Results are written to `results/` as JSON.

### E0 notes

With 10 or 20 clauses all 10 seeds reach 100% hard accuracy and the exact
XOR rule. With the minimal 4 clauses about 3 in 10 seeds get stuck: two
same-polarity clauses learn the same sub-pattern and one XOR case is left
uncovered (vote sum 0). A standard TM avoids this with its target T, which
stops feedback to a sub-pattern once it has enough votes; plain cross-entropy
keeps rewarding the covered pattern. Try `--clauses 4` to reproduce.
