# Relaxed (differentiable) Tsetlin Machine experiments

A relaxed Tsetlin Machine replaces each Tsetlin Automaton's include/exclude
decision with a learned include-probability, so the whole machine can be
trained by gradient descent and then hardened (rounded) back to 0/1 logic.
The work goes in stages: first plain classification, then the relaxed TM as
the update rule of a cellular automaton (a DiffLogic CA analogue).

The model itself lives in [`cakit/tm/`](../../cakit/tm/); see its
[README](../../cakit/tm/README.md) for how the code maps to the model.

## Stages

| Stage | Folder | Status |
| --- | --- | --- |
| Tier 0: classification without CA (noisy XOR, Iris) | [`classification/`](classification/) | In progress: E0 done |
| Later: relaxed TM as a CA update rule | to be added as sibling folders | Not started |

Each stage folder has the same shape:

```text
<stage>/
  README.md      status of each experiment, how to run it
  PLAN.md        the experiment plan
  data.py        datasets for this stage
  run_eN.py      one script per experiment
  reports/       one write-up per experiment (setup, results, comments)
  results/       raw results as JSON (committed)
  data/          downloaded datasets (git-ignored)
```

Code shared by all stages is in [`common/`](common/):

| File | Contents |
| --- | --- |
| `train.py` | `TrainConfig`, `train()` (Adam, BCE or softmax cross-entropy, L1 and entropy penalties, per-epoch soft/hard accuracy), `evaluate()` (soft, hard, gap), `summarize()` (Granmo's statistics: mean ± 95% CI, 5th/95th percentile, min, max) |

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt   # adds torch and scikit-learn
pip install -e .
pytest tests/tm                    # unit tests for cakit.tm
```

Run experiments from the repository root as modules, e.g.
`python -m experiments.relaxed_tm.classification.run_e0`.
