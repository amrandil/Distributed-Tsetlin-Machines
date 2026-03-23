# Cakit Examples

This directory contains example scripts demonstrating the cakit framework.

## Running Examples

Make sure you have installed the package first:

```bash
# From the project root
pip install -e .
```

Then run any example:

```bash
python examples/demo_rule_110.py
python examples/demo_cla.py
```

## Available Examples

### `demo_rule_110.py` - Elementary CA with Rule 110

Demonstrates a classical cellular automaton with no learning. Uses:
- `WolframAutomaton` with Rule 110 (Turing-complete, Class 4 behavior)
- 101-cell 1D grid with single center initialization
- Zero boundaries
- 50 generations

**Output**: `rule_110_spacetime.png` - Classic black/white space-time diagram showing the characteristic Rule 110 triangle pattern.

### `demo_cla.py` - Binary CLA with Tsetlin Automata

Demonstrates a learning cellular automaton where each cell learns to synchronize with its neighbors. Uses:
- `TsetlinAutomaton` with 10 states per arm
- 51-cell 1D grid with random initialization
- Periodic boundaries
- `NeighbourhoodAgreementFeedback` - rewards cells matching neighborhood majority
- 100 generations

**Outputs**:
- `cla_spacetime_standard.png` - Standard binary view showing cell states over time
- `cla_spacetime_ta_augmented.png` - Augmented view encoding both cell state (red=alive, blue=dead) and TA confidence (darker=more committed)

## Creating Your Own Experiments

The framework is designed to be easily extensible. Here's a template:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cakit.automata import TsetlinAutomaton, WolframAutomaton
from cakit.grid import Grid1D
from cakit.system import StateSystem, RuleSystem
from cakit.feedback import NeighbourhoodAgreementFeedback
from cakit.visualization import SpaceTimePlot

# 1. Configure your grid
grid = Grid1D(
    size=100,
    radius=1,
    boundary='periodic',
    initial_states='random'  # or 'single_center', 'half', or a numpy array
)

# 2. Choose your automaton
# For plain CA:
automaton_type = WolframAutomaton
automaton_params = {'rule': 30}

# For learning CLA:
# automaton_type = TsetlinAutomaton
# automaton_params = {'n_states': 5, 'initial_state': 'random'}

# 3. Optional: Create feedback for learning
# feedback_fn = NeighbourhoodAgreementFeedback(radius=1)
feedback_fn = None  # For pure CA

# 4. Create and run system
system = StateSystem(
    grid=grid,
    automaton_type=automaton_type,
    automaton_params=automaton_params,
    feedback_fn=feedback_fn,
    feedback_radius=1,
    seed=42
)

grid_history, automaton_history = system.run(generations=50)

# 5. Visualize
plotter = SpaceTimePlot(grid_history=grid_history)
plotter.plot_standard(save_path='my_experiment.png', show=False)
```

## Tips

- Use `boundary='zero'` for Rule 110 visualization (creates clean triangular patterns)
- Use `boundary='periodic'` for CLA experiments (avoids edge effects)
- For CLA experiments, 100+ generations are needed to observe learning convergence
- Use `seed` parameter for reproducible experiments
- The TA-augmented plot is only available when using learning automata (TsetlinAutomaton)
