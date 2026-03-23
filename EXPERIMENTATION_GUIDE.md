# Experimentation Guide

This guide explains how to run different experiments with the `cakit` framework.

## Quick Start

### Method 1: Use the Experimentation Script (Easiest)

The `examples/experiments.py` script is designed for quick experimentation:

1. **Edit the configuration section** at the top of the file
2. **Run the script**: `python examples/experiments.py`

```python
# Choose experiment type
EXPERIMENT_TYPE = 'wolfram_ca'  # or 'tsetlin_cla'

# For Wolfram CA experiments
WOLFRAM_RULE = 30              # Try: 30, 110, 90, 184
WOLFRAM_ITERATIONS = 100       # More iterations = longer simulation
WOLFRAM_GRID_SIZE = 201        # Larger grid = more spatial detail
WOLFRAM_INITIAL = 'single_center'  # or 'random', 'alternating'

# For Tsetlin CLA experiments
TSETLIN_N_STATES = 4           # Fewer states = faster learning, less memory
TSETLIN_ITERATIONS = 150       # More iterations = more learning time
TSETLIN_GRID_SIZE = 51
TSETLIN_INITIAL = 'random'     # or 'single_center', 'alternating', etc.
```

### Method 2: Modify Existing Demos

Edit `examples/demo_rule_110.py` or `examples/demo_cla.py` directly:

**For Rule 110 with more iterations:**
```python
# In demo_rule_110.py, line 46
grid_history, automaton_history = system.run(generations=200)  # was 50
```

**For Rule 30 instead of Rule 110:**
```python
# In demo_rule_110.py, line 38
automaton_params={'rule': 30},  # was 110
```

**For fewer Tsetlin states:**
```python
# In demo_cla.py, line 44
automaton_params={'n_states': 4, 'initial_state': 'random'},  # was 10
```

### Method 3: Write Your Own Script

Copy one of the example scripts and customize it:

```bash
cp examples/demo_rule_110.py my_experiment.py
# Edit my_experiment.py
python my_experiment.py
```

---

## Detailed Parameter Reference

### Wolfram CA Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `rule` | int | 0-255 | Wolfram rule number (try 30, 90, 110, 184) |
| `generations` | int | Any | Number of time steps to simulate |
| `grid.size` | int | Any odd | Number of cells (odd numbers work better for single_center) |
| `grid.radius` | int | 1 | Always 1 for elementary CA |
| `grid.boundary` | str | `'periodic'`, `'zero'`, `'fixed'` | How edges behave |
| `initial_states` | str/list | See below | Starting configuration |

**Popular Wolfram Rules:**
- **Rule 30**: Chaotic, random-looking patterns (used in Mathematica's random number generator)
- **Rule 90**: Sierpiński triangle patterns
- **Rule 110**: Complex, Turing-complete behavior
- **Rule 184**: Traffic flow model
- **Rule 150**: XOR operation patterns

### Tsetlin CLA Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `n_states` | int | 2+ | States per arm (total states = `n_states * 2`) |
| `generations` | int | Any | Number of time steps to simulate |
| `grid.size` | int | Any | Number of cells |
| `feedback_radius` | int | 1+ | Neighborhood size for feedback |
| `initial_state` | str | `'random'`, `'center'`, `'random_arm'` | TA initial state distribution |

**Effect of n_states:**
- **Fewer states (2-4)**: Faster learning, more volatile, less "memory"
- **More states (10-20)**: Slower learning, more stable, better "memory"

### Initial State Options

| Option | Description | Best For |
|--------|-------------|----------|
| `'single_center'` | One cell active in center | Observing pattern propagation |
| `'random'` | Random 0s and 1s | Testing synchronization/consensus |
| `'alternating'` | Alternating 0-1-0-1 pattern | Testing stability |
| `'all_zeros'` | All cells start at 0 | Observing spontaneous activation |
| `'all_ones'` | All cells start at 1 | Testing equilibrium |
| `[0,1,1,0,...]` | Custom list | Specific test patterns |

### Boundary Conditions

| Boundary | Behavior | Best For |
|----------|----------|----------|
| `'periodic'` | Wraps around (ring topology) | Avoiding edge effects |
| `'zero'` | Pads with 0s at edges | Observing edge influence |
| `'fixed'` | Uses specific fixed values | Custom boundary behavior |

---

## Example Experiments

### Experiment 1: Rule 30 for 200 Generations

```python
# In examples/experiments.py
EXPERIMENT_TYPE = 'wolfram_ca'
WOLFRAM_RULE = 30
WOLFRAM_ITERATIONS = 200
WOLFRAM_GRID_SIZE = 201
WOLFRAM_INITIAL = 'single_center'
BOUNDARY = 'zero'
```

**Run:** `python examples/experiments.py`

**Expected result:** Chaotic, pseudo-random triangular pattern expanding from center.

### Experiment 2: Rule 110 for 500 Generations

```python
# In examples/experiments.py
EXPERIMENT_TYPE = 'wolfram_ca'
WOLFRAM_RULE = 110
WOLFRAM_ITERATIONS = 500
WOLFRAM_GRID_SIZE = 301
WOLFRAM_INITIAL = 'single_center'
BOUNDARY = 'zero'
```

**Run:** `python examples/experiments.py`

**Expected result:** Complex structures with "gliders" and repeating motifs.

### Experiment 3: Low-State Tsetlin CLA (4 states/arm)

```python
# In examples/experiments.py
EXPERIMENT_TYPE = 'tsetlin_cla'
TSETLIN_N_STATES = 4
TSETLIN_ITERATIONS = 150
TSETLIN_GRID_SIZE = 51
TSETLIN_INITIAL = 'random'
TSETLIN_FEEDBACK_RADIUS = 1
BOUNDARY = 'periodic'
```

**Run:** `python examples/experiments.py`

**Expected result:** Fast but noisy convergence to synchronized regions. Check the augmented plot to see confidence levels.

### Experiment 4: Tsetlin CLA with Alternating Initial Pattern

```python
# In examples/experiments.py
EXPERIMENT_TYPE = 'tsetlin_cla'
TSETLIN_N_STATES = 6
TSETLIN_ITERATIONS = 200
TSETLIN_GRID_SIZE = 51
TSETLIN_INITIAL = 'alternating'
TSETLIN_FEEDBACK_RADIUS = 1
BOUNDARY = 'periodic'
```

**Run:** `python examples/experiments.py`

**Expected result:** Will the TAs maintain the alternating pattern or converge to homogeneity?

### Experiment 5: High-State Tsetlin CLA (20 states/arm)

```python
# In examples/experiments.py
EXPERIMENT_TYPE = 'tsetlin_cla'
TSETLIN_N_STATES = 20
TSETLIN_ITERATIONS = 300
TSETLIN_GRID_SIZE = 51
TSETLIN_INITIAL = 'random'
TSETLIN_FEEDBACK_RADIUS = 1
BOUNDARY = 'periodic'
```

**Run:** `python examples/experiments.py`

**Expected result:** Slower but very stable convergence. The augmented plot will show strong confidence (dark colors) once converged.

---

## Interpreting Visualizations

### Standard Space-Time Plots
- **Horizontal axis**: Spatial position (cell index)
- **Vertical axis**: Time (top = initial, bottom = final)
- **Black**: State 1 (active)
- **White**: State 0 (inactive)

### TA-Augmented Plots (CLA only)
- **Color intensity** encodes Tsetlin Automaton confidence:
  - **Dark red**: High confidence in action 1 (state near 1)
  - **Dark blue**: High confidence in action 0 (state near `2*n_states`)
  - **Light pink/cyan**: Low confidence (state near middle)

**What to look for:**
- **Convergence**: Patterns stabilize over time
- **Oscillations**: Repeating patterns in time
- **Spatial structures**: Clusters, stripes, domains
- **Edge effects**: How boundaries influence behavior

---

## Tips for Systematic Experimentation

1. **Start small**: Use fewer iterations and smaller grids to iterate quickly
2. **Vary one parameter at a time**: Isolate effects
3. **Use consistent seeds**: Set `SEED = 42` for reproducible results
4. **Compare initial conditions**: Same rule, different starts
5. **Save outputs**: Files are named automatically with parameters
6. **Document observations**: Keep notes on what works/doesn't work

---

## Common Patterns to Explore

### For Wolfram CA:
- **Class 1 (Uniform)**: Rules 0, 8, 32, 160
- **Class 2 (Periodic)**: Rules 4, 108, 250
- **Class 3 (Chaotic)**: Rules 30, 45, 73, 105
- **Class 4 (Complex)**: Rules 110, 124, 137, 193

### For Tsetlin CLA:
- **Synchronization**: Random start, periodic boundary
- **Pattern formation**: Alternating start, zero boundary
- **Edge influence**: Single_center start, zero boundary
- **Stability**: All_ones or all_zeros start

---

## Advanced: Using the Framework Directly

For full control, write custom Python scripts:

```python
from cakit.automata import TsetlinAutomaton
from cakit.grid import Grid1D
from cakit.system import StateSystem
from cakit.feedback import NeighbourhoodAgreementFeedback
from cakit.visualization import SpaceTimePlot

# Create custom grid
grid = Grid1D(
    size=100,
    radius=2,  # Larger neighborhoods!
    boundary='periodic',
    initial_states=[1 if i % 3 == 0 else 0 for i in range(100)]  # Custom pattern
)

# Create custom feedback
feedback = NeighbourhoodAgreementFeedback(radius=2)

# Create system
system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': 8, 'initial_state': 'center'},
    feedback_fn=feedback,
    feedback_radius=2,
    seed=123
)

# Run and visualize
grid_history, ta_history = system.run(generations=200)
plotter = SpaceTimePlot(grid_history, ta_history, n_states=8)
plotter.plot_ta_augmented(save_path='my_experiment.png')
```

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cakit'`
- **Solution**: Run `pip install -e .` from the project root

**Issue**: Plot window pops up and freezes
- **Solution**: Make sure `matplotlib.use('Agg')` is at the very top of your script

**Issue**: Simulation is too slow
- **Solution**: Reduce grid size or iterations, or use fewer Tsetlin states

**Issue**: Patterns look too noisy/chaotic
- **Solution**: For CLA, increase `n_states` or run more iterations

**Issue**: Nothing interesting happens
- **Solution**: Try different initial states or Wolfram rules from different classes

---

## Next Steps

- Try implementing custom feedback functions (`cakit/feedback/`)
- Extend to 2D grids (requires implementing `Grid2D`)
- Implement new automaton types
- Analyze convergence rates quantitatively
- Export data to CSV for external analysis

Happy experimenting! 🧪
