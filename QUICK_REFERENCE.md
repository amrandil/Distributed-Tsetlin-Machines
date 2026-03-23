# Cakit Quick Reference

## Import Cheat Sheet

```python
# Automata
from cakit.automata import (
    Automaton,              # Abstract base
    LearningAutomaton,      # Abstract base for LA
    TsetlinAutomaton,       # 2-armed FSSA
    WolframAutomaton        # Fixed rule CA
)

# Grid
from cakit.grid import (
    Grid,                   # Abstract base
    Grid1D                  # 1D grid implementation
)

# Systems
from cakit.system import (
    AutomataSystem,         # Abstract base
    StateSystem,            # Output = cell state
    RuleSystem              # Output = rule selector
)

# Feedback
from cakit.feedback import (
    FeedbackFunction,                    # Abstract base
    NeighbourhoodAgreementFeedback       # Majority synchronization
)

# Visualization
from cakit.visualization import SpaceTimePlot
```

---

## Common Patterns

### Pattern 1: Pure CA (No Learning)

```python
grid = Grid1D(size=100, radius=1, boundary='periodic', initial_states='random')

system = StateSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params={'rule': 110},
    feedback_fn=None  # Key: no feedback = pure CA
)

history, _ = system.run(generations=50)
```

### Pattern 2: Binary CLA

```python
grid = Grid1D(size=50, radius=1, boundary='periodic', initial_states='random')
feedback = NeighbourhoodAgreementFeedback(radius=1)

system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': 10, 'initial_state': 'random'},
    feedback_fn=feedback,  # Key: feedback enables learning
    feedback_radius=1
)

grid_history, ta_history = system.run(generations=100)
```

### Pattern 3: Rule-Selection CLA

```python
grid = Grid1D(size=50, radius=1, boundary='periodic', initial_states='random')
feedback = NeighbourhoodAgreementFeedback(radius=1)

system = RuleSystem(
    rules=[30, 110],  # LA chooses between Rule 30 and Rule 110
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': 5, 'initial_state': 'random'},
    feedback_fn=feedback,
    feedback_radius=1
)

grid_history, ta_history = system.run(generations=100)
```

---

## Parameter Reference

### Grid1D Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `int >= 3` | required | Number of cells |
| `radius` | `int >= 1` | `1` | Neighborhood radius (size = 2r+1) |
| `boundary` | `str` | `'periodic'` | `'periodic'`, `'zero'`, or `'fixed'` |
| `initial_states` | `array` or `str` | `'random'` | `'random'`, `'single_center'`, `'half'`, or array |
| `k_states` | `int >= 2` | `2` | Number of possible cell states |

### TsetlinAutomaton Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | `int >= 1` | `5` | States per arm (total = 2N) |
| `initial_state` | `int` or `'random'` | `'random'` | Starting state in {1, ..., 2N} |

### WolframAutomaton Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rule` | `int` | required | Wolfram rule number (0-255) |

### AutomataSystem Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `Grid` | required | Grid instance |
| `automaton_type` | `class` | required | Automaton class to use |
| `automaton_params` | `dict` | `{}` | Parameters for automaton constructor |
| `feedback_fn` | `callable` or `None` | `None` | Feedback function (None = pure CA) |
| `feedback_radius` | `int >= 1` | `1` | Neighborhood radius for feedback |
| `seed` | `int` or `None` | `None` | Random seed |

### RuleSystem Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rules` | `list[int]` | required | List of Wolfram rules to choose from |

---

## Method Quick Reference

### Automaton Methods

```python
automaton.get_output(input)  # Returns output (state or action)
automaton.get_state()        # Returns internal state
automaton.reset()            # Resets to initial state

# LearningAutomaton only:
automaton.reward()           # Apply reward transition
automaton.penalize()         # Apply penalty transition
```

### Grid Methods

```python
grid.get_neighborhood(pos)   # Returns neighborhood array
grid.get_state(pos)          # Returns cell state
grid.set_state(pos, state)   # Sets cell state
grid.get_all_states()        # Returns full grid array
grid.set_all_states(array)   # Sets full grid
grid.get_positions()         # Returns iterable of positions
grid.reset()                 # Resets to initial state
```

### System Methods

```python
system.step()                # Advance one generation
system.run(generations)      # Run N generations, return history
system.reset()               # Reset all automata and grid
```

### SpaceTimePlot Methods

```python
plotter = SpaceTimePlot(grid_history, ta_state_history, n_states)
plotter.plot_standard(save_path, show)      # Binary plot
plotter.plot_ta_augmented(save_path, show)  # Confidence-encoded plot
```

---

## Common Wolfram Rules

| Rule | Description | Class |
|------|-------------|-------|
| 0 | All cells die | 1 (uniform) |
| 30 | Chaotic, random-looking | 3 (chaotic) |
| 110 | Complex structures, Turing-complete | 4 (complex) |
| 255 | All cells alive | 1 (uniform) |

---

## Boundary Conditions

| Type | Behavior | Use Case |
|------|----------|----------|
| `'periodic'` | Wrap-around (ring topology) | Removes edge effects, good for CLA |
| `'zero'` | Pad with zeros beyond edge | Classic CA, clean visualizations |
| `'fixed'` | Pad with fixed value | Similar to zero for binary states |

---

## Initial State Patterns

| Pattern | Description |
|---------|-------------|
| `'random'` | Random binary states |
| `'single_center'` | One cell at center = 1, rest = 0 |
| `'half'` | First half = 1, second half = 0 |
| `numpy.array([...])` | Custom pattern |

---

## Troubleshooting

### Import Error: No module named 'cakit'

```bash
# Install in editable mode from project root
pip install -e .
```

### Matplotlib GUI Crash on macOS

```python
# Add before any cakit imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Tests Not Found

```bash
# Make sure pytest is installed
pip install pytest

# Run from project root
python -m pytest tests/ -v
```

---

## Testing Individual Components

```bash
# Test automata only
pytest tests/automata/ -v

# Test specific file
pytest tests/automata/test_tsetlin.py -v

# Test specific test
pytest tests/automata/test_tsetlin.py::TestTsetlinAutomatonReward -v

# Show test coverage
pytest tests/ --cov=cakit --cov-report=html
```

---

## Performance Tips

1. **Use `seed` parameter** for reproducible experiments
2. **Periodic boundaries** are fastest (no special-case handling)
3. **For large grids**, consider using smaller `n_states` in TsetlinAutomaton
4. **Save plots with `show=False`** to avoid GUI overhead
5. **History arrays** can be large for many generations - consider processing incrementally

---

## Research Workflow

```python
# 1. Define experimental configuration
config = {
    'grid_size': 100,
    'n_states': 10,
    'feedback_radius': 1,
    'generations': 200,
    'seed': 42
}

# 2. Set up system
grid = Grid1D(size=config['grid_size'], radius=1, boundary='periodic')
feedback = NeighbourhoodAgreementFeedback(radius=config['feedback_radius'])
system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': config['n_states']},
    feedback_fn=feedback,
    seed=config['seed']
)

# 3. Run experiment
grid_history, ta_history = system.run(generations=config['generations'])

# 4. Analyze results
convergence_point = analyze_convergence(grid_history)
final_pattern = grid_history[-1]

# 5. Visualize
plotter = SpaceTimePlot(grid_history, ta_history, n_states=config['n_states'])
plotter.plot_ta_augmented(save_path=f"experiment_{config['seed']}.png")
```

---

## Quick Checks

Verify installation:
```bash
python -c "import cakit; print(cakit.__version__)"
```

Verify all tests pass:
```bash
pytest tests/ -q
```

Generate sample output:
```bash
python examples/demo_rule_110.py
ls -lh *.png
```
