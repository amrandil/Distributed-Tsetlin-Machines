# cakit - Cellular Automata Kit

A modular Python framework for building cellular automata (CA), learning automata (LA), and cellular learning automata (CLA) systems.

## Overview

`cakit` is a research framework designed for experimenting with distributed learning systems based on cellular automata. It provides a clean, modular architecture where:

- Each component can be used standalone or as part of a larger system
- New automaton types, feedback strategies, and system configurations can be added through subclassing
- The same code works for both classical CA and learning-augmented CLA systems

## Features

- **Modular architecture** with four independent layers: automata, grids, systems, feedback
- **Multiple automaton types**: WolframAutomaton (fixed rules), TsetlinAutomaton (2-armed FSSA)
- **Flexible grid configurations**: 1D grids with periodic, zero, or fixed boundaries
- **Two system types**:
  - `StateSystem`: automaton output becomes cell state directly
  - `RuleSystem`: automaton output selects which Wolfram rule to apply
- **Pluggable feedback functions** for learning experiments
- **Visualization tools**: standard space-time plots and TA-augmented plots showing automaton confidence
- **Fully tested**: 124 unit tests covering all components

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Distributed-Tsetlin-Machines

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install cakit in editable mode
pip install -e .
```

## Quick Start

### Example 1: Rule 110 Elementary CA

```python
from cakit.automata import WolframAutomaton
from cakit.grid import Grid1D
from cakit.system import StateSystem
from cakit.visualization import SpaceTimePlot

# Create a 1D grid with single center cell activated
grid = Grid1D(
    size=101,
    radius=1,
    boundary='zero',
    initial_states='single_center'
)

# Create a StateSystem with Rule 110
system = StateSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params={'rule': 110},
    feedback_fn=None  # Pure CA, no learning
)

# Run for 50 generations
grid_history, _ = system.run(generations=50)

# Visualize
plotter = SpaceTimePlot(grid_history=grid_history)
plotter.plot_standard(save_path='rule_110.png', show=False)
```

### Example 2: Binary CLA with Tsetlin Automata

```python
from cakit.automata import TsetlinAutomaton
from cakit.grid import Grid1D
from cakit.system import StateSystem
from cakit.feedback import NeighbourhoodAgreementFeedback
from cakit.visualization import SpaceTimePlot

# Create grid and feedback function
grid = Grid1D(size=51, radius=1, boundary='periodic', initial_states='random')
feedback_fn = NeighbourhoodAgreementFeedback(radius=1)

# Create a CLA system
cla = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': 10, 'initial_state': 'random'},
    feedback_fn=feedback_fn,
    feedback_radius=1
)

# Run and visualize
grid_history, ta_history = cla.run(generations=100)

plotter = SpaceTimePlot(
    grid_history=grid_history,
    ta_state_history=ta_history,
    n_states=10
)
plotter.plot_ta_augmented(save_path='cla_augmented.png', show=False)
```

## Architecture

```
cakit/
├── automata/       # Automaton implementations
│   ├── base.py              # Automaton (abstract)
│   ├── learning.py          # LearningAutomaton (abstract)
│   ├── tsetlin.py           # TsetlinAutomaton
│   └── wolfram.py           # WolframAutomaton
│
├── grid/           # Spatial grid structures
│   ├── base.py              # Grid (abstract)
│   └── grid_1d.py           # Grid1D
│
├── system/         # Automata systems
│   ├── base.py              # AutomataSystem (abstract)
│   ├── state_system.py      # StateSystem
│   └── rule_system.py       # RuleSystem
│
├── feedback/       # Feedback functions for learning
│   ├── base.py              # FeedbackFunction (abstract)
│   └── neighbourhood_agreement.py
│
└── visualization/  # Plotting tools
    └── spacetime.py         # SpaceTimePlot
```

## Design Principles

1. **Modularity**: Every component is plug-and-play
2. **Open extensibility**: New types added by subclassing, not by modifying core code
3. **Standalone usability**: Each layer works independently
4. **Separation of concerns**: Spatial logic, automaton logic, and feedback are fully decoupled
5. **Automaton agnosticism**: Systems call only `get_output()` — they don't distinguish between plain automata and learning automata

## Running Examples

```bash
# Rule 110 elementary CA
python examples/demo_rule_110.py

# Binary CLA with learning
python examples/demo_cla.py
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/automata/ -v
pytest tests/grid/ -v
pytest tests/system/ -v
pytest tests/feedback/ -v
```

## Extending the Framework

### Adding a new automaton type

```python
from cakit.automata import LearningAutomaton

class MyAutomaton(LearningAutomaton):
    def get_output(self, input=None):
        # Return action index
        ...
    
    def reward(self):
        # Update internal state
        ...
    
    def penalize(self):
        # Update internal state
        ...
    
    def get_state(self):
        return self._state
    
    def reset(self):
        self._state = self._initial_state

# Use it anywhere
system = StateSystem(
    grid=grid,
    automaton_type=MyAutomaton,
    automaton_params={...}
)
```

### Adding a new feedback strategy

```python
from cakit.feedback import FeedbackFunction

class MyFeedback(FeedbackFunction):
    def __call__(self, position, grid_before, grid_after):
        # Return True for reward, False for penalty
        ...

# Plug it in
system = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={...},
    feedback_fn=MyFeedback()
)
```

## Documentation

For detailed component specifications and design rationale, see [`project_description.md`](project_description.md).

## Requirements

- Python >= 3.8
- numpy >= 1.24.0
- matplotlib >= 3.5.0
- pytest >= 7.0.0 (for testing)

## License

[To be determined]

## Citation

[To be determined]

## Author

Amr Kandil
