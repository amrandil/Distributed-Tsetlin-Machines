# Distributed Cellular Learning Automata — Framework Design Specification

**Version:** 3.0 Draft  
**Author:** Amr Kandil

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Component: Automaton](#4-component-automaton)
5. [Component: LearningAutomaton](#5-component-learningautomaton)
6. [Component: TsetlinAutomaton](#6-component-tsetlinautomaton)
7. [Component: Grid](#7-component-grid)
8. [Component: Grid1D](#8-component-grid1d)
9. [Component: AutomataSystem](#9-component-automatasystem)
10. [Component: StateSystem](#10-component-statesystem)
11. [Component: RuleSystem](#11-component-rulesystem)
12. [Component: SpaceTimePlot](#12-component-spacetimeplot)
13. [Data Flow](#13-data-flow)
14. [Module Structure](#14-module-structure)
15. [Extensibility Guide](#15-extensibility-guide)
16. [Design Decisions](#16-design-decisions)
17. [Out of Scope — v1.0](#17-out-of-scope--v10)

---

## 1. Overview

This document specifies the design of a Distributed Cellular Learning Automata (CLA)
framework. The framework is intended as the codebase for an academic thesis and as a
published open-source library.

The central contribution is a **true per-cell CLA architecture**: each cell in the grid
hosts an independent Learning Automaton that learns locally from its neighbourhood. 

---

## 2. Design Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Every component is a plug-in that can be swapped without touching any other component. |
| **Open extensibility** | New automaton types, LA types, interpreter strategies, and feedback strategies are added by subclassing, not by modifying core code. |
| **Standalone usability** | `Automaton`, `LearningAutomaton`, and `Grid` subclasses can each be used independently without the full CLA stack. |
| **Separation of concerns** | Spatial mechanics (Grid), automaton logic (Automaton/LA), output interpretation (_interpret), and environment signal (feedback_fn) are four fully independent concerns. |
| **Automaton agnosticism** | `AutomataSystem` and its subclasses never distinguish between plain automata and learning automata. Both produce an output via `get_output()`. The system interprets that output the same way regardless of which automaton type produced it. |
| **Dimension agnosticism** | The `AutomataSystem` layer never handles spatial logic directly. All spatial operations are delegated to the `Grid` instance. |

---

## 3. Architecture Diagram

### Full inheritance tree

```
Automaton                          (abstract)
└── LearningAutomaton              (abstract)
    └── TsetlinAutomaton
    └── [future: VSSAutomaton, KrinkyAutomaton, ...]

Grid                               (abstract)
└── Grid1D
    [Grid2D — v2, extension path]

AutomataSystem                     (abstract)
├── StateSystem
└── RuleSystem
```

### Component interaction

```
┌──────────────────────────────────────────────────────────────────┐
│                         AutomataSystem                           │
│                                                                  │
│   ┌────────────────────┐      ┌─────────────────────────────┐   │
│   │       Grid         │      │   Automaton grid             │   │
│   │                    │      │                             │   │
│   │  cell states       │      │  [A_0][A_1] ... [A_n]       │   │
│   │  get_neighborhood()│      │                             │   │
│   │  boundary logic    │      │  any Automaton subclass     │   │
│   │  get_positions()   │      │  get_output()               │   │
│   └────────┬───────────┘      └─────────────┬───────────────┘   │
│            │ neighborhood_i                 │ output_i           │
│            └─────────────────┬─────────────┘                    │
│                              ▼                                   │
│               _interpret(output, neighborhood)                   │
│                   (defined by each subclass)                     │
│                              │                                   │
│                              ▼                                   │
│                          next_grid                               │
│                              │                                   │
│              (optional)      ▼                                   │
│               feedback_fn(cell, grid_before, grid_after)         │
│                   → reward() or penalize() on LA                 │
│                   (only called if feedback_fn is not None)       │
└──────────────────────────────────────────────────────────────────┘
```

### CA vs CLA — configuration, not class structure

The distinction between a CA and a CLA is not structural. It is determined entirely
by whether a `feedback_fn` is provided:

```
StateSystem(feedback_fn=None)                →  pure CA behaviour
StateSystem(feedback_fn=NeighbourhoodAgreementFeedback(...))  →  CLA behaviour

RuleSystem(feedback_fn=None)                 →  pure rule-based CA
RuleSystem(feedback_fn=NeighbourhoodAgreementFeedback(...))   →  rule-based CLA
```

---

## 4. Component: Automaton

> **Type:** Abstract base class  
> **Standalone:** Yes — can be used against any environment

### Purpose

The most fundamental unit in the framework. An `Automaton` has an internal state and
produces an output given its current state. In a CA, each cell holds an `Automaton`.
The neighbourhood is passed in and the output is the cell's contribution to the next
grid state.

### Interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_output` | `(input) → any` | Given an input, returns the automaton's output based on its current internal state. For a plain automaton this is the next cell state directly. For a `LearningAutomaton` this is an action index. The meaning of the output is always interpreted by the `AutomataSystem` subclass, never by the automaton itself. |
| `get_state` | `() → any` | Returns the current internal state. |
| `reset` | `() → None` | Resets internal state to the initial value. |

### Notes

- `get_output(input)` is the single canonical output method across the entire
  automaton hierarchy. The name is intentionally neutral — it does not presuppose
  whether the output is a cell state or an action index. That distinction belongs
  to the interpreter, not the automaton.
- The `input` type is unspecified at this level. For CA cells it will be a
  neighbourhood array. Subclasses define what input they expect.

---

## 5. Component: LearningAutomaton

> **Type:** Abstract base class, extends `Automaton`  
> **Standalone:** Yes — can be used against any environment that calls `reward()` or
> `penalize()`

### Purpose

Extends `Automaton` with a reinforcement learning mechanism. A `LearningAutomaton`
not only produces an output — it also learns over time which action to prefer, guided
by reward and penalty signals from the environment.

This is the interface all LA types must implement. `AutomataSystem` interacts with LAs
exclusively through the `Automaton` interface (`get_output()`) and this extension
(`reward()`, `penalize()`). This makes all LA types fully interchangeable.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `action_set_size` | `int ≥ 2` | Number of actions. Default: `2`. The action set is always integer indices `{0, 1, …, action_set_size − 1}`. |

> **On `internal_structure`:** This is an implementation detail owned entirely by each
> concrete subclass. `TsetlinAutomaton` uses an integer state. A future `VSSAutomaton`
> would use a probability vector. Neither is exposed on the base class — only the
> interface matters here.

### Interface

Inherits from `Automaton`: `get_output()`, `get_state()`, `reset()`

Additionally defines:

| Method | Signature | Description |
|--------|-----------|-------------|
| `reward` | `() → None` | Applies the reward transition. Moves the automaton toward committing to the current action. |
| `penalize` | `() → None` | Applies the penalty transition. Moves the automaton away from the current action. |

### Notes

- `get_output(input)` on a `LearningAutomaton` ignores `input` and returns the current
  action index `{0, 1, …, action_set_size − 1}` based on internal state. The `input`
  parameter is accepted for interface consistency with `Automaton` but is unused.
- Any class implementing `get_output()`, `reward()`, `penalize()`, `get_state()`, and
  `reset()` can be used as a drop-in in any `AutomataSystem` subclass.

---

## 6. Component: TsetlinAutomaton

> **Type:** Concrete class, extends `LearningAutomaton`  
> **Standalone:** Yes

### Purpose

Concrete implementation of the 2-armed Tsetlin Automaton. A Fixed Structure Stochastic
Automaton (FSSA) with deterministic state transitions. `action_set_size` is fixed at
`2` in v1.0.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_states` (N) | `int ≥ 1` | Number of states per arm. Higher N = slower learning, more stable convergence. Total states = `2 × N`. |
| `initial_state` | `int` or `'random'` | Starting state in `{1, …, 2N}`. If `'random'`, sampled uniformly. |

### Internal structure

A single integer `s ∈ {1, …, 2N}`:

- **Arm (action):** `a = ⌊(s − 1) / N⌋`
  - Arm 0: states `{1, …, N}`
  - Arm 1: states `{N+1, …, 2N}`
- **Position within arm:** `p = (s − 1) mod N`
  - `p = 0` — boundary end (least committed)
  - `p = N − 1` — extreme end (most committed)

### Transition rules

**Reward** — move one step toward the committed extreme:

```
s → s + 1    if p < N − 1
s → s         if p = N − 1   (already at extreme, stay)
```

**Penalty** — move one step toward the transition point:

```
s → s − 1              if p > 0
s → N + 1              if p = 0 and arm = 0   (at arm 0 boundary → switch to arm 1 boundary)
s → 1                  if p = 0 and arm = 1   (at arm 1 boundary → switch to arm 0 boundary)
```

### Full interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_output` | `(input=None) → int` | Returns action index `{0, 1}` based on current arm. Input ignored. |
| `get_state` | `() → int` | Returns current state `s ∈ {1, …, 2N}`. |
| `get_arm` | `() → int` | Returns current arm `{0, 1}`. Equivalent to `get_output()`. |
| `get_position` | `() → int` | Returns position within current arm `{0, …, N−1}`. Used for TA-augmented space-time plot. |
| `reward` | `() → None` | Applies reward transition. |
| `penalize` | `() → None` | Applies penalty transition. |
| `reset` | `() → None` | Resets to `initial_state`. |

### Standalone example

```python
ta = TsetlinAutomaton(n_states=5, initial_state='random')

for _ in range(100):
    action = ta.get_output()      # returns 0 or 1
    if environment_says_good:
        ta.reward()
    else:
        ta.penalize()
```

---

## 7. Component: Grid

> **Type:** Abstract base class  
> **Standalone:** Yes — can be used as a pure spatial data structure

### Purpose

Owns all spatial mechanics: cell state storage, neighbourhood extraction, and boundary
condition handling. Knows nothing about automata or learning. `AutomataSystem` delegates
all spatial operations to a `Grid` instance, which is what makes the system fully
dimension-agnostic.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `boundary` | `str` | `'periodic'` (wrap-around), `'fixed'` (constant boundary value), `'zero'` (pad with 0s). |
| `initial_states` | `array` or `str` | Initial cell states. Numpy array of correct shape, or a key: `'random'`, `'single_center'`, `'half'`. |
| `k_states` | `int ≥ 2` | Number of possible cell states. Default: `2`. Reserved for future non-binary extension. |

### Interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_neighborhood` | `(position) → array` | Returns neighbourhood array for cell at `position`. |
| `get_state` | `(position) → int` | Returns current state of cell at `position`. |
| `set_state` | `(position, state) → None` | Sets state of cell at `position`. |
| `get_all_states` | `() → array` | Returns full grid state array. |
| `set_all_states` | `(array) → None` | Sets full grid state array. |
| `get_positions` | `() → iterable` | Returns all valid positions. `int` in 1D, `(row, col)` in 2D. |
| `reset` | `() → None` | Resets grid to `initial_states`. |

### Notes

- `get_positions()` is the key method that makes `AutomataSystem` dimension-agnostic.
  The system iterates with `for pos in grid.get_positions()` and never thinks about
  whether positions are integers or tuples.
- `position` type is an `int` in `Grid1D` and a `(row, col)` tuple in `Grid2D`.
  `AutomataSystem` subclasses treat `position` as an opaque token — they pass it to
  `Grid` methods without inspecting it.

---

## 8. Component: Grid1D

> **Type:** Concrete class, extends `Grid`  
> **Standalone:** Yes

### Purpose

1D grid implementation. Cells indexed by a single integer. Neighbourhood is a
contiguous slice of the array centred on the cell.

### Parameters

Inherits from `Grid`: `boundary`, `initial_states`, `k_states`

Additionally defines:

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int ≥ 3` | Number of cells. |
| `radius` | `int ≥ 1` | Neighbourhood radius. Neighbourhood of cell `i` = `{i−r, …, i, …, i+r}`, size = `2r+1`. |

### Neighbourhood extraction

```
neighborhood = [cell_{i-r}, ..., cell_i, ..., cell_{i+r}]   (length 2r+1)
```

Boundary conditions at edges:
- `'periodic'` — wraps: position `−1` maps to `size − 1`
- `'zero'` — pads with `0` beyond the edge
- `'fixed'` — pads with a fixed constant beyond the edge

### Standalone example

```python
grid = Grid1D(size=100, radius=1, boundary='periodic',
              initial_states='single_center')

neighborhood = grid.get_neighborhood(0)   # array of 3 cell states
all_states   = grid.get_all_states()      # numpy array of length 100
```

---

## 9. Component: AutomataSystem

> **Type:** Abstract base class  
> **Standalone:** No — requires a configured `Grid` instance and an automaton type

### Purpose

Abstract base for all systems that attach automata to a grid and evolve them over time.
Owns the step loop, history recording, automaton grid construction, and the optional
feedback loop. Never handles spatial logic directly — always delegates to `Grid`.

The system is **automaton-agnostic**: it calls only `get_output()` on each cell's
automaton and passes the result to `_interpret()`. It does not know or care whether
the automaton is a plain `Automaton` or a `LearningAutomaton`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid` | `Grid` | A configured `Grid1D` instance. Provides all spatial operations. |
| `automaton_type` | `class` | The `Automaton` subclass to instantiate at each cell. Must implement the `Automaton` interface. |
| `automaton_params` | `dict` | Keyword arguments forwarded to each automaton constructor. |
| `feedback_fn` | `callable` or `None` | Feedback strategy. Called after each step if not `None`. Signature: `feedback_fn(position, grid_before, grid_after) → bool`. Returns `True` for reward, `False` for penalty. Default: `None` (pure CA behaviour). |
| `feedback_radius` | `int ≥ 1` | Neighbourhood radius used by `feedback_fn`. Configurable independently of the `Grid` radius. Only relevant when `feedback_fn` is not `None`. |
| `seed` | `int` or `None` | Random seed for reproducibility. |

### Interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `step` | `() → None` | Advances the system by one generation. |
| `run` | `(generations) → history` | Runs for N generations. Returns history. |
| `reset` | `() → None` | Resets all automata and the grid to initial state. |
| `_build_automata` | `() → dict` | Instantiates one automaton per grid position. Returns `{position: automaton}`. Subclasses override if custom construction is needed. |
| `_interpret` | `(output, neighborhood) → int` *(abstract)* | Translates automaton output into a concrete next cell state. Defined by each subclass. |

### Step logic

```
for each position in grid.get_positions():
    neighborhood = grid.get_neighborhood(position)
    output       = automata[position].get_output()
    next_state   = _interpret(output, neighborhood)

grid.set_all_states(next_states)

if feedback_fn is not None:
    for each position in grid.get_positions():
        reward = feedback_fn(position, grid_before, grid_after)
        if reward:
            automata[position].reward()
        else:
            automata[position].penalize()

record grid states and automaton states in history
```

### Notes

- `feedback_fn` calls `reward()` and `penalize()` on the automaton. For plain
  `Automaton` subclasses these methods do not exist — it is the caller's responsibility
  to only provide a `feedback_fn` when using a `LearningAutomaton` type. This is
  documented but not enforced by the type system.
- `_build_automata()` is a method rather than inline code in `__init__` so subclasses
  can override it cleanly when automata need access to grid information at construction
  time, without modifying the base class.

---

## 10. Component: StateSystem

> **Type:** Concrete class, extends `AutomataSystem`  
> **Standalone:** Yes

### Purpose

A system where each cell's automaton output is interpreted directly as the next cell
state. The neighbourhood is not used in the interpretation — the output value is the
cell state.

Used as a **pure CA** when `feedback_fn=None`, or as a **binary-output CLA** when a
`feedback_fn` is provided and the automaton type is a `LearningAutomaton`.

### Parameters

Inherits all from `AutomataSystem`. No additional parameters.

### Implements

| Method | Behaviour |
|--------|-----------|
| `_interpret(output, neighborhood)` | Returns `output` directly. Neighbourhood ignored. |

### Examples

**As a pure CA:**

```python
grid = Grid1D(size=100, radius=1, boundary='periodic',
              initial_states='single_center')

# A plain automaton that applies a fixed Wolfram rule
ca = StateSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params={'rule': 110},
    feedback_fn=None
)
history = ca.run(generations=50)
```

**As a CLA (binary output):**

```python
grid = Grid1D(size=100, radius=1, boundary='periodic',
              initial_states='random')

cla = StateSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': 5, 'initial_state': 'random'},
    feedback_fn=NeighbourhoodAgreementFeedback(radius=1),
    feedback_radius=1
)
grid_history, ta_history = cla.run(generations=100)
```

---

## 11. Component: RuleSystem

> **Type:** Concrete class, extends `AutomataSystem`  
> **Standalone:** Yes

### Purpose

A system where each cell's automaton output is interpreted as a Wolfram rule index.
The chosen rule is then applied to the cell's neighbourhood to produce the next cell
state. The neighbourhood is central to the interpretation.

Used as a **pure rule-selection CA** when `feedback_fn=None`, or as a
**rule-selection CLA** when a `feedback_fn` is provided and the automaton type is
a `LearningAutomaton`.

### Parameters

Inherits all from `AutomataSystem`. Additionally defines:

| Parameter | Type | Description |
|-----------|------|-------------|
| `rules` | `list[int]` | The Wolfram rule numbers the automaton selects between. `rules[output_index]` gives the rule number to apply. e.g. `[30, 110]` — output `0` → Rule 30, output `1` → Rule 110. Length must equal `automaton_type.action_set_size`. |

### Implements

| Method | Behaviour |
|--------|-----------|
| `_interpret(output, neighborhood)` | Looks up `rules[output]` and applies that Wolfram rule's lookup table to `neighborhood`. Returns the resulting cell state. |

### Examples

**As a pure rule-selection CA:**

```python
grid = Grid1D(size=100, radius=1, boundary='periodic',
              initial_states='random')

ca = RuleSystem(
    grid=grid,
    automaton_type=WolframAutomaton,
    automaton_params={'rule': 30},
    rules=[30, 110],
    feedback_fn=None
)
history = ca.run(generations=50)
```

**As a rule-selection CLA:**

```python
grid = Grid1D(size=100, radius=1, boundary='periodic',
              initial_states='random')

cla = RuleSystem(
    grid=grid,
    automaton_type=TsetlinAutomaton,
    automaton_params={'n_states': 5, 'initial_state': 'random'},
    rules=[30, 110],
    feedback_fn=NeighbourhoodAgreementFeedback(radius=1),
    feedback_radius=1
)
grid_history, ta_history = cla.run(generations=100)
```

---

## 12. Component: SpaceTimePlot

> **Type:** Standalone utility class

### Purpose

Produces space-time diagrams from history arrays returned by `run()`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_history` | `ndarray, shape (T, G)` | Cell states for all T generations and G cells. |
| `ta_state_history` | `ndarray, shape (T, G)` | TA internal states. Optional — only required for the augmented plot. |
| `n_states` | `int` | N (states per arm) — used to normalise shade intensity in the augmented plot. |
| `cell_size` | `int` | Pixel size per cell in the output image. |
| `figsize` | `tuple` | Matplotlib figure size. |

### Interface

| Method | Description |
|--------|-------------|
| `plot_standard()` | Classic binary space-time diagram. Black = state 1, white = state 0. |
| `plot_ta_augmented()` | Encodes both cell state and TA confidence per cell. See encoding below. |

### TA-augmented colour encoding

- **Dead cell (state = 0) — blue ramp:**
  darker blue = deeply committed to action 0 (position near `N−1`);
  lighter blue = near transition point (position near `0`).
- **Live cell (state = 1) — red ramp:**
  lighter red = near transition point (position near `0`);
  darker red = deeply committed to action 1 (position near `N−1`).

Shade is linearly interpolated: `intensity = get_position() / (N − 1)`.

---

## 13. Data Flow

### Standalone Grid1D

```
Grid1D(size, radius, boundary, initial_states)
    │
    ├── get_neighborhood(i) ──► array of 2r+1 states
    ├── get_state(i)         ──► int
    └── get_positions()      ──► [0, 1, 2, ..., size-1]
```

### Standalone TsetlinAutomaton

```
TsetlinAutomaton(n_states, initial_state)
    │
    ▼
get_output() ─────────────────────────► action index {0, 1}
    │
    │  environment responds
    ▼
reward()  or  penalize()
    │
    ▼
internal state updated ──► repeat
```

### StateSystem — pure CA (feedback_fn=None)

```
StateSystem(grid, automaton_type=WolframAutomaton, feedback_fn=None)
    │
    ▼
step():
  for each pos in grid.get_positions():
      neighborhood = grid.get_neighborhood(pos)
      output       = automata[pos].get_output()
      next_state   = _interpret(output, neighborhood)
      → returns output directly (state = output)
  grid.set_all_states(next_states)
  [no feedback — feedback_fn is None]
    │
    ▼
repeat N generations ──► grid_history
```

### RuleSystem — CLA (feedback_fn provided)

```
RuleSystem(grid, automaton_type=TsetlinAutomaton,
           rules=[30, 110], feedback_fn=NeighbourhoodAgreementFeedback(...))
    │
    ▼
step():
  ┌──────────────────────────────────────────────────────┐
  │ for each pos in grid.get_positions():                │
  │     neighborhood = grid.get_neighborhood(pos)        │  ← Grid: spatial
  │     output       = automata[pos].get_output()        │  ← LA: decides
  │     next_state   = _interpret(output, neighborhood)  │  ← subclass: meaning
  │         → looks up rules[output], applies to nbhd    │
  └──────────────────────────────────────────────────────┘
    │
    ▼
  grid.set_all_states(next_states)
    │
    ▼
  ┌──────────────────────────────────────────────────────┐
  │ for each pos in grid.get_positions():                │
  │     reward = feedback_fn(pos, grid_before, grid_after│  ← feedback: signal
  │     if reward: automata[pos].reward()                │  ← LA: learns
  │     else:      automata[pos].penalize()              │
  └──────────────────────────────────────────────────────┘
    │
    ▼
record grid_history and ta_state_history
    │
    ▼
repeat N generations ──► (grid_history, ta_state_history)
```

---

## 14. Module Structure

```
project/
├── core/
│   ├── automaton.py               # Abstract: Automaton
│   ├── learning_automaton.py      # Abstract: LearningAutomaton(Automaton)
│   ├── tsetlin_automaton.py       # Concrete: TsetlinAutomaton(LearningAutomaton)
│   ├── grid.py                    # Abstract: Grid
│   ├── grid_1d.py                 # Concrete: Grid1D(Grid)
│   ├── automata_system.py         # Abstract: AutomataSystem
│   ├── state_system.py            # Concrete: StateSystem(AutomataSystem)
│   └── rule_system.py             # Concrete: RuleSystem(AutomataSystem)
│
├── feedback/
│   ├── base.py                    # Abstract: FeedbackFunction
│   └── neighbourhood_agreement.py # Concrete: NeighbourhoodAgreementFeedback
│
├── visualization/
│   └── spacetime.py               # SpaceTimePlot
│
├── experiments/
│   └── run_experiment.py          # Config-driven experiment runner
│
├── tests/
│   ├── test_automaton.py
│   ├── test_tsetlin_automaton.py
│   ├── test_grid_1d.py
│   ├── test_automata_system.py
│   ├── test_state_system.py
│   └── test_rule_system.py
│
└── config.py                      # Default parameter values
```

---

## 15. Extensibility Guide

### Adding a new LA type

Subclass `LearningAutomaton` and implement:
`get_output()`, `get_state()`, `reset()`, `reward()`, `penalize()`.
Pass the class to any system via `automaton_type`. No other changes required.

```python
class VSSAutomaton(LearningAutomaton):
    # internal_structure: probability vector over actions
    def get_output(self, input=None): ...   # sample from probability vector
    def reward(self): ...                   # increase probability of current action
    def penalize(self): ...                 # decrease probability of current action

cla = StateSystem(grid=grid, automaton_type=VSSAutomaton, automaton_params={...},
                  feedback_fn=NeighbourhoodAgreementFeedback(radius=1))
```

### Adding a new feedback strategy

Create a callable with the signature
`__call__(position, grid_before, grid_after) → bool`
and pass it as `feedback_fn`. No other changes required.

```python
class TargetPatternFeedback:
    def __init__(self, target):
        self.target = target
    def __call__(self, position, grid_before, grid_after):
        return grid_after[position] == self.target[position]

cla = StateSystem(grid=grid, automaton_type=TsetlinAutomaton,
                  automaton_params={...},
                  feedback_fn=TargetPatternFeedback(target=my_target))
```

### Adding a new interpreter (new AutomataSystem subclass)

Subclass `AutomataSystem` and implement `_interpret(output, neighborhood) → int`.
The automaton type, grid type, and feedback strategy all remain unchanged.

```python
class ThresholdSystem(AutomataSystem):
    def _interpret(self, output, neighborhood):
        # custom mapping from output and neighborhood to next cell state
        ...
```

### Extending to 2D

Subclass `Grid` and implement the full `Grid` interface for 2D indexing.
Pass a `Grid2D` instance to any `AutomataSystem` subclass.
No changes to any automaton, system, or feedback class are required.

```python
class Grid2D(Grid):
    def get_neighborhood(self, position): ...   # position = (row, col)
    def get_positions(self): ...               # yields (row, col) tuples

cla = RuleSystem(grid=Grid2D(...), automaton_type=TsetlinAutomaton, ...)
```

---

## 16. Design Decisions

| Decision | Rationale |
|----------|-----------|
| One LA per cell | Canonical CLA definition (Beigy & Meybodi, 2004). Global-rule approaches (Westlin & Oo, 2021) are a documented deviation with known implementation bugs. |
| CA vs CLA is configuration, not class structure | The only difference between a CA and a CLA is the presence of a feedback loop. Making this a parameter (`feedback_fn`) rather than a structural branch avoids a combinatorial class hierarchy and reflects the conceptual truth: a CLA is a CA with a learning mechanism added. |
| `AutomataSystem` is automaton-agnostic | The system calls only `get_output()` on each cell's automaton. It never distinguishes between plain automata and learning automata. This means any LA type (Tsetlin, VSSA, Krinky, etc.) drops in with zero changes to the system. |
| `get_output()` as the single canonical method | Rather than having separate `update_state()` for plain automata and `get_action()` for LAs, a single `get_output()` method covers both cases. The meaning of the output is determined by `_interpret()` in the system, not by the automaton. |
| `_interpret()` is abstract on `AutomataSystem` | Different system configurations differ precisely in how they map automaton output to cell state. Making this the extension point keeps all step logic shared and stable. |
| `feedback_fn` is a plug-in callable | Different research questions require different feedback signals. A plug-in avoids hardcoding experimental assumptions into the framework. |
| `feedback_radius` is independent of `Grid` radius | The scale at which a cell evaluates its success may differ from the CA transition scale. Conflating them would be an unjustified constraint. |
| `Grid` is separate from `AutomataSystem` | Spatial mechanics are independent of what lives in cells. The separation makes 2D extension trivial — swap the `Grid`, nothing else changes. |
| `Grid` uses abstract base with `Grid1D`/`Grid2D` subclasses | 1D and 2D have meaningfully different neighbourhood logic, boundary wrapping, and position types. Subclassing is cleaner than internal dimension branching. |
| `_build_automata()` is a method not inline code | Allows subclasses to override construction cleanly if automata need grid information at build time, without modifying the base class. |

### Known limitations and future optimisations

**One automaton object per cell.** The current design instantiates one Python object
per cell, stored in a dictionary keyed by position. For 1D grids of a few hundred
cells this is negligible. For large 2D grids (e.g. 1000×1000 = one million cells)
the object overhead becomes significant in both memory and iteration speed. A future
optimisation would replace the dict of objects with a vectorised representation —
storing all TA states as a single numpy array and implementing reward/penalize as
array operations. This would not require changes to the public interface, only to
the internal implementation of `_build_automata()` and the automaton update logic.

**`Grid` abstract base vs single parameterised class.** The choice to use an abstract
`Grid` base with `Grid1D` and `Grid2D` subclasses adds class hierarchy overhead for
a feature not yet implemented (2D). If the 2D case turns out to have nearly identical
logic to 1D (differing only in indexing), it may be simpler to collapse `Grid` into
a single class parameterised by `shape` using numpy's native N-dimensional array
support. The migration path is clean: the public interface of `Grid` does not change,
only the internal structure.

---

## 17. Out of Scope — v1.0

- `Grid2D` implementation (extension path documented in Sections 8 and 15).
- Non-binary cell states (`k_states > 2`). Parameter reserved.
- Multi-arm Tsetlin Automaton (`action_set_size > 2`).
- VSSA, Krinky, and other LA types beyond `TsetlinAutomaton`.
- Asynchronous or partially synchronous update schedules.
- Multi-layer or hierarchical system architectures.
- GPU acceleration or vectorised automata (noted as future optimisation in Section 16).
- MNIST classification experiments (planned after core framework is validated).