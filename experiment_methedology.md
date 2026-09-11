---

## Experimental Setup

This experiment investigates emergent spatial behavior in a one-dimensional Cellular Learning Automata (CLA) where each cell hosts an independent Tsetlin Automaton that selects between two competing Wolfram elementary CA rules.

### System Configuration

The CLA consists of a one-dimensional grid of 201 cells with periodic boundary conditions. Each cell contains a Tsetlin Automaton with two actions (arm length N = 5, yielding 10 internal states per automaton). The two actions correspond to two distinct elementary CA rules assigned at the start of each experimental run. All cells share the same rule pair but maintain independent learning dynamics.

The grid is initialized with random binary states, and each Tsetlin Automaton begins in a randomly selected internal state. The system evolves synchronously for 500 time steps. At each step, every cell's TA selects a rule based on its current internal state, applies that rule to determine the cell's next state, receives feedback from the environment, and updates its internal state accordingly.

### Rule Selection

Of the 256 elementary CA rules, 88 are behaviorally unique under symmetry transformations. These are distributed across Wolfram's four behavioral classes as follows:

- **Class I** (fixed-point): 8 rules — 0, 8, 32, 40, 128, 136, 160, 168
- **Class II** (periodic): 65 rules — 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76, 77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142, 152, 154, 156, 162, 164, 170, 172, 178, 184, 200, 204, 232
- **Class III** (chaotic): 11 rules — 18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150
- **Class IV** (complex): 4 rules — 41, 54, 106, 110

### Experimental Conditions

The experiment tests all unique unordered cross-class pairs of rules. Within-class pairs (such as Class I × Class I or Class III × Class III) are excluded, as the primary interest lies in how rules from different behavioral classes interact when competing within the same CLA.

The cross-class comparisons yield 1,659 experimental conditions:

- Class I × Class II: 520 pairs
- Class I × Class III: 88 pairs
- Class I × Class IV: 32 pairs
- Class II × Class III: 715 pairs
- Class II × Class IV: 260 pairs
- Class III × Class IV: 44 pairs

Each rule pair is evaluated in a single run using a fixed random seed to ensure reproducibility and enable direct comparison across conditions.

### Evaluation

Results are evaluated through visual inspection of space-time diagrams. The primary phenomena of interest are:

1. **Spatial clustering**: Whether cells partition into spatial regions where neighboring cells converge to the same rule, producing vertical bands in the space-time diagram with distinct dynamical behaviors.

2. **Emergent complexity**: Whether pairing rules from ordered classes (I or II) with rules from the chaotic class (III) produces edge-of-chaos dynamics characteristic of Class IV, consistent with the hypothesis that Class IV behavior emerges at the boundary between order and chaos.

3. **Boundary dynamics**: The behavior at interfaces between regions dominated by different rules, including whether boundaries are stable, drifting, or exhibit non-trivial structure.

---

How does this look?