"""
Demo script: Binary CLA with Tsetlin Automata

This script demonstrates a Cellular Learning Automaton where each cell hosts
a Tsetlin Automaton learning to synchronize with its neighbors.
"""

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from cakit.automata import TsetlinAutomaton
from cakit.grid import Grid1D
from cakit.system import StateSystem
from cakit.feedback import NeighbourhoodAgreementFeedback
from cakit.visualization import SpaceTimePlot


def main():
    print("=" * 60)
    print("Cakit Framework Demo: Binary CLA with Tsetlin Automata")
    print("=" * 60)
    
    # Create a 1D grid with random initial states
    grid = Grid1D(
        size=51,
        radius=1,
        boundary='periodic',
        initial_states='random'
    )
    print(f"\nGrid: {grid.size} cells, radius={grid.radius}, boundary='{grid.boundary}'")
    print("Initial state: Random")
    
    # Create feedback function
    feedback_fn = NeighbourhoodAgreementFeedback(radius=1)
    print(f"\nFeedback: NeighbourhoodAgreementFeedback(radius=1)")
    print("  - Rewards cells matching neighborhood majority")
    
    # Create a StateSystem with TsetlinAutomaton
    system = StateSystem(
        grid=grid,
        automaton_type=TsetlinAutomaton,
        automaton_params={'n_states': 10, 'initial_state': 'random'},
        feedback_fn=feedback_fn,
        feedback_radius=1,
        seed=42
    )
    print(f"\nSystem: StateSystem with TsetlinAutomaton(n_states=10)")
    print("Running for 100 generations...")
    
    # Run for 100 generations
    grid_history, ta_history = system.run(generations=100)
    
    print(f"\nCompleted! History shapes:")
    print(f"  - Grid:      {grid_history.shape}")
    print(f"  - Automata:  {ta_history.shape}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plotter = SpaceTimePlot(
        grid_history=grid_history,
        ta_state_history=ta_history,
        n_states=10,
        figsize=(14, 10)
    )
    
    # Save standard plot
    save_path_std = "cla_spacetime_standard.png"
    plotter.plot_standard(save_path=save_path_std, show=False)
    print(f"✓ Saved standard space-time diagram to: {save_path_std}")
    
    # Save TA-augmented plot
    save_path_aug = "cla_spacetime_ta_augmented.png"
    plotter.plot_ta_augmented(save_path=save_path_aug, show=False)
    print(f"✓ Saved TA-augmented space-time diagram to: {save_path_aug}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
