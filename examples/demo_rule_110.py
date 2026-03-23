"""
Demo script: Rule 110 Elementary CA

This script demonstrates the cakit framework by running a classic Rule 110
cellular automaton and visualizing the space-time diagram.
"""

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from cakit.automata import WolframAutomaton
from cakit.grid import Grid1D
from cakit.system import StateSystem
from cakit.visualization import SpaceTimePlot


def main():
    print("=" * 60)
    print("Cakit Framework Demo: Rule 110 Elementary CA")
    print("=" * 60)
    
    # Create a 1D grid with single center cell activated
    grid = Grid1D(
        size=101,
        radius=1,
        boundary='zero',
        initial_states='single_center'
    )
    print(f"\nGrid: {grid.size} cells, radius={grid.radius}, boundary='{grid.boundary}'")
    print(f"Initial state: Single center cell (position {grid.size // 2})")
    
    # Create a StateSystem with WolframAutomaton (Rule 110)
    system = StateSystem(
        grid=grid,
        automaton_type=WolframAutomaton,
        automaton_params={'rule': 110},
        feedback_fn=None,  # Pure CA, no learning
        seed=42
    )
    print(f"\nSystem: StateSystem with WolframAutomaton(rule=110)")
    print("Running for 50 generations...")
    
    # Run for 50 generations
    grid_history, automaton_history = system.run(generations=50)
    
    print(f"\nCompleted! History shape: {grid_history.shape}")
    print(f"  - {grid_history.shape[0]} time steps (initial + 50 generations)")
    print(f"  - {grid_history.shape[1]} cells")
    
    # Create visualization
    print("\nGenerating space-time plot...")
    plotter = SpaceTimePlot(
        grid_history=grid_history,
        figsize=(14, 10)
    )
    
    # Save standard plot
    save_path = "rule_110_spacetime.png"
    plotter.plot_standard(save_path=save_path, show=False)
    print(f"✓ Saved space-time diagram to: {save_path}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
