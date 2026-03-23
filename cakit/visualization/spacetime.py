"""
Space-time plot visualization for cellular automata.

Produces space-time diagrams from history arrays, including a special TA-augmented
plot that encodes both cell state and Tsetlin Automaton confidence.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class SpaceTimePlot:
    """
    Space-time plot visualizer for cellular automata systems.
    
    Produces space-time diagrams where rows represent time steps and columns represent
    cell positions. Supports both standard binary plots and TA-augmented plots that
    encode automaton confidence.
    
    Parameters:
        grid_history: Cell states for all generations. Shape (T, G) where T = generations,
                     G = grid size.
        ta_state_history: Optional TA internal states. Shape (T, G). Required for
                         TA-augmented plots.
        n_states: Number of states per arm in Tsetlin Automaton (N). Used to normalize
                 shade intensity in augmented plots.
        cell_size: Pixel size per cell in the output. Default: 5.
        figsize: Matplotlib figure size. Default: (12, 8).
    """
    
    def __init__(
        self,
        grid_history: np.ndarray,
        ta_state_history: Optional[np.ndarray] = None,
        n_states: int = 5,
        cell_size: int = 5,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize a SpaceTimePlot.
        
        Args:
            grid_history: Array of shape (T, G) with cell states.
            ta_state_history: Optional array of shape (T, G) with TA states.
            n_states: States per arm in TA (N).
            cell_size: Pixel size per cell.
            figsize: Figure size tuple.
        """
        self.grid_history = grid_history
        self.ta_state_history = ta_state_history
        self.n_states = n_states
        self.cell_size = cell_size
        self.figsize = figsize
        
        self.generations, self.grid_size = grid_history.shape
    
    def plot_standard(
        self, 
        save_path: Optional[str] = None, 
        show: bool = True,
        params: Optional[dict] = None
    ) -> None:
        """
        Plot a standard binary space-time diagram.
        
        Black = state 1 (alive), white = state 0 (dead).
        
        Args:
            save_path: Optional path to save the figure. If None, not saved.
            show: Whether to display the plot. Default: True.
            params: Optional dict of experiment parameters to display below plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot with binary colormap: 0=white, 1=black
        im = ax.imshow(
            self.grid_history,
            cmap='gray_r',
            interpolation='nearest',
            aspect='auto',
            vmin=0,
            vmax=1
        )
        
        ax.set_xlabel('Cell Position', fontsize=12)
        ax.set_ylabel('Generation', fontsize=12)
        ax.set_title('Space-Time Diagram', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cell State', rotation=270, labelpad=20)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Dead (0)', 'Alive (1)'])
        
        plt.tight_layout()
        
        # Add parameter legend if provided
        if params:
            param_text = self._format_params(params)
            fig.text(
                0.5, -0.02, param_text,
                ha='center', va='top',
                fontsize=10,
                family='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.8',
                    facecolor='lightgray',
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.9
                )
            )
            # Adjust layout to make room for legend
            plt.subplots_adjust(bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_ta_augmented(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        params: Optional[dict] = None
    ) -> None:
        """
        Plot a TA-augmented space-time diagram.
        
        Encodes both cell state and TA confidence in a single color:
        - Dead cell (state=0): Blue ramp. Darker = deeply committed to action 0.
        - Live cell (state=1): Red ramp. Darker = deeply committed to action 1.
        
        Shade intensity is based on position within the arm: position / (N-1).
        
        Args:
            save_path: Optional path to save the figure.
            show: Whether to display the plot.
            params: Optional dict of experiment parameters to display below plot.
        
        Raises:
            ValueError: If ta_state_history is None.
        """
        if self.ta_state_history is None:
            raise ValueError(
                "TA-augmented plot requires ta_state_history. "
                "Provide it during initialization."
            )
        
        # Build the augmented image
        augmented = self._build_ta_augmented_image()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Display the augmented image
        im = ax.imshow(
            augmented,
            interpolation='nearest',
            aspect='auto'
        )
        
        ax.set_xlabel('Cell Position', fontsize=12)
        ax.set_ylabel('Generation', fontsize=12)
        ax.set_title(
            'TA-Augmented Space-Time Diagram',
            fontsize=14,
            fontweight='bold'
        )
        
        # Create a custom colormap for the colorbar that matches our red/blue scheme
        # Bottom = dark blue (high confidence dead), top = dark red (high confidence alive)
        # Middle = light blue meets light red (low confidence)
        colors_list = []
        n_steps = 128
        
        # Blue ramp (dead cells): dark blue at bottom -> light blue at top (reversed)
        for i in range(n_steps - 1, -1, -1):
            intensity = i / (n_steps - 1)
            colors_list.append(self._blue_ramp(intensity))
        
        # Red ramp (alive cells): light red at bottom -> dark red at top
        for i in range(n_steps):
            intensity = i / (n_steps - 1)
            colors_list.append(self._red_ramp(intensity))
        
        cmap_custom = LinearSegmentedColormap.from_list('ta_confidence', colors_list, N=256)
        
        # Create a scalar mappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        # Add colorbar with custom colormap
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('State & Confidence', rotation=270, labelpad=20)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Dead (0)\nLow → High Confidence', 'Alive (1)\nLow → High Confidence'])
        cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        
        # Add parameter legend if provided
        if params:
            param_text = self._format_params(params)
            fig.text(
                0.5, -0.02, param_text,
                ha='center', va='top',
                fontsize=10,
                family='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.8',
                    facecolor='lightgray',
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.9
                )
            )
            # Adjust layout to make room for legend
            plt.subplots_adjust(bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _build_ta_augmented_image(self) -> np.ndarray:
        """
        Build the TA-augmented image array.
        
        Returns:
            RGB image array of shape (T, G, 3) with values in [0, 1].
        """
        image = np.zeros((self.generations, self.grid_size, 3))
        
        for t in range(self.generations):
            for g in range(self.grid_size):
                cell_state = self.grid_history[t, g]
                ta_state = self.ta_state_history[t, g]
                
                # Calculate arm and position
                arm = self._get_arm(ta_state)
                position = self._get_position(ta_state)
                
                # Normalize position to [0, 1]
                if self.n_states == 1:
                    intensity = 1.0
                else:
                    intensity = position / (self.n_states - 1)
                
                # Apply color based on cell state and TA confidence
                if cell_state == 0:
                    # Dead cell: Blue ramp
                    # Light blue (low confidence) to dark blue (high confidence)
                    image[t, g] = self._blue_ramp(intensity)
                else:
                    # Live cell: Red ramp
                    # Light red (low confidence) to dark red (high confidence)
                    image[t, g] = self._red_ramp(intensity)
        
        return image
    
    def _get_arm(self, ta_state: int) -> int:
        """
        Get the arm (action) from TA state.
        
        Args:
            ta_state: TA state in {1, ..., 2N}.
        
        Returns:
            Arm index (0 or 1).
        """
        return (ta_state - 1) // self.n_states
    
    def _get_position(self, ta_state: int) -> int:
        """
        Get the position within the arm from TA state.
        
        Args:
            ta_state: TA state in {1, ..., 2N}.
        
        Returns:
            Position in {0, ..., N-1}.
        """
        return (ta_state - 1) % self.n_states
    
    def _blue_ramp(self, intensity: float) -> np.ndarray:
        """
        Generate a blue color with intensity-based darkness.
        
        Args:
            intensity: Value in [0, 1]. 0 = light blue, 1 = dark blue.
        
        Returns:
            RGB array [r, g, b] with values in [0, 1].
        """
        # Light blue: (0.6, 0.8, 1.0) -> Dark blue: (0.0, 0.0, 0.5)
        r = 0.6 * (1 - intensity)
        g = 0.8 * (1 - intensity)
        b = 1.0 - 0.5 * intensity
        return np.array([r, g, b])
    
    def _red_ramp(self, intensity: float) -> np.ndarray:
        """
        Generate a red color with intensity-based darkness.
        
        Args:
            intensity: Value in [0, 1]. 0 = light red, 1 = dark red.
        
        Returns:
            RGB array [r, g, b] with values in [0, 1].
        """
        # Light red: (1.0, 0.6, 0.6) -> Dark red: (0.5, 0.0, 0.0)
        r = 1.0 - 0.5 * intensity
        g = 0.6 * (1 - intensity)
        b = 0.6 * (1 - intensity)
        return np.array([r, g, b])
    
    def _format_params(self, params: dict) -> str:
        """
        Format parameter dictionary into a readable multi-column string.
        
        Args:
            params: Dictionary of experiment parameters.
        
        Returns:
            Formatted string for display in columns.
        """
        items = [f"{key}: {value}" for key, value in params.items()]
        
        # Organize into columns (3 params per line)
        lines = []
        for i in range(0, len(items), 3):
            line_items = items[i:i+3]
            lines.append("    ".join(line_items))
        
        return "\n".join(lines)
