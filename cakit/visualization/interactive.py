"""
PyCX-style interactive GUI for cellular automata visualization.

Tries to use a Tkinter window (native OS buttons, most reliable).
Falls back to a pure-matplotlib window (macosx / Qt) if tkinter is absent.

Controls: Start/Pause · Step · Reset · Save · [View: B&W / Color — CLA only]
"""

import os
from typing import Optional, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..system.base import AutomataSystem


class InteractiveGUI:
    """
    PyCX-style interactive GUI for cellular automata experiments.

    Displays a space-time diagram that grows row by row as the simulation
    runs.  Controls: Start/Pause, Step, Reset, Save.  When paused, a scroll
    slider (Tk path) or the plot y-axis (fallback path) lets you revisit
    earlier generations.

    Layout
    ------
    A params card sits above the diagram showing all experiment parameters.
    A legend strip on the right explains the colour encoding:
      - CA mode: white = 0 (OFF), black = 1 (ON)
      - CLA augmented: gradient bars for state 0 (blue) and state 1 (red),
        dark = confident (deep in arm), light = uncertain (near boundary)
      - CLA standard (B&W toggle): same as CA mode

    For CLA experiments (n_states provided) an additional View toggle button
    switches between the B&W binary view and the TA-confidence colour view
    without restarting the simulation — both buffers are captured every step.

    Parameters
    ----------
    system : AutomataSystem
        An initialised system (StateSystem, etc.).
    max_generations : int
        Maximum number of steps that will be recorded.
    plot_type : str
        Starting view: ``'standard'`` (binary gray) or ``'augmented'``
        (TA confidence colours).  Ignored for pure CA (no n_states).
    n_states : int, optional
        States per TA arm — required for the augmented view.  When provided
        the GUI operates in CLA mode and adds a View toggle button.
    params : dict, optional
        Experiment parameters shown in the params card.
    figsize : tuple
        Matplotlib figure size in inches.
    interval : int
        Milliseconds between automatic steps when running.
    view_window : int
        Rows visible at once.
    save_dir : str, optional
        Directory where saved images are written.  Defaults to cwd.
    """

    def __init__(
        self,
        system: AutomataSystem,
        max_generations: int = 200,
        plot_type: str = 'standard',
        n_states: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        figsize=(14, 7),
        interval: int = 80,
        view_window: int = 80,
        save_dir: Optional[str] = None,
    ):
        self.system = system
        self.max_generations = max_generations
        self.n_states = n_states
        self._cla_mode = n_states is not None
        self.plot_type = plot_type if self._cla_mode else 'standard'
        self.params = params or {}
        self.figsize = figsize
        self.interval = interval
        self._view_window = view_window
        self._save_dir = save_dir

        self._running = False
        self._current_gen = 0
        self._grid_size = len(list(system.grid.get_positions()))

        self._alloc_buffers()
        self._capture_row(0)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _alloc_buffers(self) -> None:
        self._buf_std = np.zeros(
            (self.max_generations + 1, self._grid_size), dtype=np.float32)
        if self._cla_mode:
            self._buf_aug = np.zeros(
                (self.max_generations + 1, self._grid_size, 3), dtype=np.float32)
        self._display = (
            self._buf_aug
            if self._cla_mode and self.plot_type == 'augmented'
            else self._buf_std
        )

    def _capture_row(self, gen: int) -> None:
        states = self.system.grid.get_all_states()
        self._buf_std[gen] = states.astype(np.float32)
        if self._cla_mode:
            for i in range(self._grid_size):
                cs = states[i]
                ta = self.system.automata[i]
                p  = ta.get_position()
                intensity = p / (self.n_states - 1) if self.n_states > 1 else 1.0
                if cs == 0:
                    self._buf_aug[gen, i] = [
                        0.6 * (1 - intensity),
                        0.8 * (1 - intensity),
                        1.0 - 0.5 * intensity,
                    ]
                else:
                    self._buf_aug[gen, i] = [
                        1.0 - 0.5 * intensity,
                        0.6 * (1 - intensity),
                        0.6 * (1 - intensity),
                    ]

    # ------------------------------------------------------------------
    # Layout helpers — params card and legend
    # ------------------------------------------------------------------

    def _draw_params_card(self, card_ax) -> None:
        """Fill the params card axes with experiment parameters."""
        card_ax.clear()
        card_ax.set_xlim(0, 1)
        card_ax.set_ylim(0, 1)

        # Visible card: white background, coloured border, no tick marks
        card_ax.set_facecolor('#ffffff')
        card_ax.set_xticks([])
        card_ax.set_yticks([])
        for spine in card_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor('#5c6bc0')

        if not self.params:
            return

        items = list(self.params.items())
        n = len(items)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        for idx, (k, v) in enumerate(items):
            col = idx % cols
            row = idx // cols
            x = (col + 0.5) / cols
            y = 1.0 - (row + 0.5) / rows
            card_ax.text(
                x, y, f'{k}:  {v}',
                ha='center', va='center',
                fontsize=8.5, fontfamily='monospace',
                transform=card_ax.transAxes,
            )

    def _draw_legend(self, leg_ax) -> None:
        """Fill the legend axes based on the current plot_type."""
        leg_ax.clear()
        leg_ax.set_xlim(0, 1)
        leg_ax.set_ylim(0, 1)
        leg_ax.set_axis_off()
        leg_ax.set_facecolor('#f5f5f5')

        leg_ax.text(0.5, 0.975, 'Legend',
                    ha='center', va='top', fontsize=8, fontweight='bold')

        if self.plot_type == 'augmented' and self._cla_mode:
            self._draw_legend_augmented(leg_ax)
        else:
            self._draw_legend_standard(leg_ax)

    def _draw_legend_augmented(self, leg_ax) -> None:
        """
        Single discrete-step bar representing every TA state level.

        Layout (top → bottom):
          State 1 (ON)  — n_states blocks, dark red (confident) → light red (uncertain)
          ── boundary ──
          State 0 (OFF) — n_states blocks, light blue (uncertain) → dark blue (confident)
        """
        n = self.n_states
        bar_x   = 0.18
        bar_w   = 0.45
        bar_top = 0.88
        bar_bot = 0.06
        block_h = (bar_top - bar_bot) / (2 * n)
        mid_y   = bar_bot + n * block_h

        # State 1 blocks — top half, confident (dark red) at top → uncertain (light) at mid
        for i in range(n):
            intensity = (n - 1 - i) / max(n - 1, 1)  # i=0: confident, i=n-1: uncertain
            color = [1.0 - 0.5*intensity, 0.6*(1-intensity), 0.6*(1-intensity)]
            y = bar_top - (i + 1) * block_h
            leg_ax.add_patch(mpatches.Rectangle(
                (bar_x, y), bar_w, block_h,
                facecolor=color, edgecolor='none'))

        # State 0 blocks — bottom half, uncertain (light) at mid → confident (dark blue) at bot
        for i in range(n):
            intensity = i / max(n - 1, 1)  # i=0: uncertain, i=n-1: confident
            color = [0.6*(1-intensity), 0.8*(1-intensity), 1.0 - 0.5*intensity]
            y = mid_y - (i + 1) * block_h
            leg_ax.add_patch(mpatches.Rectangle(
                (bar_x, y), bar_w, block_h,
                facecolor=color, edgecolor='none'))

        # Border around entire bar
        leg_ax.add_patch(mpatches.Rectangle(
            (bar_x, bar_bot), bar_w, bar_top - bar_bot,
            facecolor='none', edgecolor='#666666', linewidth=0.8))

        # Dashed separator at the boundary between arms
        leg_ax.plot([bar_x, bar_x + bar_w], [mid_y, mid_y],
                    color='#bbbbbb', linewidth=1.2, linestyle='--')

        rx = bar_x + bar_w  # right edge of bar

        # Side tick marks + confidence labels
        for y_pos, label in [(bar_top, 'confident'), (mid_y, 'uncertain'), (bar_bot, 'confident')]:
            leg_ax.plot([rx, rx + 0.06], [y_pos, y_pos],
                        color='#888888', linewidth=0.8)
            leg_ax.text(rx + 0.08, y_pos, label,
                        ha='left', va='center', fontsize=6, color='#555555')

        # State labels inside each half (with semi-transparent background)
        _tkw = dict(ha='center', va='center', fontsize=7.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.55, edgecolor='none'))
        leg_ax.text(bar_x + bar_w / 2, (bar_top + mid_y) / 2, '1\n(ON)',  **_tkw)
        leg_ax.text(bar_x + bar_w / 2, (mid_y + bar_bot) / 2, '0\n(OFF)', **_tkw)

    def _draw_legend_standard(self, leg_ax) -> None:
        """Simple white / black patches for B&W view."""
        pw, ph = 0.55, 0.18
        x0 = (1 - pw) / 2

        # 0 = OFF = white
        leg_ax.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.56), pw, ph,
            boxstyle='round,pad=0.01',
            facecolor='white', edgecolor='black', linewidth=1))
        leg_ax.text(0.5, 0.56 + ph / 2, '0  —  OFF',
                    ha='center', va='center', fontsize=8)

        # 1 = ON = black
        leg_ax.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.26), pw, ph,
            boxstyle='round,pad=0.01',
            facecolor='black', edgecolor='black', linewidth=1))
        leg_ax.text(0.5, 0.26 + ph / 2, '1  —  ON',
                    ha='center', va='center', fontsize=8, color='white')

    # ------------------------------------------------------------------
    # View toggle (shared by both backends)
    # ------------------------------------------------------------------

    def _switch_view(self, im, redraw_fn, leg_ax=None) -> str:
        """
        Toggle between standard and augmented views.  Updates self._display,
        self.plot_type, and the live imshow + legend.

        Returns the new button label.
        """
        n = self._current_gen + 1
        if self.plot_type == 'augmented':
            self.plot_type = 'standard'
            self._display = self._buf_std
            im.set_data(self._display[:n])
            im.set_cmap('gray_r')
            im.set_clim(0, 1)
        else:
            self.plot_type = 'augmented'
            self._display = self._buf_aug
            im.set_data(self._display[:n])
            im.set_clim(0, 1)
        if leg_ax is not None:
            self._draw_legend(leg_ax)
        redraw_fn()
        return 'View: B&W' if self.plot_type == 'augmented' else 'View: Color'

    # ------------------------------------------------------------------
    # Filename / title helpers
    # ------------------------------------------------------------------

    def _default_filename(self, suffix: str = '') -> str:
        import re
        parts = []
        for k, v in self.params.items():
            key = re.sub(r'\s+', '', str(k)).lower()
            val = re.sub(r'[^a-z0-9]', '', str(v).lower())
            parts.append(f'{key}{val}')
        parts.append(f'gen{self._current_gen}')
        name = 'spacetime_' + '_'.join(parts)
        if suffix:
            name += f'_{suffix}'
        return name + '.png'

    def _build_title(self) -> str:
        """Short axes title — params live in the card, not here."""
        return ('TA-Augmented Space-Time Diagram'
                if self.plot_type == 'augmented' else 'Space-Time Diagram')

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Open the interactive window.  Prefers Tkinter; falls back to macosx/Qt."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            matplotlib.use('TkAgg')
            self._show_tk(tk, filedialog, FigureCanvasTkAgg)
        except (ImportError, ModuleNotFoundError):
            print(
                '[InteractiveGUI] tkinter not found — using matplotlib fallback.\n'
                '  To enable native buttons install tkinter:\n'
                '    brew install python-tk@3.12'
            )
            self._show_mpl()

    # ------------------------------------------------------------------
    # Path A: Tkinter window
    # ------------------------------------------------------------------

    def _show_tk(self, tk, filedialog, FigureCanvasTkAgg) -> None:
        root = tk.Tk()
        root.title('CA / CLA Interactive Simulation')
        root.configure(bg='#efefef')

        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_facecolor('#efefef')

        # ── axes layout (Tk: buttons are outside the figure) ─────────
        # params card — narrower than the diagram, centred above it
        ax_card   = fig.add_axes([0.11, 0.88, 0.66, 0.09])
        # main plot — leaves gap on the right for legend
        ax        = fig.add_axes([0.07, 0.05, 0.74, 0.80])
        # legend strip on the right
        ax_legend = fig.add_axes([0.83, 0.05, 0.15, 0.80])

        self._draw_params_card(ax_card)
        self._draw_legend(ax_legend)

        # ── canvas + scroll slider ────────────────────────────────────
        frame_top = tk.Frame(root, bg='#efefef')
        frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mpl_canvas = FigureCanvasTkAgg(fig, master=frame_top)
        mpl_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_var = tk.IntVar(value=0)
        scrollbar = tk.Scale(
            frame_top, from_=0, to=0,
            orient=tk.VERTICAL, variable=scroll_var,
            label='Scroll', length=400, sliderlength=24,
            bg='#efefef', troughcolor='#cccccc',
            state=tk.DISABLED,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 6), pady=6)

        # ── buttons ───────────────────────────────────────────────────
        frame_btns = tk.Frame(root, bg='#efefef', pady=6)
        frame_btns.pack(side=tk.BOTTOM, fill=tk.X)

        _bs = dict(width=10, height=1,
                   font=('Helvetica', 12, 'bold'),
                   relief=tk.RAISED, bd=2, cursor='hand2')

        btn_start = tk.Button(frame_btns, text='Start',
                              bg='#a5d6a7', activebackground='#81c784', **_bs)
        btn_step  = tk.Button(frame_btns, text='Step',
                              bg='#90caf9', activebackground='#64b5f6', **_bs)
        btn_reset = tk.Button(frame_btns, text='Reset',
                              bg='#ef9a9a', activebackground='#e57373', **_bs)
        btn_save  = tk.Button(frame_btns, text='Save',
                              bg='#fff176', activebackground='#ffee58', **_bs)

        for b in (btn_start, btn_step, btn_reset, btn_save):
            b.pack(side=tk.LEFT, padx=16, pady=2)

        if self._cla_mode:
            _view_init = 'View: B&W' if self.plot_type == 'augmented' else 'View: Color'
            btn_view = tk.Button(frame_btns, text=_view_init,
                                 bg='#ce93d8', activebackground='#ba68c8', **_bs)
            btn_view.pack(side=tk.LEFT, padx=16, pady=2)

        # ── initial plot ──────────────────────────────────────────────
        _imkw = dict(aspect='auto', interpolation='nearest', origin='upper')
        if self.plot_type == 'standard':
            _imkw.update(cmap='gray_r', vmin=0, vmax=1)

        im = ax.imshow(self._display[:1], **_imkw)
        ax.set_xlabel('Cell Position', fontsize=11)
        ax.set_ylabel('Generation', fontsize=11)
        ax.set_xlim(-0.5, self._grid_size - 0.5)
        ax.set_ylim(self._view_window - 0.5, -0.5)
        ax.set_title(f'{self._build_title()}    [Gen 0]', fontsize=10)

        def _redraw():
            ax.set_title(
                f'{self._build_title()}    [Gen {self._current_gen}]', fontsize=10)
            mpl_canvas.draw_idle()

        def _refresh(view_top: Optional[int] = None) -> None:
            n = self._current_gen + 1
            im.set_data(self._display[:n])
            im.set_extent([-0.5, self._grid_size - 0.5, n - 0.5, -0.5])
            if view_top is None:
                view_top = max(0, n - self._view_window)
            ax.set_ylim(view_top + self._view_window - 0.5, view_top - 0.5)
            _redraw()

        def _on_scroll(val):
            if not self._running:
                _refresh(view_top=int(val))

        scrollbar.config(command=_on_scroll)

        def _do_step() -> bool:
            if self._current_gen >= self.max_generations:
                return False
            self.system.step()
            self._current_gen += 1
            self._capture_row(self._current_gen)
            n = self._current_gen + 1
            max_offset = max(0, n - self._view_window)
            scrollbar.config(to=max_offset)
            scrollbar.set(max_offset)
            _refresh()
            return True

        _timer_id: list = [None]

        def _tick():
            if self._running:
                if not _do_step():
                    _set_paused()
                    return
                _timer_id[0] = root.after(self.interval, _tick)

        def _set_running():
            self._running = True
            btn_start.config(text='Pause')
            scrollbar.config(state=tk.DISABLED)
            _timer_id[0] = root.after(self.interval, _tick)

        def _set_paused():
            self._running = False
            btn_start.config(text='Start')
            n = self._current_gen + 1
            scrollbar.config(state=tk.NORMAL, to=max(0, n - self._view_window))

        def on_start():
            if self._running:
                if _timer_id[0] is not None:
                    root.after_cancel(_timer_id[0])
                    _timer_id[0] = None
                _set_paused()
            else:
                _set_running()

        def on_step():
            if not self._running:
                _do_step()

        def on_reset():
            if _timer_id[0] is not None:
                root.after_cancel(_timer_id[0])
                _timer_id[0] = None
            self._running = False
            btn_start.config(text='Start')
            self.system.reset()
            self._current_gen = 0
            self._alloc_buffers()
            self._capture_row(0)
            if self.plot_type == 'standard':
                im.set_cmap('gray_r')
                im.set_clim(0, 1)
            scrollbar.config(state=tk.DISABLED, to=0)
            scrollbar.set(0)
            _refresh(view_top=0)

        def on_save():
            _title = ('Export space-time diagrams (both views will be saved)'
                      if self._cla_mode else 'Export space-time diagram')
            path = filedialog.asksaveasfilename(
                parent=root,
                title=_title,
                initialdir=self._save_dir or os.getcwd(),
                initialfile=self._default_filename(),
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'),
                           ('PDF document', '*.pdf'),
                           ('SVG vector', '*.svg')],
            )
            if not path:
                return
            self._export_all(path)

        btn_start.config(command=on_start)
        btn_step .config(command=on_step)
        btn_reset.config(command=on_reset)
        btn_save .config(command=on_save)

        if self._cla_mode:
            def on_toggle_view():
                new_label = self._switch_view(im, _redraw, leg_ax=ax_legend)
                btn_view.config(text=new_label)
            btn_view.config(command=on_toggle_view)

        mpl_canvas.draw()
        root.mainloop()

    # ------------------------------------------------------------------
    # Path B: pure-matplotlib fallback
    # ------------------------------------------------------------------

    def _show_mpl(self) -> None:
        for backend in ('macosx', 'QtAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg'):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

        from matplotlib.widgets import Button as MplButton

        fig = plt.figure(figsize=self.figsize)

        # ── axes layout (MPL: buttons live inside the figure at bottom) ──
        # params card — narrower than the diagram, centred above it
        ax_card   = fig.add_axes([0.11, 0.88, 0.66, 0.09])
        ax        = fig.add_axes([0.07, 0.17, 0.74, 0.68])
        ax_legend = fig.add_axes([0.83, 0.17, 0.15, 0.68])

        self._draw_params_card(ax_card)
        self._draw_legend(ax_legend)

        # ── buttons ───────────────────────────────────────────────────
        if self._cla_mode:
            ax_start = fig.add_axes([0.10, 0.03, 0.12, 0.09])
            ax_step  = fig.add_axes([0.25, 0.03, 0.12, 0.09])
            ax_reset = fig.add_axes([0.40, 0.03, 0.12, 0.09])
            ax_save  = fig.add_axes([0.55, 0.03, 0.12, 0.09])
            ax_view  = fig.add_axes([0.70, 0.03, 0.12, 0.09])
        else:
            ax_start = fig.add_axes([0.20, 0.03, 0.13, 0.09])
            ax_step  = fig.add_axes([0.37, 0.03, 0.13, 0.09])
            ax_reset = fig.add_axes([0.54, 0.03, 0.13, 0.09])
            ax_save  = fig.add_axes([0.71, 0.03, 0.13, 0.09])

        btn_start = MplButton(ax_start, 'Start', color='#a5d6a7', hovercolor='#81c784')
        btn_step  = MplButton(ax_step,  'Step',  color='#90caf9', hovercolor='#64b5f6')
        btn_reset = MplButton(ax_reset, 'Reset', color='#ef9a9a', hovercolor='#e57373')
        btn_save  = MplButton(ax_save,  'Save',  color='#fff176', hovercolor='#ffee58')

        for b in (btn_start, btn_step, btn_reset, btn_save):
            b.label.set_fontsize(11)

        if self._cla_mode:
            _view_init = 'View: B&W' if self.plot_type == 'augmented' else 'View: Color'
            btn_view = MplButton(ax_view, _view_init,
                                 color='#ce93d8', hovercolor='#ba68c8')
            btn_view.label.set_fontsize(11)

        # ── initial plot ──────────────────────────────────────────────
        _imkw = dict(aspect='auto', interpolation='nearest', origin='upper')
        if self.plot_type == 'standard':
            _imkw.update(cmap='gray_r', vmin=0, vmax=1)

        im = ax.imshow(self._display[:1], **_imkw)
        ax.set_xlabel('Cell Position', fontsize=11)
        ax.set_ylabel('Generation', fontsize=11)
        ax.set_xlim(-0.5, self._grid_size - 0.5)
        ax.set_ylim(self._view_window - 0.5, -0.5)
        ax.set_title(f'{self._build_title()}    [Gen 0]', fontsize=10)

        def _redraw():
            ax.set_title(
                f'{self._build_title()}    [Gen {self._current_gen}]', fontsize=10)
            fig.canvas.draw_idle()

        def _refresh() -> None:
            n = self._current_gen + 1
            im.set_data(self._display[:n])
            im.set_extent([-0.5, self._grid_size - 0.5, n - 0.5, -0.5])
            view_top = max(0, n - self._view_window)
            ax.set_ylim(view_top + self._view_window - 0.5, view_top - 0.5)
            _redraw()

        def _do_step() -> bool:
            if self._current_gen >= self.max_generations:
                return False
            self.system.step()
            self._current_gen += 1
            self._capture_row(self._current_gen)
            _refresh()
            return True

        timer = fig.canvas.new_timer(interval=self.interval)

        def _on_timer():
            if self._running:
                if not _do_step():
                    self._running = False
                    timer.stop()
                    btn_start.label.set_text('Start')
                    fig.canvas.draw_idle()

        timer.add_callback(_on_timer)

        def on_start(_e):
            self._running = not self._running
            if self._running:
                btn_start.label.set_text('Pause')
                timer.start()
            else:
                btn_start.label.set_text('Start')
                timer.stop()

        def on_step(_e):
            if not self._running:
                _do_step()

        def on_reset(_e):
            self._running = False
            timer.stop()
            btn_start.label.set_text('Start')
            self.system.reset()
            self._current_gen = 0
            self._alloc_buffers()
            self._capture_row(0)
            if self.plot_type == 'standard':
                im.set_cmap('gray_r')
                im.set_clim(0, 1)
            _refresh()

        def on_save(_e):
            base = self._save_dir or os.getcwd()
            path = os.path.join(base, self._default_filename())
            self._export_all(path)

        def on_close(_e):
            os._exit(0)

        fig.canvas.mpl_connect('close_event', on_close)

        btn_start.on_clicked(on_start)
        btn_step .on_clicked(on_step)
        btn_reset.on_clicked(on_reset)
        btn_save .on_clicked(on_save)

        if self._cla_mode:
            def on_toggle_view(_e):
                new_label = self._switch_view(im, _redraw, leg_ax=ax_legend)
                btn_view.label.set_text(new_label)
                fig.canvas.draw_idle()
            btn_view.on_clicked(on_toggle_view)

        plt.show()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export(self, path: str,
                buf: Optional[np.ndarray] = None,
                plot_type: Optional[str] = None) -> None:
        """
        Save a full-layout image (params card + space-time diagram + legend)
        to *path* using a standalone Agg figure.

        Parameters
        ----------
        path : str
            Output file path.
        buf : ndarray, optional
            Buffer to render.  Defaults to self._display (active view).
        plot_type : str, optional
            ``'standard'`` or ``'augmented'``.  Defaults to self.plot_type.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        buf       = buf       if buf       is not None else self._display
        plot_type = plot_type if plot_type is not None else self.plot_type

        n   = self._current_gen + 1
        dpi = 150
        px  = 4  # pixels per cell / generation

        # ── figure dimensions (inches) ───────────────────────────────
        diag_w = max(4.0, self._grid_size * px / dpi)
        diag_h = max(3.0, n               * px / dpi)
        leg_w  = 1.6   # legend strip
        card_h = 0.65  # params card height
        # fixed margins
        ml, mr, mb, mt = 0.70, 0.10, 0.40, 0.20
        gap_cl = 0.12  # gap between card and diagram
        gap_dl = 0.10  # gap between diagram and legend

        total_w = ml + diag_w + gap_dl + leg_w + mr
        total_h = mt + card_h + gap_cl + diag_h + mb

        sfig = Figure(figsize=(total_w, total_h), dpi=dpi)
        FigureCanvasAgg(sfig)

        def _norm(x, w, y, h):
            """Convert inch coords to normalised [0,1] figure coords."""
            return [x / total_w, y / total_h, w / total_w, h / total_h]

        # diagram axes
        sax = sfig.add_axes(_norm(ml, diag_w, mb, diag_h))

        # legend axes (right of diagram)
        leg_ax = sfig.add_axes(
            _norm(ml + diag_w + gap_dl, leg_w - mr, mb, diag_h))

        # params card (narrower than diagram, centred above it)
        card_inset = 0.10
        card_ax = sfig.add_axes(
            _norm(ml + card_inset, diag_w - 2*card_inset,
                  mb + diag_h + gap_cl, card_h))

        # ── draw content ─────────────────────────────────────────────
        _imkw = dict(aspect='auto', interpolation='nearest', origin='upper')
        if plot_type == 'standard':
            _imkw.update(cmap='gray_r', vmin=0, vmax=1)

        sax.imshow(buf[:n], **_imkw)
        sax.set_xlabel('Cell Position', fontsize=9)
        sax.set_ylabel('Generation',    fontsize=9)
        title = ('TA-Augmented Space-Time Diagram'
                 if plot_type == 'augmented' else 'Space-Time Diagram')
        sax.set_title(title, fontsize=9)

        # Temporarily set plot_type so legend/card helpers render correctly
        _saved = self.plot_type
        self.plot_type = plot_type
        self._draw_legend(leg_ax)
        self._draw_params_card(card_ax)
        self.plot_type = _saved

        sfig.savefig(path, dpi=dpi)
        print(f'Saved → {path}')

    def _export_all(self, base_path: str) -> None:
        """
        In CLA mode save both views (_bw and _color) as separate files.
        In CA mode behaves identically to _export(base_path).
        """
        if not self._cla_mode:
            self._export(base_path)
            return

        root_no_ext, ext = os.path.splitext(base_path)
        if not ext:
            ext = '.png'

        self._export(root_no_ext + '_bw'    + ext,
                     buf=self._buf_std, plot_type='standard')
        self._export(root_no_ext + '_color' + ext,
                     buf=self._buf_aug, plot_type='augmented')
