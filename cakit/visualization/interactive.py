"""
PyCX-style interactive GUI for cellular automata visualization.

Tries to use a Tkinter window (native OS buttons, most reliable).
Falls back to a pure-matplotlib window (macosx / Qt) if tkinter is absent.

Controls: Start/Pause · Step · Reset · Save
"""

from typing import Optional, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..system.base import AutomataSystem


class InteractiveGUI:
    """
    PyCX-style interactive GUI for cellular automata experiments.

    Displays a space-time diagram that grows row by row as the simulation
    runs.  Controls: Start/Pause, Step, Reset, Save.  When paused, a scroll
    slider (Tk path) or the plot y-axis (fallback path) lets you revisit
    earlier generations.

    Parameters
    ----------
    system : AutomataSystem
        An initialised system (StateSystem, etc.).
    max_generations : int
        Maximum number of steps that will be recorded.
    plot_type : str
        ``'standard'`` (binary gray) or ``'augmented'`` (TA confidence colours).
    n_states : int, optional
        States per TA arm — required when *plot_type* is ``'augmented'``.
    params : dict, optional
        Experiment parameters shown in the plot title.
    figsize : tuple
        Matplotlib figure size in inches.
    interval : int
        Milliseconds between automatic steps when running.
    view_window : int
        Rows visible at once.  Row pixel height = plot_height / view_window.
        History scrolls automatically (or manually when paused).
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
    ):
        self.system = system
        self.max_generations = max_generations
        self.plot_type = (plot_type
                          if plot_type == 'augmented' and n_states is not None
                          else 'standard')
        self.n_states = n_states
        self.params = params or {}
        self.figsize = figsize
        self.interval = interval
        self._view_window = view_window

        self._running = False
        self._current_gen = 0
        self._grid_size = len(list(system.grid.get_positions()))

        self._display = self._alloc_display()
        self._capture_row(0)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _alloc_display(self) -> np.ndarray:
        if self.plot_type == 'augmented':
            return np.zeros(
                (self.max_generations + 1, self._grid_size, 3), dtype=np.float32)
        return np.zeros(
                (self.max_generations + 1, self._grid_size), dtype=np.float32)

    def _capture_row(self, gen: int) -> None:
        states = self.system.grid.get_all_states()
        if self.plot_type == 'augmented':
            for i in range(self._grid_size):
                cs = states[i]
                ta = self.system.automata[i]
                p  = ta.get_position()
                intensity = p / (self.n_states - 1) if self.n_states > 1 else 1.0
                if cs == 0:
                    self._display[gen, i] = [
                        0.6 * (1 - intensity),
                        0.8 * (1 - intensity),
                        1.0 - 0.5 * intensity,
                    ]
                else:
                    self._display[gen, i] = [
                        1.0 - 0.5 * intensity,
                        0.6 * (1 - intensity),
                        0.6 * (1 - intensity),
                    ]
        else:
            self._display[gen] = states.astype(np.float32)

    def _default_filename(self) -> str:
        """Build a descriptive filename from experiment params + current gen.

        Example: ``spacetime_rule30_grid201_gen152.png``
        """
        import re
        parts = []
        for k, v in self.params.items():
            key = re.sub(r'\s+', '', str(k)).lower()
            val = re.sub(r'[^a-z0-9]', '', str(v).lower())
            parts.append(f'{key}{val}')
        parts.append(f'gen{self._current_gen}')
        return 'spacetime_' + '_'.join(parts) + '.png'

    def _build_title(self) -> str:
        base = ('TA-Augmented Space-Time Diagram'
                if self.plot_type == 'augmented' else 'Space-Time Diagram')
        if self.params:
            base += '  |  ' + '   '.join(f'{k}: {v}' for k, v in self.params.items())
        return base

    # ------------------------------------------------------------------
    # Public entry point — selects best available backend
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
    # Path A: Tkinter window (native OS buttons — most reliable)
    # ------------------------------------------------------------------

    def _show_tk(self, tk, filedialog, FigureCanvasTkAgg) -> None:
        root = tk.Tk()
        root.title('CA / CLA Interactive Simulation')
        root.configure(bg='#efefef')

        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor('#efefef')
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.10)

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

        def _refresh(view_top: Optional[int] = None) -> None:
            n = self._current_gen + 1
            im.set_data(self._display[:n])
            im.set_extent([-0.5, self._grid_size - 0.5, n - 0.5, -0.5])
            if view_top is None:
                view_top = max(0, n - self._view_window)
            ax.set_ylim(view_top + self._view_window - 0.5, view_top - 0.5)
            ax.set_title(
                f'{self._build_title()}    [Gen {self._current_gen}]', fontsize=10)
            mpl_canvas.draw_idle()

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
            self._display = self._alloc_display()
            self._capture_row(0)
            scrollbar.config(state=tk.DISABLED, to=0)
            scrollbar.set(0)
            _refresh(view_top=0)

        def on_save():
            path = filedialog.asksaveasfilename(
                parent=root,
                title='Export space-time diagram',
                initialfile=self._default_filename(),
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'),
                           ('PDF document', '*.pdf'),
                           ('SVG vector', '*.svg')],
            )
            if not path:
                return
            self._export(path)

        btn_start.config(command=on_start)
        btn_step .config(command=on_step)
        btn_reset.config(command=on_reset)
        btn_save .config(command=on_save)

        mpl_canvas.draw()
        root.mainloop()

    # ------------------------------------------------------------------
    # Path B: pure-matplotlib fallback (macosx / Qt / whatever is available)
    # ------------------------------------------------------------------

    def _show_mpl(self) -> None:
        # Pick an interactive backend (avoid Agg)
        for backend in ('macosx', 'QtAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg'):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

        from matplotlib.widgets import Button as MplButton

        fig, ax = plt.subplots(figsize=self.figsize)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.18)

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

        _imkw = dict(aspect='auto', interpolation='nearest', origin='upper')
        if self.plot_type == 'standard':
            _imkw.update(cmap='gray_r', vmin=0, vmax=1)

        im = ax.imshow(self._display[:1], **_imkw)
        ax.set_xlabel('Cell Position', fontsize=11)
        ax.set_ylabel('Generation', fontsize=11)
        ax.set_xlim(-0.5, self._grid_size - 0.5)
        ax.set_ylim(self._view_window - 0.5, -0.5)
        ax.set_title(f'{self._build_title()}    [Gen 0]', fontsize=10)

        def _refresh() -> None:
            n = self._current_gen + 1
            im.set_data(self._display[:n])
            im.set_extent([-0.5, self._grid_size - 0.5, n - 0.5, -0.5])
            view_top = max(0, n - self._view_window)
            ax.set_ylim(view_top + self._view_window - 0.5, view_top - 0.5)
            ax.set_title(
                f'{self._build_title()}    [Gen {self._current_gen}]', fontsize=10)
            fig.canvas.draw_idle()

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
            self._display = self._alloc_display()
            self._capture_row(0)
            _refresh()

        def on_save(_e):
            import os
            path = os.path.join(os.getcwd(), self._default_filename())
            self._export(path)

        def on_close(_e):
            import os
            os._exit(0)   # hard kill — bypasses Cocoa event loop cleanup

        fig.canvas.mpl_connect('close_event', on_close)

        btn_start.on_clicked(on_start)
        btn_step .on_clicked(on_step)
        btn_reset.on_clicked(on_reset)
        btn_save .on_clicked(on_save)

        plt.show()

    # ------------------------------------------------------------------
    # Shared export helper
    # ------------------------------------------------------------------

    def _export(self, path: str) -> None:
        """
        Save the full space-time diagram to *path*.

        Uses a standalone Figure + Agg canvas — completely decoupled from the
        live interactive window, so no backend conflicts or button artefacts.
        Each cell is rendered as a square pixel block; the figure expands
        vertically with the number of recorded generations.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        n = self._current_gen + 1
        dpi = 150
        px_per_cell = 4          # pixels per cell at the chosen dpi

        plot_w = self._grid_size * px_per_cell / dpi   # inches
        plot_h = n               * px_per_cell / dpi

        # Fixed margins in inches for labels / title
        left, right, bottom, top = 0.7, 0.2, 0.45, 0.40

        fig_w = plot_w + left + right
        fig_h = plot_h + bottom + top

        sfig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        FigureCanvasAgg(sfig)

        sax = sfig.add_axes([
            left   / fig_w,
            bottom / fig_h,
            plot_w / fig_w,
            plot_h / fig_h,
        ])

        _imkw = dict(aspect='equal', interpolation='nearest', origin='upper')
        if self.plot_type == 'standard':
            _imkw.update(cmap='gray_r', vmin=0, vmax=1)

        sax.imshow(self._display[:n], **_imkw)
        sax.set_xlabel('Cell Position', fontsize=9)
        sax.set_ylabel('Generation',    fontsize=9)
        sax.set_title(self._build_title(), fontsize=8)

        sfig.savefig(path, dpi=dpi)
        print(f'Saved → {path}')
