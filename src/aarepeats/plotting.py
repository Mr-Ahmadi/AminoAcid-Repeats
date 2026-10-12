"""Publication-style figures for each detector (matplotlib).

All functions draw into a supplied ``Axes`` (or create one) and return it, so
they compose into multi-panel figures for the report, slides and notebooks.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from .alignment import SelfAlignment, windowed_dotplot
from .entropy import LCRResult
from .fourier import PeriodicityMap, PeriodicRegion
from .tandem import TandemRepeat

# Validated categorical palette (first three slots are safe all-pairs) and inks.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
}


def use_style() -> None:
    """Apply the project figure style globally."""
    plt.rcParams.update(STYLE)


def _ax(ax: Axes | None, size=(7, 2.4)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def plot_entropy(
    result: LCRResult, seq: str, trigger: float, extension: float,
    ax: Axes | None = None, show_residues: bool | None = None,
) -> Axes:
    """Sliding-window entropy with thresholds and shaded LCRs."""
    ax = _ax(ax)
    for r in result.regions:
        ax.axvspan(r.start - 0.5, r.end - 0.5, color=ORANGE, alpha=0.15, lw=0, zorder=0)
    ax.plot(result.centers, result.entropies, color=BLUE, lw=1.4, zorder=3, label="window entropy")
    ax.axhline(trigger, color=ORANGE, lw=1, ls="--", zorder=2, label=f"trigger {trigger:g} bits")
    ax.axhline(extension, color=MUTED, lw=1, ls=":", zorder=2, label=f"extension {extension:g} bits")
    ax.set_xlim(-0.5, len(seq) - 0.5)
    ax.set_ylim(0, max(result.entropies.max(), extension) * 1.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7,
              handlelength=1.6, columnspacing=1.2)
    ax.set_ylabel("Shannon entropy (bits)")
    if show_residues is None:
        show_residues = len(seq) <= 90
    if show_residues:
        ax.set_xticks(range(len(seq)))
        ax.set_xticklabels(list(seq), fontfamily="monospace", fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    else:
        ax.set_xlabel("Residue position")
    return ax


def plot_spectrum(freq: np.ndarray, power: np.ndarray, ax: Axes | None = None,
                  max_period: float = 12, label_peaks: int = 1) -> Axes:
    """Power spectrum against period (1/frequency)."""
    ax = _ax(ax, (3.4, 2.4))
    period = 1 / freq
    keep = period <= max_period
    ax.vlines(period[keep], 0, power[keep], color=BLUE, lw=0.9)
    ax.set_xlim(1.8, max_period)
    ax.set_xlabel("Period (residues)")
    ax.set_ylabel("Relative power")
    order = np.argsort(power[keep])[::-1][:label_peaks]
    for k in order:
        p, h = period[keep][k], power[keep][k]
        ax.annotate(f"period {p:.2f}", (p, h), xytext=(8, -2), textcoords="offset points",
                    fontsize=7.5, color=INK, va="top")
    ax.grid(axis="x", visible=False)
    return ax


def plot_periodicity_map(pmap: PeriodicityMap, ax: Axes | None = None,
                         regions: Sequence[PeriodicRegion] = ()) -> Axes:
    """Heatmap of the local excess periodicity score (period x position)."""
    ax = _ax(ax)
    n = pmap.score.shape[1]
    im = ax.imshow(pmap.score, aspect="auto", origin="lower", cmap=SEQUENTIAL,
                   vmin=0, vmax=1, interpolation="nearest",
                   extent=(0.5, n + 0.5, pmap.periods[0] - 0.5, pmap.periods[-1] + 0.5))
    for r in regions:
        ax.plot([r.start + 1, r.end], [r.period, r.period], color=ORANGE, lw=1.6, solid_capstyle="butt")
    ax.set_ylabel("Period p")
    ax.set_xlabel("Residue position")
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Excess score $E_p$", fontsize=8, color=INK2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7)
    return ax


def plot_dotplot(seq: str, alignments: Sequence[SelfAlignment] = (), window: int = 9,
                 threshold: float = 0.66, ax: Axes | None = None, offset: int = 0) -> Axes:
    """Filtered self dot plot (upper triangle) with reported alignments overlaid."""
    ax = _ax(ax, (3.6, 3.6))
    d = windowed_dotplot(seq, window) >= threshold * window
    ys, xs = np.nonzero(np.triu(d, 1))
    # A hit at (i, j) marks the window ending there; draw the whole diagonal stroke.
    for i, j in zip(ys, xs):
        ax.plot([j - window + 1.5 + offset, j + 1.5 + offset], [i - window + 1.5 + offset, i + 1.5 + offset],
                color=MUTED, lw=0.5, alpha=0.6, solid_capstyle="butt")
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    for k, a in enumerate(alignments):
        ax.plot([a.start2 + 1, a.end2], [a.start1 + 1, a.end1], color=colors[k % 4], lw=2.2,
                solid_capstyle="round", label=f"#{k + 1}  score {a.score:.0f}")
    n = len(seq)
    ax.plot([offset, offset + n], [offset, offset + n], color=INK2, lw=0.6)
    ax.set_xlim(offset + 0.5, offset + n + 0.5)
    ax.set_ylim(offset + n + 0.5, offset + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Position j")
    ax.set_ylabel("Position i")
    ax.grid(False)
    if alignments:
        ax.legend(loc="lower left", handlelength=1.2)
    return ax


def plot_repeat_map(length: int, tracks: dict[str, list[tuple[int, int]]], ax: Axes | None = None,
                    title: str | None = None, annotations: Sequence[tuple[int, int, str]] = ()) -> Axes:
    """One row per detector; each interval ``[start, end)`` drawn as a bar."""
    ax = _ax(ax, (7, 1.4))
    colors = {"Entropy (LCR)": ORANGE, "Tandem": BLUE, "Fourier (periodic)": AQUA,
              "Self-alignment": YELLOW}
    names = list(tracks)
    for row, name in enumerate(names):
        y = len(names) - 1 - row
        ax.add_patch(plt.Rectangle((0, y - 0.03), length, 0.06, color=GRID, lw=0))
        for a, b in tracks[name]:
            w = max(b - a, length * 0.002)
            ax.add_patch(plt.Rectangle((a, y - 0.3), w, 0.6, color=colors.get(name, BLUE), lw=0))
    for a, b, text in annotations:
        ax.annotate(text, ((a + b) / 2, len(names) - 0.55), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=INK)
    ax.set_xlim(0, length)
    ax.set_ylim(-0.6, len(names) - 0.2 + (0.6 if annotations else 0))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    if title:
        ax.set_title(title)
    return ax
