"""
thesis_style.py — shared figure style for all thesis figures.
=============================================================
Look modeled on the Stepanski benchmark thesis:
  - white background, no box: only bottom spine + light y-gridlines
  - thin muted lines; black = headline series, dashed black = secondary
  - light gray vertical bands for event episodes
  - bold 'A. Title' panel labels, frameless horizontal legend below
  - NO titles inside figures — captions belong in LaTeX

Usage:
    from thesis_style import apply_style, style_axis, shade_events, \
                             legend_below, panel_label, save_fig, PALETTE
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# series colors: headline black, then muted red / blue / green / orange / purple
PALETTE = ["#000000", "#c0392b", "#2e6da4", "#3a7d44", "#e67e22", "#6c5b7b"]
GRAY_BAND = "#d9d9d9"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Latin Modern serif to match the thesis body font (\usepackage{lmodern})
        # and the Ch.4 background figures (fig41_overview.py).
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,   # left spine kept — matches Ch.4 figures
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.left": False,
        "lines.linewidth": 1.0,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def style_axis(ax, ylabel=None, pct=False):
    """Apply per-axis cosmetics."""
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)
    ax.margins(x=0.01)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if pct:
        ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def shade_events(ax, dates, halfwidth_days=10):
    """Light gray vertical bands around event dates (calendar days)."""
    for d in dates:
        d = pd.Timestamp(d)
        ax.axvspan(d - pd.Timedelta(days=halfwidth_days),
                   d + pd.Timedelta(days=halfwidth_days),
                   color=GRAY_BAND, alpha=0.6, lw=0, zorder=0)


def panel_label(ax, text):
    """Bold 'A. Title' label at the top-left of a panel."""
    ax.set_title(text, loc="left", fontsize=8, fontweight="bold", pad=4)


def legend_below(fig, handles=None, labels=None, ncol=None, y=-0.02):
    """Frameless horizontal legend centered below the figure."""
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    ncol = ncol or min(len(labels), 6)
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), handlelength=1.8,
               columnspacing=1.2, frameon=False)


def save_fig(fig, path_stem, legend_pad=0.0):
    """Save as 300dpi PNG + vector PDF with room for a bottom legend."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if legend_pad:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + legend_pad)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
