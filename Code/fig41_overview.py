"""
fig41_overview.py — Figure 4.1 overview panels (dollar / oil / GPR).
====================================================================
Rebuilds the three Chapter-4 context figures from the CURRENT data with the
thesis house style: Latin Modern serif, black series line, left+bottom spine
only with light y-gridlines, each of the four main events marked by a dot on
the series and a thin leader line to a serif label (no colour bands, no box).

Reconstructed 2026-07-15 (the original scratchpad script was never committed).
Run from the thesis root AFTER get_data.py:
    python3 "Code/fig41_overview.py"
Outputs -> Output/figures/{fig_overview_dollar,fig_overview_oil,fig_overview_gpr}.{png,pdf}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
RAW = ROOT / "Data/raw"
FIGS = ROOT / "Output/figures/ch04_background"
FIGS.mkdir(parents=True, exist_ok=True)

START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2026-06-30")

# ---- house style (no usetex: use the Latin Modern Roman font directly) -------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
})

# four main events: date + display label
EVENTS = {
    "ukraine":        (pd.Timestamp("2022-02-24"), "Ukraine invasion"),
    "liberation_day": (pd.Timestamp("2025-04-02"), "Liberation Day"),
    "iran_12day":     (pd.Timestamp("2025-06-13"), "12-day war"),
    "hormuz":         (pd.Timestamp("2026-02-28"), "Hormuz"),
}

# per-figure label placement, in AXES FRACTION (x, y) + horizontal alignment.
# Tuned to match the original Figure 4.1 layout.
# explicit y-limits for label headroom (matches original Fig 4.1); guarded
# against the data exceeding the top.
YLIM = {"dollar": (110, 136), "oil": (5, 150), "gpr": (0, 620)}

PLACEMENT = {
    "dollar": {
        "ukraine":        (0.42, 0.93, "center"),
        "liberation_day": (0.81, 0.93, "center"),
        "iran_12day":     (0.845, 0.06, "center"),
        "hormuz":         (0.955, 0.94, "center"),
    },
    "oil": {
        "ukraine":        (0.41, 0.95, "center"),
        "liberation_day": (0.80, 0.78, "center"),
        "iran_12day":     (0.85, 0.28, "center"),
        "hormuz":         (0.95, 0.95, "center"),
    },
    "gpr": {
        "ukraine":        (0.32, 0.82, "center"),
        "liberation_day": (0.72, 0.63, "center"),
        "iran_12day":     (0.81, 0.90, "center"),
        "hormuz":         (0.93, 0.81, "center"),
    },
}


def load_series():
    panel = pd.read_csv(PROC / "daily_panel.csv", parse_dates=["date"]).set_index("date")
    dollar = panel["DTWEXBGS"].dropna()
    oil = panel["DCOILBRENTEU"].dropna()
    gpr_raw = pd.read_csv(RAW / "gpr_daily.csv", parse_dates=["date"]).set_index("date")
    gpr = gpr_raw["GPRD"].dropna()
    clip = lambda s: s[(s.index >= START) & (s.index <= END)]
    return clip(dollar), clip(oil), clip(gpr)


def value_at(series, date):
    """Series value on the event date (nearest available trading day)."""
    idx = series.index.get_indexer([date], method="nearest")[0]
    return series.index[idx], series.iloc[idx]


def make_panel(series, ylabel, key, stem):
    fig, ax = plt.subplots(figsize=(10.2, 3.5))
    ax.plot(series.index, series.values, color="black", linewidth=0.8)

    for ev, (tx, ty, ha) in PLACEMENT[key].items():
        date, _label = EVENTS[ev]
        dot_x, dot_y = value_at(series, date)
        ax.plot([dot_x], [dot_y], "o", color="black", markersize=4.5, zorder=5)
        ax.annotate(
            _label,
            xy=(dot_x, dot_y), xycoords="data",
            xytext=(tx, ty), textcoords="axes fraction",
            ha=ha, va="center", fontsize=13,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="black",
                            shrinkA=1, shrinkB=4),
        )

    ax.set_xlim(START, END)
    lo, hi = YLIM[key]
    ax.set_ylim(lo, max(hi, series.max() * 1.03))
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(width=0.7, length=3.5, left=False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.margins(x=0.01)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def main():
    dollar, oil, gpr = load_series()
    print(f"dollar {len(dollar)}, oil {len(oil)}, gpr {len(gpr)} obs "
          f"({dollar.index.min().date()}..{dollar.index.max().date()})")
    make_panel(dollar, "Broad dollar index", "dollar", "fig_overview_dollar")
    make_panel(oil, "Brent crude (USD/bbl)", "oil", "fig_overview_oil")
    make_panel(gpr, "Geopolitical risk (daily)", "gpr", "fig_overview_gpr")
    print("Done -> Output/figures/")


if __name__ == "__main__":
    main()
