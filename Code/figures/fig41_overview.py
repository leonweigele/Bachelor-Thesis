"""
fig41_overview.py — Figure 4.1 overview panels (dollar / oil / GPR).
====================================================================
Rebuilds the three Chapter-4 context figures from the CURRENT data with the
thesis house style: Latin Modern serif, black series line, left+bottom spine
only with light y-gridlines, each of the four main events marked by a dot on
the series and a thin leader line to a serif label (no colour bands, no box).

Reconstructed 2026-07-15 (the original scratchpad script was never committed).
Day-0 mapping corrected 2026-08-20 (CODE_AUDIT.md): markers now resolve with
searchsorted, matching the rest of the pipeline, instead of "nearest".

Run from the thesis root AFTER get_data.py:
    python3 "Code/figures/fig41_overview.py"              # dry run -> _ch04_regen/
    python3 "Code/figures/fig41_overview.py" --install    # writes the thesis copies
Outputs -> Output/figures/ch04_background/{fig_overview_dollar,fig_overview_oil,
           fig_overview_gpr}.{png,pdf}

NOTE: load_series() reads Data/processed/daily_panel.csv and Data/raw/gpr_daily.csv,
both gitignored, so this script cannot run on a fresh clone until get_data.py has
rebuilt them.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # shared helpers live in Code/common/
from es_common import ensure_latin_modern, load_events

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "Data/processed"
RAW = ROOT / "Data/raw"
FINAL_OUT = ROOT / "Output/figures/ch04_background"
TEMP_OUT = ROOT / "Output/figures/_ch04_regen"
FIGS = TEMP_OUT          # set by main(); --install switches it to FINAL_OUT

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

# four main events: display labels here, dates from Data/processed/events.csv
# via es_common (this script used to hold a fourth hard-coded copy of them).
EVENT_LABELS = {
    "ukraine":        "Ukraine invasion",
    "liberation_day": "Liberation Day",
    "iran_12day":     "12-day war",
    "hormuz":         "Hormuz",
}
_, _EVENT_DATES = load_events()
_missing = [k for k in EVENT_LABELS if k not in _EVENT_DATES]
if _missing:
    raise KeyError(f"events.csv has no row for {_missing}")
EVENTS = {k: (pd.Timestamp(_EVENT_DATES[k]), lab)
          for k, lab in EVENT_LABELS.items()}

# per-figure label placement, in AXES FRACTION (x, y) + horizontal alignment.
# Tuned to match the original Figure 4.1 layout.
# explicit y-limits for label headroom (matches original Fig 4.1); guarded
# against the data exceeding the top.
YLIM = {"dollar": (110, 136), "oil": (5, 150), "gpr": (0, 620)}

# NOTE: an x equal to the event's own axes-fraction position gives a vertical
# leader line. Event x-positions for START=2019-01-01, END=2026-06-30:
#   ukraine 0.4202 | liberation_day 0.8341 | iran_12day 0.8604 | hormuz 0.9554
PLACEMENT = {
    "dollar": {
        "ukraine":        (0.42, 0.90, "center"),
        "liberation_day": (0.81, 0.90, "center"),
        "iran_12day":     (0.8604, 0.10, "center"),   # vertical leader
        "hormuz":         (0.955, 0.90, "center"),
    },
    "oil": {
        "ukraine":        (0.37, 0.95, "center"),
        "liberation_day": (0.80, 0.78, "center"),
        "iran_12day":     (0.8604, 0.22, "center"),   # vertical leader
        "hormuz":         (0.95, 0.95, "center"),
    },
    "gpr": {
        "ukraine":        (0.32, 0.82, "center"),
        "liberation_day": (0.72, 0.63, "center"),
        "iran_12day":     (0.81, 0.90, "center"),
        "hormuz":         (0.93, 0.90, "center"),
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
    """Series value on day 0: the first observation at or AFTER the event date.

    This is the day-0 convention of the rest of the pipeline (searchsorted in
    03_event_study.py:68, 04:77, 05:63, 07:82 and es_common.rel_day) and of
    Section 5: an announcement on a non-trading day shifts FORWARD to the first
    day the market can react. The previous `get_indexer(..., method="nearest")`
    could resolve BACKWARDS — for the Saturday 2026-02-28 Hormuz date it picked
    Friday 27 Feb, putting the oil marker $5.92 below where it belongs and
    before the spike it labels.
    """
    idx = series.index
    pos = idx.searchsorted(pd.Timestamp(date))
    if pos >= len(idx):
        raise ValueError(
            f"event date {pd.Timestamp(date).date()} falls past the end of the "
            f"series (last observation {idx[-1].date()}) — no day 0 exists")
    return idx[pos], series.iloc[pos]


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


def report_markers(panels):
    """Print the resolved day-0 marker for every event and panel."""
    print("\nresolved markers (searchsorted = first trading day at or after):")
    for key, series in panels.items():
        for ev, (date, _lab) in EVENTS.items():
            d, v = value_at(series, date)
            shift = "" if d == date else f"  <- shifted from {date.date()}"
            print(f"  {key:7s} {ev:15s} {d.date()}  {v:>9.2f}{shift}")


def main(install=False):
    global FIGS
    FIGS = FINAL_OUT if install else TEMP_OUT
    FIGS.mkdir(parents=True, exist_ok=True)
    ensure_latin_modern()

    dollar, oil, gpr = load_series()
    print(f"dollar {len(dollar)}, oil {len(oil)}, gpr {len(gpr)} obs "
          f"({dollar.index.min().date()}..{dollar.index.max().date()})")
    report_markers({"dollar": dollar, "oil": oil, "gpr": gpr})
    print()
    make_panel(dollar, "Broad dollar index", "dollar", "fig_overview_dollar")
    make_panel(oil, "Brent crude (USD/bbl)", "oil", "fig_overview_oil")
    make_panel(gpr, "Geopolitical risk (daily)", "gpr", "fig_overview_gpr")
    print(f"Done -> {FIGS.relative_to(ROOT)}"
          + ("" if install else "   (dry run; pass --install to write the thesis copies)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install", action="store_true",
                    help="write into Output/figures/ch04_background/ "
                         "instead of the temp directory")
    main(**vars(ap.parse_args()))
