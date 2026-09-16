"""
06_make_thesis_figures.py — regenerate every Chapter-6/Appendix figure.
=======================================================================
WHY. CODE_AUDIT.md (2026-08-20), section E: 36 of the 39 figures the thesis
includes were produced by no script in the repository (they came out of a
one-off session on 2026-08-04 and only the images landed on disk), and the
fig_diff_* captions assert a confidence band that nothing on disk computed.
This script puts the whole figure layer back under code.

SCOPE — 40 files, of which the thesis includes 36:
  fig_car_<event>_<series>_half      4 events x 6 series          = 24
  fig_diff_<pair>_half               4 pairs                      =  4
  fig_w50_<event>_<series>_half      3 events x 4 series          = 12
      (fig_w50_ukraine_* exist on disk but are not \\includegraphics'd
       anywhere in the thesis; regenerated for completeness only.)

CONVENTIONS (matching the captions in content/06_results.tex and
content/11_appendix_a.tex, and the pinned result CSVs):
  fig_car_*   constant-mean ARs per 03_event_study.py, estimation window
              [-140,-21], event window [-20,+20]; plotted span -5..+20.
  fig_w50_*   per 04_event_study_w50.py, estimation [-170,-51], event
              [-50,+50]; plotted span -10..+50 as the captions state (04
              hard-codes its own two-panel figures to +/-50; the caption's
              -10..+50 is what the thesis displays and is used here).
  fig_diff_*  per 05_cross_event_tests.py: band 1.96*sqrt(L*(s_A^2+s_B^2)),
              L = h+1, sigmas from each event's own estimation window.

VERTICAL SCALES (Option B, adopted 2026-09-15 after the 2026-09-11 comparison
in Output/figures/ch06_results_B_shared_scale/). Within one figure the panels
that show the same kind of quantity share one y-range: the five currency panels
of each fig_car_<event> figure, the usd/oilexp/oilimp panels of each fig_w50
figure, and all four fig_diff panels. Brent keeps its own range (it moves five
to ten times as much). The shared range is the union of the ranges matplotlib
would choose for the panels on their own (same data, same margins), so nothing
is clipped and no number changes; the ranges used are written to YLIMITS.txt
next to the figures. The captions state the rule (content/06_results.tex,
fig:car_liberation_day and fig:diff_dollar).

ANCHORING. Every path is re-anchored to zero at day MINUS ONE, so the value
plotted at day h is CAR(0,h) — the exact quantity the tables test. Anchoring
at day 0 (as 03_event_study.py:242 does for its own cross-event figure via
`p - p.loc[:0].iloc[-1]`) would plot CAR(1,h) and put the headline endpoint
at -6.42 pp instead of the -7.237 pp that Table 6.3 / cross_event_diff.csv
report. Verified numerically in verify_diff() below.

CONTEXT MARKERS. Dated sub-events are marked with a dot and a text label.
Two kinds:
  - rows of Data/processed/events.csv (tariff pause, Hormuz ceasefire);
  - context dates that are NOT events.csv rows but are documented in the
    thesis and its sources — kept because the printed captions name them:
      13 Mar 2026 Kharg Island strikes   content/04_background.tex:24
      13 Apr 2026 US naval blockade      content/04_background.tex:26 (also
                                         events.csv description of
                                         hormuz_ceasefire: "collapsed 13 Apr")
      24 Jun 2025 twelve-day-war ceasefire  content/04_background.tex:22
  No entry is added to events.csv; the dates live only in this script's
  MARKERS tables, with their provenance, and the discrepancy between the
  Hormuz +/-50 caption and events.csv is reported by report_discrepancies().

VERIFICATION (run automatically, hard-fails on any mismatch):
  fig_car_*   every recomputed CAR path must equal the corresponding column
              of car_paths_<event>.csv bit-for-bit (tol 1e-12), and the
              anchored endpoints must equal the CAR/t of car_summary.csv for
              windows (0,1)/(0,5)/(0,10)/(0,20).
  fig_w50_*   recomputed CAR(0,5)/(0,20)/(0,50) and t must equal
              car_persistence_w50_long.csv (written by 04; tol 1e-5, that
              file is rounded to 6 decimals).
  fig_diff_*  endpoints at h=1/5/10/20 must equal cross_event_diff.csv
              (diff to its 3 printed decimals, t to its 2).
The old PNG/PDFs are never used as a comparison target — they are style
reference only, since their provenance is unknown by construction.

USAGE
  python3 Code/figures/06_make_thesis_figures.py            # -> Output/figures/ch06_results_regen/
  python3 Code/figures/06_make_thesis_figures.py --install  # -> Output/figures/ch06_results/
The default run never touches Output/figures/ch06_results/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # shared helpers live in Code/common/
from es_common import (ROOT, ES_OUT, EST_WIN_MAIN, EVT_WIN_MAIN,
                       EST_WIN_W50, EVT_WIN_W50, load_data, load_events,
                       rel_day, const_mean_event, ensure_latin_modern)
from thesis_style import apply_style, style_axis, save_fig

import matplotlib.pyplot as plt

TEMP_OUT = ROOT / "Output/figures/ch06_results_regen"
FINAL_OUT = ROOT / "Output/figures/ch06_results"

# ---------------------------------------------------------------- figure sets
# series key -> (data column, title text)
# MONOCHROME BY DESIGN (2026-08-20). Every panel in the thesis plots exactly one
# series and names it in its own title, so a per-series colour carried no
# information — it only made 26 of the 39 printed figures colour-dependent. Black
# line, grey interval band, which is what the fig_diff_* panels already used.
SERIES = {
    "usd":    ("r_DTWEXBGS",     "broad dollar index"),
    "brent":  ("r_DCOILBRENTEU", "Brent crude"),
    "safe":   ("SAFE",           "safe currencies"),
    "risky":  ("RISKY",          "risky currencies"),
    "oilexp": ("OIL_EXP",        "oil-exporter currencies"),
    "oilimp": ("OIL_IMP",        "oil-importer currencies"),
}
LINE_COLOR = "#000000"      # every series line and every marker dot
BAND_COLOR = "#000000"      # drawn at BAND_ALPHA, so it reads as grey
BAND_ALPHA = 0.15
CAR_EVENTS = ["liberation_day", "iran_12day", "hormuz", "ukraine"]
W50_EVENTS = ["hormuz", "liberation_day", "ukraine"]
W50_SERIES = ["usd", "brent", "oilexp", "oilimp"]
DIFF_PAIRS = [("liberation_day", "hormuz"), ("liberation_day", "ukraine"),
              ("hormuz", "ukraine"), ("liberation_day", "iran_12day")]

# figure titles: car figures use the chapter-6 event names, diff figures the
# shorter names of 05_cross_event_tests.py (visible in the existing images)
EVENT_TITLE = {"liberation_day": "Liberation Day", "iran_12day": "Twelve-day war",
               "hormuz": "Hormuz crisis", "ukraine": "Ukraine invasion"}
DIFF_TITLE = {"liberation_day": "Liberation Day", "hormuz": "Hormuz crisis",
              "ukraine": "Ukraine", "iran_12day": "Twelve-day war"}

# ---- sub-event markers: (date, label, side). Provenance in the docstring. --
PAUSE = ("events.csv", "tariff_pause")            # resolved from events.csv
CEASE = ("events.csv", "hormuz_ceasefire")        # resolved from events.csv
CAR_MARKERS = {
    "liberation_day": [(PAUSE, "tariff pause", "below")],
    "iran_12day": [(("2025-06-24",), "ceasefire", "below")],          # 04_background.tex:22
    "hormuz": [(("2026-03-13",), "Kharg Island strikes", "above")],   # 04_background.tex:24
    "ukraine": [],
}
W50_MARKERS = {
    "liberation_day": [(PAUSE, "tariff pause", "below")],
    "hormuz": [(("2026-03-13",), "Kharg Island strikes", "above"),    # 04_background.tex:24
               (CEASE, "ceasefire", "above"),
               (("2026-04-13",), "US blockade", "below")],            # 04_background.tex:26
    "ukraine": [],
}

FIGSIZE = (3.0, 2.3)
Z = 1.96                       # 95% band, same normal quantile as the stars


# ---------------------------------------------------------------- helpers
def resolve_marker_date(spec, ev_dates):
    if spec[0] == "events.csv":
        return ev_dates[spec[1]]
    return spec[0]


def base_axis(ax, xlo, xhi, xstep, ylabel, event_line=True):
    if event_line:
        ax.axvline(0, color="#999999", lw=0.8, zorder=1)
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_xticks(np.arange(xlo, xhi + 1, xstep))
    ax.set_xlim(xlo - 0.6, xhi + 0.6)
    ax.set_xlabel("Trading days relative to the event")
    ax.yaxis.set_major_locator(
        plt.MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    style_axis(ax, ylabel=ylabel)


def add_marker(ax, x, y, label, side, color, i_above=0):
    """Dot on the path plus a text label inside the axes, connected by a
    thin gray leader line — the layout of the existing figures. Labels on
    the 'above' side are staggered downward when there is more than one."""
    ax.plot([x], [y], marker="o", ms=5.5, color=color, zorder=6)
    ylo, yhi = ax.get_ylim()
    rng = yhi - ylo
    if side == "above":
        y_text = min(y + 0.42 * rng, yhi - (0.03 + 0.12 * i_above) * rng)
        va = "top"
    else:
        y_text = max(y - 0.40 * rng, ylo + 0.04 * rng)
        va = "bottom"
    ax.annotate(label, xy=(x, y), xytext=(x, y_text), textcoords="data",
                ha="center", va=va, fontsize=8, zorder=6,
                arrowprops=dict(arrowstyle="-", color="#999999",
                                lw=0.6, shrinkA=2, shrinkB=4))
    ax.set_ylim(ylo, yhi)


def car_figure(res, event, skey, out_dir, wide=False, markers=(),
               data_index=None, ev_date=None, ev_dates=None,
               ylim=None, save=True):
    """One single-series CAR panel, re-anchored to zero at day -1.
    ylim: fixed y-range (Option B shared scale); None = matplotlib's own.
    save=False only measures the natural range and closes the figure.
    Returns (stem, y-range used)."""
    col, nice = SERIES[skey]
    anch = (res["car"] - res["car"].loc[-1]) * 100.0
    xlo, xhi, xstep = (-10, 50, 10) if wide else (-5, 20, 5)
    span = anch.loc[xlo:xhi]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    hs = np.array([h for h in anch.index if 0 <= h <= xhi])
    band = Z * res["sigma"] * np.sqrt(hs + 1) * 100.0
    ax.fill_between(hs, anch.loc[hs] - band, anch.loc[hs] + band,
                    color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=2)
    ax.plot(span.index, span.values, color=LINE_COLOR, lw=1.6, zorder=5,
            solid_capstyle="round")
    base_axis(ax, xlo, xhi, xstep, "CAR (%)")
    ax.margins(y=0.24)
    if ylim is not None:
        ax.set_ylim(*ylim)
    used = ax.get_ylim()
    i_above = 0
    for spec, label, side in markers:
        d = resolve_marker_date(spec, ev_dates)
        r = rel_day(data_index, ev_date, d)
        if xlo <= r <= xhi and r in anch.index:
            add_marker(ax, r, anch.loc[r], label, side, LINE_COLOR, i_above)
            i_above += (side == "above")
    ax.set_title(f"{EVENT_TITLE[event]}: {nice}", fontsize=9, pad=8)
    stem = (f"fig_w50_{event}_{skey}_half" if wide
            else f"fig_car_{event}_{skey}_half")
    if save:
        save_fig(fig, out_dir / stem)
    else:
        plt.close(fig)
    return stem, used


def diff_figure(res_a, res_b, pair, out_dir, ylim=None, save=True):
    """Difference in the broad dollar's CAR(0,h), band per 05's SE.
    ylim / save as in car_figure. Returns (stem, diff, band, y-range used)."""
    a, b = pair
    hs = np.arange(0, 21)
    da = (res_a["car"] - res_a["car"].loc[-1]).loc[hs]
    db = (res_b["car"] - res_b["car"].loc[-1]).loc[hs]
    diff = (da - db) * 100.0
    band = Z * np.sqrt((hs + 1) *
                       (res_a["sigma"] ** 2 + res_b["sigma"] ** 2)) * 100.0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(hs, diff.values - band, diff.values + band,
                    color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=2)
    ax.plot(hs, diff.values, color=LINE_COLOR, lw=1.6, zorder=5,
            solid_capstyle="round")
    base_axis(ax, 0, 20, 5, "$\\Delta$ CAR (pp)", event_line=False)
    ax.margins(y=0.15)
    if ylim is not None:
        ax.set_ylim(*ylim)
    used = ax.get_ylim()
    ax.set_title(f"{DIFF_TITLE[a]} − {DIFF_TITLE[b]}", fontsize=9, pad=8)
    stem = f"fig_diff_{a}_{b}_half"
    if save:
        save_fig(fig, out_dir / stem)
    else:
        plt.close(fig)
    return stem, diff, band, used


def union(rngs):
    """Smallest y-range containing every range in rngs (Option B shared scale)."""
    return (min(r[0] for r in rngs), max(r[1] for r in rngs))


# Panels that share one vertical scale within a figure (Option B); Brent is
# always on its own scale.
CAR_SHARED = ["usd", "safe", "risky", "oilexp", "oilimp"]
W50_SHARED = ["usd", "oilexp", "oilimp"]


# ---------------------------------------------------------------- verification
class Check:
    def __init__(self):
        self.rows, self.failed = [], 0

    def add(self, fig, what, ok, detail=""):
        self.rows.append((fig, what, "OK" if ok else "FAIL", detail))
        if not ok:
            self.failed += 1

    def report(self):
        w = max(len(r[0]) for r in self.rows) + 2
        lines = [f"{r[0]:{w}s} {r[1]:34s} {r[2]:5s} {r[3]}" for r in self.rows]
        return "\n".join(lines)


def verify_car(check, stem, res, event, col, paths_csv, summary):
    csv_col = paths_csv[f"{col}__cm"]
    d = float((res["car"] - csv_col).abs().max())
    check.add(stem, "path == car_paths csv", d < 1e-12, f"max|d|={d:.1e}")
    for h in (1, 5, 10, 20):
        row = summary[(summary.event == event) & (summary.series == col)
                      & (summary.method == "const_mean")
                      & (summary.window == f"(0,{h})")].iloc[0]
        car = res["car"].loc[h] - res["car"].loc[-1]
        t = car / (res["sigma"] * np.sqrt(h + 1))
        ok = abs(car - row.CAR) < 1e-12 and abs(t - row.t) < 1e-9
        check.add(stem, f"CAR/t (0,{h}) == car_summary", ok,
                  f"{car*100:+.2f}%, t={t:+.2f}")


def verify_w50(check, stem, res, event, col, w50_long):
    for h in (5, 20, 50):
        row = w50_long[(w50_long.event == event) & (w50_long.series == col)
                       & (w50_long.window == f"(0,{h})")]
        if row.empty:
            check.add(stem, f"(0,{h}) in w50_long", False, "row missing")
            continue
        row = row.iloc[0]
        car = res["car"].loc[h] - res["car"].loc[-1]
        t = car / (res["sigma"] * np.sqrt(h + 1))
        ok = abs(car - row.CAR) < 1e-5 and abs(t - row.t) < 1e-4
        check.add(stem, f"CAR/t (0,{h}) == w50_long", ok,
                  f"{car*100:+.2f}%, t={t:+.2f}")


def verify_diff(check, stem, pair, diff, res_a, res_b, ced):
    a, b = pair
    for h in (1, 5, 10, 20):
        row = ced[(ced.series == "r_DTWEXBGS") & (ced.pair == f"{a}-{b}")
                  & (ced.window == f"(0,{h})")].iloc[0]
        se = np.sqrt((h + 1) * (res_a["sigma"] ** 2 + res_b["sigma"] ** 2))
        t = (diff.loc[h] / 100.0) / se
        ok = (round(float(diff.loc[h]), 3) == row["diff_%"]
              and round(float(t), 2) == row["t"])
        check.add(stem, f"endpoint h={h} == cross_event_diff", ok,
                  f"{diff.loc[h]:+.3f} pp, t={t:+.2f}")


def report_discrepancies():
    return (
        "CAPTION DISCREPANCIES (reported, not papered over):\n"
        "(a) The Hormuz +/-50 caption (content/06_results.tex, fig:car_hormuz_w50)\n"
        "    marks a '13 April naval blockade' that has no row in events.csv.\n"
        "    Handling: the marker is drawn from a context date hard-coded in this\n"
        "    script, sourced to content/04_background.tex:26 and the events.csv\n"
        "    description of hormuz_ceasefire ('collapsed 13 Apr'). No events.csv\n"
        "    row was invented. Same treatment for the 13 Mar Kharg Island strikes\n"
        "    (04_background.tex:24) and the 24 Jun 2025 ceasefire (:22).\n"
        "(b) The same caption says the panel spans -10 to +50, while\n"
        "    04_event_study_w50.py hard-codes its own two-panel figures to +/-50.\n"
        "    Handling: these single-series panels follow the CAPTION (-10..+50) —\n"
        "    that is what the thesis prints and what the old images show; 04's\n"
        "    +/-50 axis belongs to its own fig_car_<event>_w50 figures, which are\n"
        "    not part of the thesis. 04 was not modified.\n")


# ---------------------------------------------------------------- main
def main(install=False):
    out_dir = FINAL_OUT if install else TEMP_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    ensure_latin_modern()

    data = load_data()
    events, ev_dates = load_events()
    summary = pd.read_csv(ES_OUT / "car_summary.csv")
    ced = pd.read_csv(ES_OUT / "cross_event_diff.csv")
    w50_long = pd.read_csv(ES_OUT / "car_persistence_w50_long.csv")
    check = Check()
    made = []
    ylog = []          # y-ranges used, written to YLIMITS.txt

    # ---- fig_car_* : per event the currency panels share one scale ---------
    for event in CAR_EVENTS:
        paths_csv = pd.read_csv(ES_OUT / f"car_paths_{event}.csv",
                                index_col="rel_day")
        res = {skey: const_mean_event(data, col, ev_dates[event])
               for skey, (col, _) in SERIES.items()}
        kw = dict(markers=CAR_MARKERS[event], data_index=data.index,
                  ev_date=ev_dates[event], ev_dates=ev_dates)
        natural = [car_figure(res[s], event, s, out_dir, save=False, **kw)[1]
                   for s in CAR_SHARED]
        shared = union(natural)
        for skey, (col, _) in SERIES.items():
            stem, used = car_figure(res[skey], event, skey, out_dir,
                                    ylim=shared if skey in CAR_SHARED else None,
                                    **kw)
            made.append(stem)
            ylog.append(f"{stem}  ylim {used[0]:+.2f} .. {used[1]:+.2f}"
                        + ("  (shared)" if skey in CAR_SHARED else "  (own)"))
            verify_car(check, stem, res[skey], event, col, paths_csv, summary)

    # ---- fig_w50_* : per event usd/oilexp/oilimp share one scale -----------
    for event in W50_EVENTS:
        res = {skey: const_mean_event(data, SERIES[skey][0], ev_dates[event],
                                      est_win=EST_WIN_W50, evt_win=EVT_WIN_W50)
               for skey in W50_SERIES}
        kw = dict(wide=True, markers=W50_MARKERS[event], data_index=data.index,
                  ev_date=ev_dates[event], ev_dates=ev_dates)
        natural = [car_figure(res[s], event, s, out_dir, save=False, **kw)[1]
                   for s in W50_SHARED]
        shared = union(natural)
        for skey in W50_SERIES:
            col = SERIES[skey][0]
            stem, used = car_figure(res[skey], event, skey, out_dir,
                                    ylim=shared if skey in W50_SHARED else None,
                                    **kw)
            made.append(stem)
            ylog.append(f"{stem}  ylim {used[0]:+.2f} .. {used[1]:+.2f}"
                        + ("  (shared)" if skey in W50_SHARED else "  (own)"))
            verify_w50(check, stem, res[skey], event, col, w50_long)

    # ---- fig_diff_* : all four panels share one scale ----------------------
    resd = {pair: (const_mean_event(data, "r_DTWEXBGS", ev_dates[pair[0]]),
                   const_mean_event(data, "r_DTWEXBGS", ev_dates[pair[1]]))
            for pair in DIFF_PAIRS}
    natural = [diff_figure(*resd[pair], pair, out_dir, save=False)[3]
               for pair in DIFF_PAIRS]
    shared = union(natural)
    for pair in DIFF_PAIRS:
        res_a, res_b = resd[pair]
        stem, diff, _, used = diff_figure(res_a, res_b, pair, out_dir, ylim=shared)
        made.append(stem)
        ylog.append(f"{stem}  ylim {used[0]:+.2f} .. {used[1]:+.2f}  (shared)")
        verify_diff(check, stem, pair, diff, res_a, res_b, ced)

    (out_dir / "YLIMITS.txt").write_text("\n".join(ylog) + "\n")

    # ---- report ------------------------------------------------------------
    print(f"\nWrote {len(made)} x (png+pdf) to {out_dir.relative_to(ROOT)}/")
    print("\n" + check.report())
    n_ok = len(check.rows) - check.failed
    print(f"\nVERIFICATION: {n_ok}/{len(check.rows)} checks passed, "
          f"{check.failed} failed.")
    print("\n" + report_discrepancies())
    (out_dir / "VERIFICATION.txt").write_text(
        check.report() + f"\n\n{n_ok}/{len(check.rows)} passed, "
        f"{check.failed} failed.\n\n" + report_discrepancies())
    if check.failed:
        sys.exit(f"{check.failed} verification checks FAILED — figures in "
                 f"{out_dir} must not be installed.")
    if not install:
        print("Temp run only — nothing in Output/figures/ch06_results/ was "
              "touched. Re-run with --install after approval.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true",
                    help="write into Output/figures/ch06_results/ "
                         "(default: ch06_results_regen/)")
    main(install=ap.parse_args().install)
