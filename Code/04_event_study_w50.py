"""
04_event_study_w50.py — long-horizon (+/-50 trading day) robustness for the event study.
========================================================================================
Companion to 03_event_study.py. Same constant-mean abnormal-return method, but a much
WIDER event window, to see whether each initial reaction PERSISTS or REVERTS over the
~10 weeks after the event. This is a robustness / persistence exhibit, NOT a second
identification of the event effect (see the note below).

Key differences from 03_event_study.py
  - EVT_WIN = (-50, +50)        vs (-20, +20)
  - EST_WIN = (-170, -51)       pushed back so it does NOT overlap the wider event
                                window. 03's (-130,-11) would put the [-50,-11]
                                pre-event days inside the estimation window, so the
                                "normal" mean would be estimated partly on event-window
                                data. 120 clean trading days still remain for mu/sigma.
  - Events that need more post-event data than the panel holds are SKIPPED, not
    truncated. The panel ends 2026-06-11, so the 2026 ceasefire (0 post days) and
    US strikes (~10 post days) cannot get a real +50 window.
  - Per-event figures get a `_w50` suffix; the +/-20 thesis figures are left untouched.
  - Any other event / sub-event falling inside the +/-50 window is marked on the chart,
    so contamination of the long window is visible rather than hidden.

Outputs
  Output/figures/fig_car_<event>_w50.png / .pdf      portfolio CAR paths, +/-50 days
  Data/processed/event_study/car_persistence_w50.csv CAR(0,+5)/(0,+20)/(0,+50) + t

METHOD NOTE for the thesis. The formal event effect remains the short-window
CAR(0,+1)/CAR(0,+5) from 03_event_study.py. Over +/-50 trading days the window absorbs
unrelated macro news (Fed meetings, CPI/jobs prints, other tariff headlines) and, for
the Apr/Jun 2025 events, grazes the neighbouring event. So this window measures medium-
term DRIFT, not the announcement shock. Read these figures as "did the move hold?".
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
OUT = PROC / "event_study"
FIGS = ROOT / "Output/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

EST_WIN = (-170, -51)        # estimation window — ends before the wide event window
EVT_WIN = (-50, 50)          # long event window (+/-50 trading days)
CAR_WINDOWS = [(0, 5), (0, 20), (0, 50)]   # reaction -> short hold -> full horizon
MIN_POST = 45                # need at least this many post-event days for a figure

FIG_SERIES = ["USD_EW", "SAFE", "RISKY", "OIL_EXP", "OIL_IMP"]
KEY_SERIES = ["r_DTWEXBGS", "USD_EW", "SAFE", "RISKY", "CARRY_HML",
              "OIL_EXP", "OIL_IMP", "r_DCOILBRENTEU", "r_DCOILWTICO",
              "r_SP500", "r_Gold", "d_VIXCLS", "d_DGS10"]


def stars(t):
    return "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else \
           "*" if abs(t) > 1.65 else ""


def load():
    rets = pd.read_csv(PROC / "returns_daily.csv",
                       parse_dates=["date"], index_col="date")
    ports = pd.read_csv(PROC / "portfolios_daily.csv",
                        parse_dates=["date"], index_col="date")
    data = pd.concat([rets, ports], axis=1)
    events = pd.read_csv(PROC / "events.csv")
    return data, events


def event_slices(index, event_date):
    """Estimation / event window positions and how many post-event days exist."""
    pos = index.searchsorted(pd.Timestamp(event_date))
    est = index[max(0, pos + EST_WIN[0]): max(0, pos + EST_WIN[1] + 1)]
    evt_lo, evt_hi = pos + EVT_WIN[0], pos + EVT_WIN[1]
    if evt_lo < 0 or pos >= len(index):
        return None, None, None, 0
    evt = index[evt_lo: min(len(index), evt_hi + 1)]
    rel = np.arange(EVT_WIN[0], EVT_WIN[0] + len(evt))
    post_avail = int((rel > 0).sum())          # trading days after the event present
    return est, evt, rel, post_avail


def run_event(data, name, date):
    est_idx, evt_idx, rel, post_avail = event_slices(data.index, date)
    if est_idx is None or len(est_idx) < 60:
        return None, [], post_avail
    if post_avail < MIN_POST:
        return None, [], post_avail            # not enough post data for +/-50

    paths, rows = {}, []
    for col in data.columns:
        est = data.loc[est_idx, col].dropna()
        evt = data.loc[evt_idx, col]
        if len(est) < 60 or evt.isna().all():
            continue
        mu, sig = est.mean(), est.std()
        ar = evt - mu
        car = ar.cumsum()
        paths[col] = pd.Series(car.values, index=rel[:len(car)])
        if col in KEY_SERIES:
            for (a, b) in CAR_WINDOWS:
                m = (rel >= a) & (rel <= b)
                c = ar.iloc[m[:len(ar)]].sum()
                L = int(m[:len(ar)].sum())
                t = c / (sig * np.sqrt(L)) if sig > 0 and L > 0 else np.nan
                rows.append({"event": name, "series": col,
                             "window": f"({a},{b})", "CAR": c, "t": t,
                             "sig": stars(t)})

    path_df = pd.DataFrame(paths)
    path_df.index.name = "rel_day"
    return path_df, rows, post_avail


# ----------------------------------------------------------------------------- figures
from thesis_style import (apply_style, style_axis, legend_below, save_fig,  # noqa: E402
                          PALETTE)
import matplotlib.pyplot as plt  # noqa: E402
apply_style()

NICE = {"USD_EW": "Dollar (EW index)", "SAFE": "Safe currencies",
        "RISKY": "Risky currencies", "OIL_EXP": "Oil exporters",
        "OIL_IMP": "Oil importers"}
EVENT_NICE = {"ukraine": "Ukraine invasion (Feb 2022)",
              "liberation_day": "Liberation Day (Apr 2025)",
              "iran_12day": "Twelve-day war (Jun 2025)",
              "hormuz": "Hormuz crisis (Feb 2026)"}
SHORT = {"ukraine": "Ukraine", "liberation_day": "Liberation Day",
         "tariff_pause": "tariff pause", "iran_12day": "12-day war",
         "hormuz": "Hormuz", "hormuz_closure": "Hormuz closure",
         "hormuz_ceasefire": "2-wk ceasefire",
         "us_strikes": "US strikes", "ceasefire": "ceasefire"}
STYLES = {"USD_EW": ("black", "--"), "SAFE": (PALETTE[2], "-"),
          "RISKY": (PALETTE[1], "-"), "OIL_EXP": (PALETTE[3], "-"),
          "OIL_IMP": (PALETTE[4], "-")}
# crude benchmarks on a secondary (right) axis — over +/-50 days oil's CAR runs
# ~10-30x the FX baskets, so it cannot share the left axis.
OIL_STYLE = {"r_DCOILBRENTEU": ("#8c6d31", "-.", "Brent crude (rhs)"),
             "r_DCOILWTICO": ("#666666", ":", "WTI crude (rhs)")}


def mark_neighbors(ax, index, this_event, this_date, events):
    """Thin dotted markers for other events that fall inside the +/-50 window."""
    p0 = index.searchsorted(pd.Timestamp(this_date))
    for _, o in events.iterrows():
        if o["event"] == this_event:
            continue
        off = int(index.searchsorted(pd.Timestamp(o["date"])) - p0)
        if EVT_WIN[0] < off < EVT_WIN[1] and off != 0:
            ax.axvline(off, color="#b0b0b0", lw=0.6, ls=":", zorder=1)
            ax.annotate(SHORT.get(o["event"], o["event"]),
                        xy=(off, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -2), textcoords="offset points",
                        fontsize=6, color="#808080", rotation=90,
                        ha="left", va="top")


def figure_event(name, paths, index, date, events, post_avail):
    cols = [c for c in FIG_SERIES if c in paths.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for c in cols:
        color, ls = STYLES.get(c, (PALETTE[5], "-"))
        ax.plot(paths.index, 100 * paths[c], lw=1.1, color=color, ls=ls,
                label=NICE.get(c, c))

    # --- crude oil (Brent + WTI) on a secondary right-hand axis ------------
    ax2 = ax.twinx()
    for c, (color, ls, lab) in OIL_STYLE.items():
        if c in paths.columns:
            ax2.plot(paths.index, 100 * paths[c], lw=1.0, color=color, ls=ls,
                     label=lab)
    ax2.set_ylabel("Crude CAR (%)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.grid(False)
    ax2.yaxis.set_ticks_position("right")
    ax2.tick_params(left=False, right=True, labelright=True, width=0.6, length=3)
    ax2.margins(x=0.01)

    mark_neighbors(ax, index, name, date, events)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.axhline(0, color="#999999", lw=0.5)
    # if the window was clipped by end-of-sample, say so on the figure
    last = int(paths.index.max())
    if last < EVT_WIN[1]:
        ax.annotate(f"data ends at +{last}", xy=(last, 0),
                    xytext=(-4, 6), textcoords="offset points",
                    fontsize=6, color="#808080", ha="right")
    ax.set_xlabel("Trading days relative to event")
    ax.set_xlim(EVT_WIN[0], EVT_WIN[1])
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(10))
    style_axis(ax, ylabel="FX CAR (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend_below(fig, handles=h1 + h2, labels=l1 + l2, ncol=5, y=-0.04)
    save_fig(fig, FIGS / f"fig_car_{name}_w50", legend_pad=0.16)


if __name__ == "__main__":
    data, events = load()
    print(f"Loaded {data.shape[1]} series, {len(events)} events  "
          f"(panel ends {data.index.max().date()})")
    print(f"Windows: estimation {EST_WIN}, event {EVT_WIN}; "
          f"need >= {MIN_POST} post-event days.\n")

    all_rows, kept, skipped = [], [], []
    for _, ev in events.iterrows():
        paths, rows, post = run_event(data, ev["event"], ev["date"])
        if paths is None:
            skipped.append((ev["event"], ev["date"], post))
            continue
        all_rows += rows
        figure_event(ev["event"], paths, data.index, ev["date"], events, post)
        kept.append(ev["event"])
        clip = "" if int(paths.index.max()) >= EVT_WIN[1] else \
            f"  [clipped at +{int(paths.index.max())}]"
        print(f"  OK   {ev['event']:16s} ({ev['date']})  "
              f"{paths.shape[1]:3d} CAR paths, {post} post-days{clip}")

    for name, date, post in skipped:
        print(f"  SKIP {name:16s} ({date})  only {post} post-event days "
              f"(< {MIN_POST}) — +/-50 window not available")

    # persistence table: reaction -> hold -> full horizon
    pers = pd.DataFrame(all_rows)
    if not pers.empty:
        pers["scale"] = np.where(pers["series"].str.startswith("d_"), 1.0, 100.0)
        pers["value"] = (pers["scale"] * pers["CAR"]).round(2)
        pers["cell"] = pers["value"].astype(str) + pers["sig"]
        wide = (pers.pivot(index=["series", "event"], columns="window",
                           values="cell")
                    .reindex(columns=[f"({a},{b})" for a, b in CAR_WINDOWS]))
        wide.to_csv(OUT / "car_persistence_w50.csv")
        pers.drop(columns=["scale", "value", "cell"]).to_csv(
            OUT / "car_persistence_w50_long.csv", index=False)

    print(f"\nKept {len(kept)} events: {', '.join(kept)}")
    print(f"Saved: fig_car_<event>_w50.(png|pdf), car_persistence_w50.csv")
    if not pers.empty:
        print("\nPersistence preview — USD & portfolios, CAR(%) by horizon "
              "(* p<.10 ** p<.05 *** p<.01):")
        prev = wide.loc[wide.index.get_level_values("series")
                        .isin(["r_DTWEXBGS", "USD_EW", "SAFE", "RISKY",
                               "r_DCOILBRENTEU", "r_DCOILWTICO"])]
        with pd.option_context("display.width", 140):
            print(prev.to_string())
