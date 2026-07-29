"""
03_event_study.py — abnormal returns and CARs around each event.
================================================================
Method (per the research design):
  - Main: constant-mean abnormal returns.
      estimation window [-140, -21] trading days, event window [-20, +20]
      AR_t  = r_t - mean(r, estimation window)
      CAR   = sum of ARs;  t = CAR / (sigma_est * sqrt(L))
  - Appendix robustness: market-model ARs for currencies,
      r_i = a + b * USD_EW + e  estimated over the same window.

Inputs:  Data/processed/returns_daily.csv, portfolios_daily.csv, events.csv
Outputs: Data/processed/event_study/
           car_paths_<event>.csv      CAR paths, all series, both methods
           car_summary.csv            CARs + t-stats for standard windows
           cross_event_table.csv      key series x events, CAR(0,+10)
         Output/figures/fig_car_<event>.png      portfolio CAR paths
         Output/figures/fig_cross_event_usd.png  USD response, all events

Rerun whenever the data updates — everything regenerates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
OUT = PROC / "event_study"
FIGS = ROOT / "Output/figures/ch06_results"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

EST_WIN = (-140, -21)        # estimation window (trading days rel. to event); ends at -21 so it does not overlap the [-20,+20] event window
EVT_WIN = (-20, 20)          # event window
CAR_WINDOWS = [(-1, 1), (0, 1), (0, 5), (0, 10), (0, 20)]
MAIN_EVENTS = ["ukraine", "liberation_day", "iran_12day", "hormuz"]

# Series shown in the portfolio CAR figures (if available)
FIG_SERIES = ["USD_EW", "SAFE", "RISKY", "OIL_EXP", "OIL_IMP"]
# Key series for the cross-event comparison table
KEY_SERIES = ["r_DTWEXBGS", "USD_EW", "SAFE", "RISKY", "CARRY_HML",
              "OIL_EXP", "OIL_IMP", "r_DCOILBRENTEU", "r_DCOILWTICO", "r_SP500",
              "r_Gold", "r_XAU", "r_Oil_EUR", "r_Gold_EUR",
              "d_VIXCLS", "d_DGS10"]


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
    """Positions of estimation and event windows around the event day."""
    pos = index.searchsorted(pd.Timestamp(event_date))
    est = index[max(0, pos + EST_WIN[0]): max(0, pos + EST_WIN[1] + 1)]
    evt_lo, evt_hi = pos + EVT_WIN[0], pos + EVT_WIN[1]
    if evt_lo < 0 or pos >= len(index):
        return None, None, None
    evt = index[evt_lo: min(len(index), evt_hi + 1)]
    rel = np.arange(EVT_WIN[0], EVT_WIN[0] + len(evt))
    return est, evt, rel


def run_event(data, name, date):
    est_idx, evt_idx, rel = event_slices(data.index, date)
    if est_idx is None or len(est_idx) < 60:
        print(f"  !!  {name}: not enough history — skipped")
        return None, []

    paths = {}
    rows = []
    usd = data["USD_EW"] if "USD_EW" in data.columns else None

    for col in data.columns:
        est = data.loc[est_idx, col].dropna()
        evt = data.loc[evt_idx, col]
        if len(est) < 60 or evt.isna().all():
            continue

        # ---- constant-mean ----------------------------------------------
        mu, sig = est.mean(), est.std()
        ar = (evt - mu)
        car = ar.cumsum()
        paths[(col, "cm")] = pd.Series(car.values, index=rel[:len(car)])
        for (a, b) in CAR_WINDOWS:
            m = (rel >= a) & (rel <= b)
            c = ar.iloc[m[:len(ar)]].sum()
            L = int(m[:len(ar)].sum())
            t = c / (sig * np.sqrt(L)) if sig > 0 and L > 0 else np.nan
            rows.append({"event": name, "series": col, "method": "const_mean",
                         "window": f"({a},{b})", "CAR": c, "t": t,
                         "sig": stars(t)})

        # ---- market model (currencies & portfolios only, vs USD_EW) ------
        if usd is not None and col != "USD_EW" and not col.startswith(("d_", "r_")):
            pair = pd.concat([data[col], usd], axis=1).loc[est_idx].dropna()
            if len(pair) >= 60:
                y, x = pair.iloc[:, 0], pair.iloc[:, 1]
                beta = y.cov(x) / x.var()
                alpha = y.mean() - beta * x.mean()
                resid_sig = (y - alpha - beta * x).std()
                ar_mm = evt - (alpha + beta * data.loc[evt_idx, "USD_EW"])
                paths[(col, "mm")] = pd.Series(ar_mm.cumsum().values,
                                               index=rel[:len(ar_mm)])
                for (a, b) in CAR_WINDOWS:
                    m = (rel >= a) & (rel <= b)
                    c = ar_mm.iloc[m[:len(ar_mm)]].sum()
                    L = int(m[:len(ar_mm)].sum())
                    t = (c / (resid_sig * np.sqrt(L))
                         if resid_sig > 0 and L > 0 else np.nan)
                    rows.append({"event": name, "series": col,
                                 "method": "market_model",
                                 "window": f"({a},{b})", "CAR": c, "t": t,
                                 "sig": stars(t)})

    path_df = pd.DataFrame({f"{c}__{m}": s for (c, m), s in paths.items()})
    path_df.index.name = "rel_day"
    path_df.to_csv(OUT / f"car_paths_{name}.csv")
    return path_df, rows


from thesis_style import (apply_style, style_axis, legend_below,
                          panel_label, save_fig, PALETTE)
import matplotlib.pyplot as plt
apply_style()

NICE = {"USD_EW": "Dollar (EW index)", "SAFE": "Safe currencies",
        "RISKY": "Risky currencies", "OIL_EXP": "Oil exporters",
        "OIL_IMP": "Oil importers", "r_DTWEXBGS": "Broad USD index",
        "CARRY_HML": "Carry (risky−safe)"}
EVENT_NICE = {"ukraine": "Ukraine invasion (Feb 2022)",
              "liberation_day": "Liberation Day (Apr 2025)",
              "iran_12day": "Twelve-day war (Jun 2025)",
              "hormuz": "Hormuz crisis (Feb 2026)"}
STYLES = {  # color, linestyle — headline black solid, USD dashed black
    "USD_EW": ("black", "--"), "SAFE": (PALETTE[2], "-"),
    "RISKY": (PALETTE[1], "-"), "OIL_EXP": (PALETTE[3], "-"),
    "OIL_IMP": (PALETTE[4], "-")}


# Two stacked panels (small multiples) instead of one 5-line "spaghetti
# chart" — Schwabish (2014, JEP 28(1)), citing Tufte (2006). Panel A =
# safe-haven hierarchy, Panel B = oil channel with Brent (only) on the right
# axis: Brent is the world benchmark (~2/3 of internationally traded crude,
# EIA); the WTI-Brent spread is US-local Cushing noise (Fattouh 2011).
PANEL_A = ["USD_EW", "SAFE", "RISKY"]
PANEL_B = ["OIL_EXP", "OIL_IMP"]
BRENT = "r_DCOILBRENTEU"
SHORT = {"ukraine": "the Ukraine invasion", "liberation_day": "Liberation Day",
         "tariff_pause": "the tariff pause", "iran_12day": "the 12-day war",
         "hormuz": "the Hormuz crisis", "hormuz_closure": "the Hormuz closure",
         "us_strikes": "the US strikes", "ceasefire": "the ceasefire"}


def figure_event(name, paths):
    cols_a = [c for c in PANEL_A if f"{c}__cm" in paths.columns]
    cols_b = [c for c in PANEL_B if f"{c}__cm" in paths.columns]
    if not cols_a and not cols_b:
        return
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    for ax, cols in ((axA, cols_a), (axB, cols_b)):
        for c in cols:
            color, ls = STYLES.get(c, (PALETTE[5], "-"))
            ax.plot(paths.index, 100 * paths[f"{c}__cm"], lw=1.1,
                    color=color, ls=ls, label=NICE.get(c, c))
        ax.axvline(0, color="#999999", lw=0.7)
        ax.axhline(0, color="#999999", lw=0.5)

    axB2 = None
    if f"{BRENT}__cm" in paths.columns:
        axB2 = axB.twinx()
        axB2.plot(paths.index, 100 * paths[f"{BRENT}__cm"], lw=1.0,
                  color="#8c6d31", ls="-.", label="Brent crude (rhs)")
        axB2.set_ylabel("Brent CAR (%)")
        axB2.spines["top"].set_visible(False)
        axB2.spines["right"].set_visible(True)
        axB2.spines["right"].set_linewidth(0.6)
        axB2.grid(False)
        axB2.yaxis.set_ticks_position("right")
        axB2.tick_params(left=False, right=True, labelright=True,
                         width=0.6, length=3)
        axB2.margins(x=0.01)

    # No in-figure panel titles — panels are described in the LaTeX caption
    # (top = currency portfolios, bottom = oil-exposure portfolios + Brent).
    axB.set_xlabel(f"Trading days around {SHORT.get(name, name)}")
    style_axis(axA, ylabel="CAR (%)")
    style_axis(axB, ylabel="CAR (%)")
    hA, lA = axA.get_legend_handles_labels()
    hB, lB = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels() if axB2 else ([], [])
    legend_below(fig, handles=hA + hB + h2, labels=lA + lB + l2, ncol=3, y=0.0)
    save_fig(fig, FIGS / f"fig_car_{name}", legend_pad=0.05)


# Sub-events marked on their parent's CAR path (events are NOT point events;
# the formal cross-event comparison uses short windows CAR(0,+1)/(0,+5)).
SUB_EVENTS = {"liberation_day": [("tariff_pause", "pause")],
              "hormuz": [("hormuz_closure", "closure")]}
CROSS_WIN = (-5, 15)        # shorter window for the comparison figure


def rel_day(index, parent_date, sub_date):
    return int(index.searchsorted(pd.Timestamp(sub_date))
               - index.searchsorted(pd.Timestamp(parent_date)))


def figure_cross_event(all_paths, data_index, events):
    series = "r_DTWEXBGS__cm" if any("r_DTWEXBGS__cm" in p.columns
                                     for p in all_paths.values()) else "USD_EW__cm"
    ev_dates = dict(zip(events["event"], events["date"]))
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    styles = [("black", "-"), ("black", "--"), (PALETTE[1], "-"),
              (PALETTE[2], "-")]
    for (name, paths), (color, ls) in zip(
            [(n, p) for n, p in all_paths.items() if n in MAIN_EVENTS], styles):
        if series not in paths.columns:
            continue
        p = paths[series].loc[CROSS_WIN[0]:CROSS_WIN[1]]
        p = p - p.loc[:0].iloc[-1]              # re-anchor: CAR = 0 on day 0
        ax.plot(p.index, 100 * p, lw=1.1, color=color, ls=ls,
                label=EVENT_NICE.get(name, name))
        for sub, sublab in SUB_EVENTS.get(name, []):
            if sub in ev_dates:
                rd = rel_day(data_index, ev_dates[name], ev_dates[sub])
                if CROSS_WIN[0] <= rd <= CROSS_WIN[1] and rd in p.index:
                    ax.plot(rd, 100 * p.loc[rd], marker="o", ms=4,
                            color=color, zorder=5)
                    ax.annotate(sublab, xy=(rd, 100 * p.loc[rd]),
                                xytext=(3, 5), textcoords="offset points",
                                fontsize=6.5, color=color)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xlabel("Trading days relative to event")
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    style_axis(ax, ylabel="USD CAR (%), anchored at event day")
    legend_below(fig, ncol=2, y=-0.04)
    save_fig(fig, FIGS / "fig_cross_event_usd", legend_pad=0.12)


if __name__ == "__main__":
    data, events = load()
    print(f"Loaded {data.shape[1]} series, {len(events)} events")

    all_rows, all_paths = [], {}
    for _, ev in events.iterrows():
        res = run_event(data, ev["event"], ev["date"])
        if res[0] is None:
            continue
        paths, rows = res
        all_paths[ev["event"]] = paths
        all_rows += rows
        figure_event(ev["event"], paths)
        print(f"  OK  {ev['event']:16s} ({ev['date']})  "
              f"{paths.shape[1]} CAR paths")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT / "car_summary.csv", index=False)

    # cross-event table: key series x events, CAR(0,+10), constant-mean
    tbl = (summary.query("method == 'const_mean' and window == '(0,10)'")
                  .loc[lambda d: d["series"].isin(KEY_SERIES)]
                  .assign(scale=lambda d: np.where(
                              d["series"].str.startswith("d_"), 1.0, 100.0),
                          cell=lambda d: (d["scale"] * d["CAR"]).round(2)
                          .astype(str) + d["sig"])
                  .pivot(index="series", columns="event", values="cell")
                  .reindex([s for s in KEY_SERIES]))
    tbl.to_csv(OUT / "cross_event_table.csv")
    figure_cross_event(all_paths, data.index, events)

    print(f"\nSaved: car_summary.csv ({len(summary)} rows), "
          "cross_event_table.csv, figures.")
    print("\nCross-event preview — CAR(0,+10), % (const-mean, "
          "* p<.10 ** p<.05 *** p<.01):")
    with pd.option_context("display.width", 140):
        print(tbl.dropna(how="all").to_string())
