"""
es_common.py — shared event-study constants and helpers.
========================================================
WHY THIS MODULE EXISTS. 03_event_study.py, 04_event_study_w50.py and
05_cross_event_tests.py each hard-code the estimation windows, the event
list and the constant-mean machinery independently — CODE_AUDIT.md
(2026-08-20) flags the duplication. New code (starting with
06_make_thesis_figures.py) imports the shared pieces from here instead of
adding a further copy.

The three existing scripts DELIBERATELY still read their own copies: they
are verified as-is against the pinned result CSVs, and this module must
never drift from them. Every constant below is copied verbatim from the
script named beside it. If 03/04/05 ever change, change this file in the
same commit and re-run the verification in 06_make_thesis_figures.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
ES_OUT = PROC / "event_study"

# ---- windows ---------------------------------------------------------------
EST_WIN_MAIN = (-140, -21)   # 03_event_study.py:36 / 05_cross_event_tests.py:37
EVT_WIN_MAIN = (-20, 20)     # 03_event_study.py:37
EST_WIN_W50 = (-170, -51)    # 04_event_study_w50.py:47
EVT_WIN_W50 = (-50, 50)      # 04_event_study_w50.py:48
MIN_EST_DAYS = 60            # 03_event_study.py run_event(): skip below this


def stars(t):
    """Identical in 03/04/05: 10/5/1% two-sided normal thresholds."""
    if t is None or not np.isfinite(t):
        return ""
    a = abs(t)
    return "***" if a > 2.58 else "**" if a > 1.96 else "*" if a > 1.65 else ""


def load_data():
    """returns_daily + portfolios_daily, concatenated as in 03/05 load()."""
    rets = pd.read_csv(PROC / "returns_daily.csv",
                       parse_dates=["date"], index_col="date")
    ports = pd.read_csv(PROC / "portfolios_daily.csv",
                        parse_dates=["date"], index_col="date")
    return pd.concat([rets, ports], axis=1)


def load_events():
    """events.csv as a DataFrame plus an {event: date} dict."""
    ev = pd.read_csv(PROC / "events.csv")
    return ev, dict(zip(ev["event"], ev["date"]))


def rel_day(index, base_date, other_date):
    """Trading-day offset of other_date relative to base_date's day 0
    (03_event_study.py rel_day(): searchsorted maps weekend dates to the
    next trading day, e.g. Sat 2026-02-28 -> Mon 2026-03-02)."""
    return int(index.searchsorted(pd.Timestamp(other_date))
               - index.searchsorted(pd.Timestamp(base_date)))


def const_mean_event(data, series, date,
                     est_win=EST_WIN_MAIN, evt_win=EVT_WIN_MAIN):
    """Constant-mean abnormal returns for one series around one event,
    replicating 03_event_study.py (and 04 for the wide windows) operation
    for operation so the resulting floats are bit-identical to the pinned
    car_paths_<event>.csv columns.

    Returns dict(mu, sigma, n_est, ar, car) with ar/car indexed by relative
    trading day, or None when the event cannot be run (as in 03).
    """
    idx = data.index
    pos = idx.searchsorted(pd.Timestamp(date))
    est_idx = idx[max(0, pos + est_win[0]): max(0, pos + est_win[1] + 1)]
    evt_lo = pos + evt_win[0]
    if evt_lo < 0 or pos >= len(idx):
        return None
    evt_idx = idx[evt_lo: min(len(idx), pos + evt_win[1] + 1)]
    rel = np.arange(evt_win[0], evt_win[0] + len(evt_idx))

    est = data.loc[est_idx, series].dropna()
    evt = data.loc[evt_idx, series]
    if len(est) < MIN_EST_DAYS or evt.isna().all():
        return None
    mu, sig = est.mean(), est.std()
    ar = evt - mu
    car = ar.cumsum()
    return {"mu": mu, "sigma": sig, "n_est": len(est),
            "ar": pd.Series(ar.values, index=rel),
            "car": pd.Series(car.values, index=rel)}


def ensure_latin_modern():
    """Make matplotlib render in Latin Modern Roman, the thesis body font.

    matplotlib does not scan TeX Live's font tree, so on a machine where
    Latin Modern exists only as part of a TeX installation `font.serif`
    silently falls through to DejaVu Serif — which embeds a visibly
    different face than the figures already in the thesis (LMRoman).
    Register the OTFs from TeX Live when they are present; warn loudly
    when they are not, because the figures would then not match.
    """
    from matplotlib import font_manager

    def resolved():
        return Path(font_manager.findfont(
            font_manager.FontProperties(family=["Latin Modern Roman"]),
            fallback_to_default=True)).name

    if "lmroman" in resolved().lower():
        return
    for base in sorted(Path("/usr/local/texlive").glob("*/texmf-dist/fonts/"
                                                      "opentype/public/lm")):
        for face in ("lmroman10-regular.otf", "lmroman10-bold.otf",
                     "lmroman10-italic.otf"):
            if (base / face).exists():
                font_manager.fontManager.addfont(str(base / face))
    name = resolved()
    if "lmroman" in name.lower():
        print(f"  Latin Modern registered from TeX Live ({name})")
    else:
        print(f"  ! Latin Modern NOT found — matplotlib will use {name}. "
              f"Figures will not match the rest of the thesis; do not "
              f"--install this run.")
