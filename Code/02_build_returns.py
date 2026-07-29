"""
02_build_returns.py — log returns, currency classification, portfolios,
event windows, and Figure 1.
========================================================================
Runs on whatever get_data.py has downloaded so far; warns about gaps
instead of crashing. Rerun any time new data lands (e.g. LSEG exports).

Inputs:
  Data/processed/daily_panel.csv      (preferred)  — or assembled from Data/raw/
  Data/manual/factors_course.csv      (Project 2; xlsx in disguise)
                                       fallback: Main/Data (old)/data/raw/

Outputs:
  Data/processed/returns_daily.csv     log returns, FC-appreciation convention
  Data/processed/portfolios_daily.csv  safe/risky, oil exp/imp, EW dollar
  Data/processed/classification.csv    which currency landed in which bucket
  Data/processed/events.csv            event dates + trading-day windows
  Output/figures/fig1_overview.png/.pdf

Conventions (state once in Ch. 4):
  - Panel FX levels are FOREIGN PER 1 USD.
  - Returns below are CURRENCY returns vs USD:  r_i = -dln(S_i)
    -> positive = foreign currency APPRECIATES against the dollar.
  - USD_EW = equal-weighted average of dln(S_i) -> positive = USD appreciates.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
FIGS = ROOT / "Output/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# CLASSIFICATION SETTINGS
# ----------------------------------------------------------------------------
# Safe/risky comes from the course data (average interest differential,
# terciles). Override by hand here if Dalgic wants a specific split:
SAFE_OVERRIDE:  list = []          # e.g. ["JPY", "CHF", "EUR", "DKK", "SEK"]
RISKY_OVERRIDE: list = []          # e.g. ["TRY", "BRL", "ZAR", "MXN", "COP", "IDR"]

# Oil classification from net oil trade position (literature-standard):
OIL_EXPORTERS = ["NOK", "CAD", "MXN", "COP", "BRL", "SAR", "KWD"]
OIL_IMPORTERS = ["JPY", "KRW", "INR", "THB", "TWD", "TRY"]

# Named safe-haven baskets (literature-standard), reported ALONGSIDE the
# carry-sorted SAFE/RISKY as a robustness panel in the event study (Ch. 5).
HAVEN_SAFE = ["JPY", "CHF", "EUR"]
HAVEN_RISKY = ["TRY", "BRL", "ZAR", "MXN", "COP", "IDR"]

# Course-file column names -> ISO codes
NAME2ISO = {
    "AUSTRALIAN": "AUD", "BRAZILIAN": "BRL", "BRITISH": "GBP", "UK": "GBP",
    "BULGARIAN": "BGN", "CANADIAN": "CAD", "CHILEAN": "CLP", "CHINESE": "CNY",
    "COLOMBIAN": "COP", "CROATIAN": "HRK", "CZECH": "CZK", "DANISH": "DKK",
    "EURO": "EUR", "HONG KONG": "HKD", "HUNGARIAN": "HUF", "ICELAND": "ISK",
    "INDIAN": "INR", "INDONESIAN": "IDR", "ISRAELI": "ILS", "JAPANESE": "JPY",
    "KOREAN": "KRW", "KUWAITI": "KWD", "MALAYSIAN": "MYR", "MEXICAN": "MXN",
    "NEWZEALAND": "NZD", "NEW ZEALAND": "NZD", "NORWEGIAN": "NOK",
    "PERU": "PEN", "PHILIPPINE": "PHP", "POLISH": "PLN", "ROMANIAN": "RON",
    "RUSSIAN": "RUB", "SAUDI": "SAR", "SINGAPORE": "SGD",
    "SOUTH AFRICAN": "ZAR", "SOUTH": "ZAR", "SWEDISH": "SEK", "SWISS": "CHF",
    "TAIWAN": "TWD", "THAI": "THB", "TURKISH": "TRY",
    "KASAKHSTAN": "KZT", "KENYAN": "KES", "MOROCCAN": "MAD",
    "PAKISTANI": "PKR", "TUNISIAN": "TND",
}

FX_UNIVERSE = sorted(set(NAME2ISO.values()) | {"EUR", "GBP", "AUD", "NZD"})

# ----------------------------------------------------------------------------
# EVENTS  (sub-events indented under their parent in the thesis timeline)
# ----------------------------------------------------------------------------
EVENTS = {
    "ukraine":        ("2022-02-24", "Russia invades Ukraine (non-US control)"),
    "liberation_day": ("2025-04-02", "Liberation Day tariff announcement"),
    "tariff_pause":   ("2025-04-09", "90-day tariff pause"),
    "iran_12day":     ("2025-06-13", "Israel/US strikes on Iran (12-day war)"),
    "hormuz":         ("2026-02-28", "Iran escalation, Hormuz crisis begins"),
    "hormuz_closure": ("2026-03-02", "Iran declares Strait of Hormuz closed"),
    "hormuz_ceasefire": ("2026-04-07", "Two-week ceasefire announced (collapsed 13 Apr)"),
    "us_strikes":     ("2026-05-25", "US strikes on Iran"),
    "ceasefire":      ("2026-06-11", "60-day ceasefire announced"),
}
WINDOW = 20            # trading days on each side for the event windows


# ----------------------------------------------------------------------------
# LOAD PANEL
# ----------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    f = PROC / "daily_panel.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"], index_col="date")
    print("  no daily_panel.csv yet — assembling from Data/raw/")
    parts = []
    for raw in sorted((ROOT / "Data/raw").glob("*.csv")):
        if raw.stem in ("gpr_monthly", "epu_daily", "tpu_daily"):
            continue
        try:
            df = pd.read_csv(raw)
            datecol = next(c for c in df.columns if c.lower() in ("date", "day"))
            df[datecol] = pd.to_datetime(df[datecol])
            parts.append(df.set_index(datecol))
        except Exception as e:
            print(f"  skip {raw.name}: {e}")
    if not parts:
        sys.exit("No data found — run get_data.py first.")
    panel = pd.concat(parts, axis=1).sort_index()
    return panel[panel.index.dayofweek < 5]


# ----------------------------------------------------------------------------
# CLASSIFICATION from course data
# ----------------------------------------------------------------------------
def classify(fx_cols):
    if SAFE_OVERRIDE and RISKY_OVERRIDE:
        return list(SAFE_OVERRIDE), list(RISKY_OVERRIDE), pd.DataFrame()

    # Prefer the REBUILT, up-to-date carry classification (build_fx_factors.py)
    # over the course factor, which is monthly and ends in 2017.
    cc = PROC / "carry_classification.csv"
    if cc.exists():
        c = pd.read_csv(cc)
        safe = [x for x in c.loc[c["bucket"] == "safe", "currency"] if x in fx_cols]
        risky = [x for x in c.loc[c["bucket"] == "risky", "currency"] if x in fx_cols]
        if len(safe) >= 3 and len(risky) >= 3:
            print(f"  classification from rebuilt carry factor "
                  f"(carry_classification.csv): {len(safe)} safe / {len(risky)} risky")
            return safe, risky, c.set_index("currency")
        print("  ! carry_classification.csv present but too few names in panel — "
              "falling back to course factor.")

    for cand in (ROOT / "Data/manual/factors_course.csv",
                 ROOT / "Main/Data (old)/data/raw/factors_course.csv"):
        if cand.exists():
            course = pd.read_excel(cand)   # xlsx wearing a .csv name
            break
    else:
        print("  ! course file missing — using literature fallback classification")
        return (["JPY", "CHF", "EUR", "DKK", "SGD"],
                ["TRY", "BRL", "ZAR", "MXN", "COP", "IDR"], pd.DataFrame())

    course = course.set_index(course.columns[0])

    # Map course currency columns to ISO codes (longest match first, so
    # e.g. 'SOUTH AFRICAN' beats 'SOUTH')
    keys = sorted(NAME2ISO, key=len, reverse=True)
    cur_cols = {}
    for col in course.columns:
        key = next((k for k in keys if k in col.upper()), None)
        if key:
            cur_cols[col] = NAME2ISO[key]

    if "CarryRisk" in course.columns:
        # Proper classification: carry-factor beta per currency
        # (this is the Project 2 / Lustig logic from the thesis notes).
        f = course["CarryRisk"]
        betas = {}
        for col, iso in cur_cols.items():
            pair = pd.concat([course[col], f], axis=1).dropna()
            if len(pair) > 24:
                betas[iso] = (pair.iloc[:, 0].cov(pair.iloc[:, 1])
                              / pair.iloc[:, 1].var())
        rank = pd.Series(betas, name="carry_beta").sort_values()
        metric = "carry beta (low = safe)"
    else:
        rank = pd.Series({iso: course[col].mean()
                          for col, iso in cur_cols.items()},
                         name="avg_level").sort_values()
        metric = "average level (low = safe)"

    print(f"  classification by {metric}:")
    print("   ", ", ".join(f"{c}:{v:.2f}" for c, v in rank.items()))
    n = max(3, len(rank) // 3)
    bottom, top = rank.index[:n].tolist(), rank.index[-n:].tolist()
    if not ({"JPY", "CHF"} & set(bottom)):
        print("  ! WARNING: neither JPY nor CHF in the safe tercile — "
              "verify the course data definition with Dalgic.")
    safe = [c for c in bottom if c in fx_cols]
    risky = [c for c in top if c in fx_cols]
    if len(risky) < 3:
        print(f"  ! only {len(risky)} risky-tercile currencies in the panel so "
              "far — fills out once the LSEG FX (TRY, COP, PLN, ...) arrives.")
    return safe, risky, rank.to_frame()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    panel = load_panel()
    # Forward-fill short gaps (US holidays, staggered closing days) so that
    # returns have no NaN holes: holiday return = 0, the next trading day
    # carries the cumulated move. Longer gaps (publication lags at the end
    # of sample) stay NaN. -> data note for Ch. 4.
    panel = panel.ffill(limit=3)

    # Canonicalise a gold price column to "Gold" so it becomes r_Gold downstream.
    # LSEG exports gold (XAU=) under various column names; map the first match.
    GOLD_ALIASES = ["Gold", "XAU", "XAU=", "GOLD", "gold", "XAUUSD", "Gold_USD"]
    _g = next((c for c in GOLD_ALIASES if c in panel.columns), None)
    if _g and _g != "Gold":
        panel = panel.rename(columns={_g: "Gold"})
        print(f"  gold: mapped '{_g}' -> 'Gold' (becomes r_Gold)")
    elif _g == "Gold":
        print("  gold: 'Gold' level present (becomes r_Gold)")
    else:
        print("  gold: not in panel yet — add Data/manual/lseg_gold.csv (XAU=) "
              "and rerun; Section 5.4 fills automatically")

    print(f"Panel: {panel.shape[0]} days x {panel.shape[1]} series "
          f"({panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d})")

    fx_cols = [c for c in panel.columns if c in FX_UNIVERSE]
    print(f"FX columns found: {len(fx_cols)}"
          + ("" if fx_cols else "  ! none — FX block still missing"))

    # --- returns -------------------------------------------------------------
    rets = pd.DataFrame(index=panel.index)
    for c in fx_cols:                                  # FC appreciation vs USD
        rets[c] = -np.log(panel[c]).diff()
    for c in ("DTWEXBGS", "DTWEXAFEGS", "DTWEXEMEGS", "SP500",
              "DCOILBRENTEU", "DCOILWTICO", "DHHNGSP",
              "Gold", "XAU", "Brent_fut", "WTI_fut", "EuroStoxx50_EUR"):
        if c in panel.columns:                          # guard: WTI < 0 in Apr 2020
            rets[f"r_{c}"] = np.log(panel[c].where(panel[c] > 0)).diff()
    for c in ("DGS10", "DGS5", "DFII5", "T5YIE", "VIXCLS",
              "GPRD", "GPRD_ACT", "GPRD_THREAT", "VXY_Global"):
        if c in panel.columns:
            rets[f"d_{c}"] = panel[c].diff()           # first differences

    # --- EUR-denominated commodities (denomination robustness) ---------------
    # rets["EUR"] = dln(USD per EUR)  (euro appreciation vs USD), so in logs:
    # r(X in EUR) = r(X in USD) - rets["EUR"].  Strips the mechanical
    # dollar-in-the-denominator effect from USD-priced commodities.
    if "EUR" in rets.columns:
        if "r_DCOILBRENTEU" in rets.columns:
            rets["r_Oil_EUR"] = rets["r_DCOILBRENTEU"] - rets["EUR"]
        if "r_Gold" in rets.columns:
            rets["r_Gold_EUR"] = rets["r_Gold"] - rets["EUR"]

    # --- portfolios ----------------------------------------------------------
    ports = pd.DataFrame(index=panel.index)
    if fx_cols:
        ports["USD_EW"] = -rets[fx_cols].mean(axis=1)  # +ve = USD appreciates
        safe, risky, rank = classify(fx_cols)
        print(f"  SAFE : {safe}\n  RISKY: {risky}")
        if safe:
            ports["SAFE"] = rets[[c for c in safe if c in rets]].mean(axis=1)
        if risky:
            ports["RISKY"] = rets[[c for c in risky if c in rets]].mean(axis=1)
        if safe and risky:
            ports["CARRY_HML"] = ports["RISKY"] - ports["SAFE"]
        # Named safe-haven baskets (robustness panel; see HAVEN_* above)
        hs = [c for c in HAVEN_SAFE if c in rets]
        hr = [c for c in HAVEN_RISKY if c in rets]
        if hs:
            ports["HAVEN_SAFE"] = rets[hs].mean(axis=1)
        if hr:
            ports["HAVEN_RISKY"] = rets[hr].mean(axis=1)
        oilx = [c for c in OIL_EXPORTERS if c in rets]
        oilm = [c for c in OIL_IMPORTERS if c in rets]
        if oilx:
            ports["OIL_EXP"] = rets[oilx].mean(axis=1)
        if oilm:
            ports["OIL_IMP"] = rets[oilm].mean(axis=1)
        if not rank.empty:
            rank.to_csv(PROC / "classification.csv")

    # --- events --------------------------------------------------------------
    bdays = panel.index
    rows = []
    for name, (date, desc) in EVENTS.items():
        d = pd.Timestamp(date)
        pos = bdays.searchsorted(d)
        if pos >= len(bdays):
            continue
        lo, hi = max(0, pos - WINDOW), min(len(bdays) - 1, pos + WINDOW)
        rows.append({"event": name, "date": date, "description": desc,
                     "win_start": bdays[lo].date(), "win_end": bdays[hi].date()})
    events = pd.DataFrame(rows)
    events.to_csv(PROC / "events.csv", index=False)

    rets.to_csv(PROC / "returns_daily.csv")
    ports.to_csv(PROC / "portfolios_daily.csv")
    print(f"Saved returns ({rets.shape[1]} series), portfolios "
          f"({ports.shape[1]}), events ({len(events)}).")

    # --- Figure 1 ------------------------------------------------------------
    from thesis_style import (apply_style, style_axis, shade_events,
                              panel_label, save_fig)
    import matplotlib.pyplot as plt
    apply_style()

    def pick(*names):
        return next((n for n in names if n in panel.columns), None)

    spec = [(pick("DTWEXBGS"), "A. Broad U.S. Dollar Index", "Index"),
            (pick("VIXCLS"), "B. VIX", "Index"),
            (pick("DCOILBRENTEU", "Brent_fut"), "C. Brent Crude Oil",
             "USD per barrel"),
            (pick("XAU", "Gold"), "D. Gold", "USD per ounce"),
            (pick("GPRD"), "D. Geopolitical Risk Index (Daily)", "Index")]
    seen, final = set(), []
    for c, lab, yl in spec:                    # gold OR gpr as 4th panel
        if c and lab[0] not in seen:
            final.append((c, lab, yl))
            seen.add(lab[0])
    if not final:
        sys.exit("Nothing to plot yet.")

    main_events = ["ukraine", "liberation_day", "iran_12day", "hormuz"]
    event_dates = [EVENTS[e][0] for e in main_events]

    fig, axes = plt.subplots(len(final), 1, figsize=(6.3, 1.7 * len(final)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, lab, yl) in zip(axes, final):
        ax.plot(panel.index, panel[col], lw=0.9, color="black")
        style_axis(ax, ylabel=yl)
        panel_label(ax, lab)
        shade_events(ax, event_dates, halfwidth_days=15)
    save_fig(fig, FIGS / "fig1_overview")
    print(f"Figure 1 -> Output/figures/fig1_overview.png ({len(final)} panels)")
    print("LaTeX caption suggestion: 'Key variables, 2019-2026. Shaded bands "
          "mark the four main events (Ukraine invasion, Liberation Day, "
          "twelve-day war, Hormuz crisis).'")
