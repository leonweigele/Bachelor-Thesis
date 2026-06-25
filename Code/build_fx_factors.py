"""
build_fx_factors.py — rebuild the Dollar (DOL) and Carry (HML) FX factors
=========================================================================
WHY: the course factors in Data/manual/factors_course.csv are MONTHLY and
END IN JUNE 2017. This script reconstructs the same two factors from spot +
1-month forward exchange rates, through the sample freeze (2026-06-30), and
writes an up-to-date safe/risky classification that 02_build_returns.py
consumes directly (so the daily CARRY_HML portfolio becomes current too).

METHOD  (Lustig-Roussanov-Verdelhan 2011; Menkhoff-Sarno-Schmeling-Schrimpf 2012)
  Convention here: S, F are FOREIGN CURRENCY PER 1 USD (same as daily_panel.csv).
  Forward discount (the interest-rate signal you sort on):
        fd_t      = ln F_t - ln S_t            (high fd = high-yield currency)
  Realised 1-month log excess return of being long the foreign currency:
        rx_{t+1}  = ln F_t - ln S_{t+1}
                  = fd_t + (ln S_t - ln S_{t+1})   = interest diff + FC appreciation
  Each month, sort currencies on fd_t into N portfolios:
        DOL_t   = mean rx over all (floating) currencies        -> "DollarRisk"
        CARRY_t = mean rx top quantile - mean rx bottom quantile -> "CarryRisk"

INPUTS
  Data/processed/daily_panel.csv            spot levels, FC per USD (get_data.py)
  Data/manual/lseg_fx_fwd1m_outright.csv    1M forward OUTRIGHTS (see LSEG_TODO.md)
      - Preferred: outright rates (Datastream WMR "1 month forward rate").
      - If you only have forward POINTS, set FWD_IS_POINTS=True; the script then
        also reads Data/manual/lseg_fx_spot.csv (raw) to form outright = spot +
        points/PIP, before any inversion.
      - Majors EUR/GBP/AUD/NZD are quoted USD per FC by LSEG; with
        MAJORS_USD_PER_FC=True they are inverted to FC per USD to match the panel.

OUTPUTS
  Data/processed/fx_factors_monthly.csv     Date, DOL, CARRY, n, P1..Pk (month-end)
  Data/processed/carry_classification.csv   currency, avg_fd_ann, bucket

RUN ORDER:   get_data.py  ->  build_fx_factors.py  ->  02_build_returns.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
MANUAL = ROOT / "Data/manual"
FREEZE = "2026-06-30"

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Spot panel: prefer the consolidated LSEG file (raw quotes, majors USD per FC);
# fall back to the FC-per-USD daily_panel.csv built by get_data.py.
SPOT_FILE = MANUAL / "lseg_fx_spot.csv"
FWD_FILE = MANUAL / "lseg_fx_fwd1m_points.csv"     # LSEG Workspace forwards = POINTS
FWD_IS_POINTS = True                 # set False if FWD_FILE already holds outrights
RAW_SPOT_FILE = SPOT_FILE            # raw spot used to turn points into outrights

MAJORS = ["EUR", "GBP", "AUD", "NZD"]              # LSEG quotes these USD per FC
MAJORS_USD_PER_FC = True             # invert majors -> FC per USD

N_PORT = 5                           # quintiles; auto-falls back to terciles
# Pegged / managed -> excluded from the carry sort AND the dollar basket:
PEG_EXCLUDE = ["HKD", "DKK", "SAR", "KWD", "CNY"]

# pip divisor for points -> outright (only used if FWD_IS_POINTS). Calibrated per
# currency against realised 2019-2026 rate differentials (JPY/CHF lowest, TRY/BRL/
# MXN highest). LSEG quotes forward points in heterogeneous units:
#   /10000 = standard "pip" pairs, /100 = 2-decimal quotes, /1 = NDF currency units.
PIP = {}
PIP.update({c: 10000.0 for c in [
    "EUR", "GBP", "AUD", "NZD", "CHF", "CAD", "SEK", "NOK", "DKK", "BRL", "ZAR",
    "TRY", "SGD", "MYR", "PLN", "CZK", "ILS", "CNY", "HKD", "SAR", "KWD"]})
PIP.update({c: 100.0 for c in ["JPY", "INR", "KRW", "THB", "HUF"]})
PIP.update({c: 1.0 for c in ["MXN", "IDR", "TWD"]})  # NDF, points in currency units

UNIVERSE = sorted(set(PIP) | set(MAJORS))


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def _read_wide(path: Path) -> pd.DataFrame:
    """Read a wide date x currency file (csv or xlsx) with a 'date' column."""
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(path)
    datecol = next(c for c in df.columns if c.lower() in ("date", "day"))
    df[datecol] = pd.to_datetime(df[datecol], errors="coerce")
    df = df.dropna(subset=[datecol]).set_index(datecol).sort_index()
    df.index.name = "date"
    # keep only numeric currency columns we know about
    keep = [c for c in df.columns if c in UNIVERSE]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def month_end_log(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end level -> natural log, version-robust (no 'M'/'ME' warnings)."""
    g = df.groupby(df.index.to_period("M")).last()
    g.index = g.index.to_timestamp(how="end").normalize()
    return np.log(g)


def load_spot() -> pd.DataFrame:
    """Spot levels in FC per USD. Prefer the consolidated LSEG panel (raw quotes,
    majors inverted here); else the already-inverted daily_panel.csv."""
    if SPOT_FILE.exists():
        df = _read_wide(SPOT_FILE)                  # raw: majors USD per FC
        for c in MAJORS:                            # -> FC per USD
            if c in df.columns:
                df[c] = 1.0 / df[c]
    elif (PROC / "daily_panel.csv").exists():
        df = pd.read_csv(PROC / "daily_panel.csv", parse_dates=["date"],
                         index_col="date")
        df = df[[c for c in df.columns if c in UNIVERSE]]
    else:
        raise SystemExit("No spot found — need Data/manual/lseg_fx_spot.csv or "
                         "Data/processed/daily_panel.csv (run get_data.py first).")
    if df.shape[1] == 0:
        raise SystemExit("No known FX columns in the spot file.")
    return df.loc[:FREEZE]


def load_forward(spot: pd.DataFrame) -> pd.DataFrame:
    if not FWD_FILE.exists():
        raise SystemExit(
            f"Forward file not found: {FWD_FILE.name}\n"
            "Pull the 1M forwards (see Data/manual/LSEG_TODO.md, Block 2) and "
            "save them there, or point FWD_FILE at your file.")
    fwd = _read_wide(FWD_FILE)

    if FWD_IS_POINTS:
        # outright = spot + points / PIP, in the RAW quote convention,
        # so we need the raw (un-inverted) spot panel for this step.
        raw_spot = _read_wide(RAW_SPOT_FILE) if RAW_SPOT_FILE.exists() else None
        if raw_spot is None:
            raise SystemExit("FWD_IS_POINTS=True needs Data/manual/lseg_fx_spot.csv "
                             "(raw spot, same convention as the points).")
        out = pd.DataFrame(index=fwd.index)
        for c in fwd.columns:
            if c in raw_spot.columns and c in PIP:
                s = raw_spot[c].reindex(fwd.index).ffill()
                out[c] = s + fwd[c] / PIP[c]
        fwd = out

    if MAJORS_USD_PER_FC:             # USD per FC -> FC per USD
        for c in MAJORS:
            if c in fwd.columns:
                fwd[c] = 1.0 / fwd[c]

    return fwd.loc[:FREEZE]


# ----------------------------------------------------------------------------
# FACTORS
# ----------------------------------------------------------------------------
def build():
    spot = load_spot()
    fwd = load_forward(spot)
    common = [c for c in spot.columns if c in fwd.columns]
    if len(common) < 6:
        raise SystemExit(f"Only {len(common)} currencies have BOTH spot and "
                         "forward — need >=6. Check the forward file columns.")
    print(f"Currencies with spot + forward: {len(common)}  ({', '.join(common)})")

    s_m = month_end_log(spot[common])
    f_m = month_end_log(fwd[common])

    fd = f_m - s_m                    # forward discount (signal), month-end t
    sig = fd.shift(1)                 # signal known at the START of each month
    ret = f_m.shift(1) - s_m          # excess return realised over the month

    rows, used_n = [], []
    for t in ret.index:
        sg = sig.loc[t].dropna()
        rr = ret.loc[t].dropna()
        names = [c for c in sg.index if c in rr.index and c not in PEG_EXCLUDE]
        if len(names) < 6:
            continue
        dol = rr[names].mean()
        k = N_PORT if len(names) >= 2 * N_PORT else 3
        order = sg[names].sort_values()
        q = pd.qcut(order.rank(method="first"), k, labels=False)
        low, high = order.index[q == 0], order.index[q == k - 1]
        row = {"DOL": dol, "CARRY": rr[high].mean() - rr[low].mean(),
               "n": len(names)}
        for p in range(k):
            row[f"P{p + 1}"] = rr[order.index[q == p]].mean()
        rows.append(pd.Series(row, name=t))
        used_n.append(len(names))

    F = pd.DataFrame(rows)
    F.index.name = "date"
    F = F.loc[:FREEZE]

    # --- classification: average forward discount over the sample -----------
    avg_fd = fd[[c for c in common if c not in PEG_EXCLUDE]].mean().dropna()
    avg_fd = avg_fd.sort_values()
    n = max(3, len(avg_fd) // 3)
    safe = avg_fd.index[:n].tolist()          # lowest yield  = safe / funding
    risky = avg_fd.index[-n:].tolist()         # highest yield = risky / carry
    mid = [c for c in avg_fd.index if c not in safe + risky]

    classif = pd.DataFrame({"avg_fd_ann": avg_fd * 12.0})  # ~annualised
    classif["bucket"] = ["safe" if c in safe else "risky" if c in risky else "mid"
                         for c in classif.index]
    classif.index.name = "currency"

    # --- sanity check (catches a flipped quote convention) ------------------
    if not ({"JPY", "CHF"} & set(safe)):
        print("  ! WARNING: neither JPY nor CHF landed in the safe (low-yield) "
              "bucket — likely a forward/spot quote-convention mismatch. "
              "Check MAJORS_USD_PER_FC and the forward file direction.")
    else:
        print("  sanity ok: JPY/CHF in the low-yield (safe) bucket.")

    PROC.mkdir(parents=True, exist_ok=True)
    F.round(6).to_csv(PROC / "fx_factors_monthly.csv")
    classif.round(4).to_csv(PROC / "carry_classification.csv")
    ret.loc[:FREEZE].round(6).to_csv(PROC / "fx_excess_returns_monthly.csv")

    print(f"\nMonths built: {len(F)}  ({F.index.min():%Y-%m} -> "
          f"{F.index.max():%Y-%m}),  portfolios/month median = "
          f"{int(np.median(used_n)) if used_n else 0}")
    ann = lambda x: x.mean() * 12
    if len(F):
        print(f"  DOL   mean {ann(F['DOL']):+.2%} p.a.   "
              f"CARRY mean {ann(F['CARRY']):+.2%} p.a.")
    print(f"  SAFE : {safe}")
    print(f"  RISKY: {risky}")
    print("\nWrote:")
    print("  Data/processed/fx_factors_monthly.csv")
    print("  Data/processed/carry_classification.csv")
    print("\nNext: run 02_build_returns.py — it now picks up "
          "carry_classification.csv automatically.")


if __name__ == "__main__":
    build()
