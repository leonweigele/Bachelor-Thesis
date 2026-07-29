"""
lseg_pull.py — pull the full LSEG block via the Python Data Library, one command.
================================================================================
Mac-friendly alternative to the Excel add-in (which is Windows-only). Connects
to your RUNNING LSEG Workspace desktop app and downloads every series the thesis
needs, writing the SAME files consolidate_lseg.py would produce — so after this
you go straight to get_data.py (no consolidate step).

PREREQUISITES (do once):
  1. LSEG Workspace desktop app OPEN and logged in on this Mac.
  2. pip install lseg-data
  3. An app key for the Data Library:
       - In Workspace, search the app "App Key Generator" (or "APPKEY").
       - Create a key, tick the "Data Library"/"Eikon Data API" use.
       - Put it in a file named  lseg-data.config.json  next to this script:
           {
             "sessions": {
               "default": "desktop.workspace",
               "desktop": { "workspace": { "app-key": "YOUR_APP_KEY_HERE" } }
             }
           }
     (open_session() reads that file automatically.)

RUN:
    python3 "Code/lseg_pull.py"            # full pull
    python3 "Code/lseg_pull.py" --test     # just connect + pull EUR=, print columns

THEN:
    python3 "Code/get_data.py"             # merges the lseg_*.csv written here
    python3 "Code/build_fx_factors.py"     # set FWD_IS_POINTS=True (see note below)
    python3 "Code/02_build_returns.py"
    python3 "Code/03_event_study.py"       # after the EST_WIN=(-140,-21) fix

Outputs written to Data/manual/:
    lseg_fx_spot.csv            32 spot, RAW quotes, bare-ISO headers, MID
    lseg_fx_fwd1m_points.csv    32 x <CCY>1M=  (forward POINTS) -> FWD_IS_POINTS=True
    lseg_gold.csv               column 'Gold'            (XAU= MID)
    lseg_stoxx.csv              column 'EuroStoxx50_EUR'  (.STOXX50E CLOSE)
    lseg_ttf.csv                column 'TTF'              (TRNLTTFMc1 CLOSE)
    lseg_vix.csv / lseg_brent.csv / lseg_spx.csv   optional cross-check backups
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# lseg-data 2.1.1 internally calls .fillna/.ffill in a way pandas will change
# in a future release; pandas warns about LSEG's code, not ours, once per
# instrument (~60x per run), drowning out real errors. Suppress exactly this
# message — all other warnings stay visible. Remove after upgrading lseg-data.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="Downcasting object dtype arrays")

try:
    import lseg.data as ld
except ImportError:
    sys.exit("lseg-data not installed. Run:  pip install lseg-data")

# ----------------------------------------------------------------------------
START = "2019-01-01"
END = "2026-06-30"                      # sample freeze (matches get_data.py)
ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "Data/manual"
MANUAL.mkdir(parents=True, exist_ok=True)

# 32-currency panel (RIC = <ISO>= ). Majors first (USD per FC), rest FC per USD.
SPOT_RICS = [
    "EUR=", "GBP=", "AUD=", "NZD=",
    "JPY=", "CHF=", "CAD=", "SEK=", "NOK=", "DKK=",
    "MXN=", "BRL=", "ZAR=", "TRY=", "INR=", "KRW=", "TWD=", "THB=", "SGD=", "MYR=",
    "IDR=", "COP=", "CLP=", "PEN=", "PLN=", "HUF=", "CZK=", "ILS=", "CNY=", "HKD=",
    "SAR=", "KWD=",
]
# BRL/COP/CLP/PEN have no deliverable onshore 1M forward under <ISO>1M= (they
# trade as NDFs with different RICs), so requesting them only produced
# "universe not found" errors. Skipped explicitly: they drop out of the
# forward-discount sort either way, so the SAFE/RISKY classification is
# unaffected (see build_fx_factors.py / carry_classification.csv).
NDF_ONLY = {"BRL=", "COP=", "CLP=", "PEN="}
FWD_RICS = [r.replace("=", "1M=") for r in SPOT_RICS
            if r not in NDF_ONLY]                        # EUR1M=, JPY1M=, ...

# single series: (RIC, out_file, column_name, price_kind)
SINGLES = [
    ("XAU=",        "lseg_gold.csv",  "Gold",            "mid"),
    (".STOXX50E",   "lseg_stoxx.csv", "EuroStoxx50_EUR", "close"),
    ("TRNLTTFMc1",  "lseg_ttf.csv",   "TTF",             "close"),
    ("LCOc1",       "lseg_brent.csv", "Brent",           "close"),
    # .JPMVXYGL is NOT pulled via API: the API only returns BID/ASK for it, and
    # the ASK side goes stale for weeks (all of Aug 2020 frozen at 8.62 —
    # verified in Workspace 2026-07-23; BID and MID move normally). The series
    # comes from a manual Workspace Price-History export (Mid Price) saved as
    # Data/manual/lseg_vxy.csv. Do not re-add here or the pull would overwrite
    # the good mid-based file with stale-ask data.
]
# .VIX and .SPX are NOT requested: the licence has no permission for CBOE/S&P
# index history (verified 2026-07-23 in the Workspace UI as well, not just the
# API), so no cross-check is possible. FRED VIXCLS / SP500 are the sole source
# for both — documented in the data appendix / sources_log.

MID_COLS = ["BID", "ASK"]
CLOSE_COLS = ["TRDPRC_1", "CLOSE", "OFFCL_CODE"]   # first present wins


def ric_to_iso(ric):
    """EUR= -> EUR ; EUR1M= -> EUR"""
    return ric.replace("1M=", "").replace("=", "")


def fetch(ric):
    """One RIC -> raw daily DataFrame (default columns). None on failure."""
    try:
        df = ld.get_history(universe=ric, interval="daily", start=START, end=END)
    except Exception as e:
        print(f"  ! {ric}: {e}")
        return None
    if df is None or len(df) == 0:
        print(f"  ! {ric}: empty")
        return None
    df = df.sort_index()
    return df


def pick_series(df, kind):
    """Collapse a raw df to one daily series (mid or close)."""
    cols = list(df.columns.astype(str))
    if kind == "mid":
        have = [c for c in MID_COLS if c in cols]
        if len(have) == 2:
            return (df[have[0]].astype(float) + df[have[1]].astype(float)) / 2.0
        if len(have) == 1:
            return df[have[0]].astype(float)
        if "MID_PRICE" in cols:
            return df["MID_PRICE"].astype(float)
    else:
        for c in CLOSE_COLS:
            if c in cols:
                return df[c].astype(float)
    num = df.select_dtypes("number")
    if num.shape[1]:
        print(f"    (using fallback column '{num.columns[0]}')")
        return num.iloc[:, 0].astype(float)
    return None


def build_wide(rics, kind, label):
    """Loop a list of RICs -> wide DataFrame keyed by ISO code."""
    print(f"\n{label}: {len(rics)} instruments")
    out, missing = {}, []
    for ric in rics:
        df = fetch(ric)
        s = pick_series(df, kind) if df is not None else None
        if s is None or s.dropna().empty:
            missing.append(ric)
            continue
        out[ric_to_iso(ric)] = s
        time.sleep(0.1)
    if missing:
        print(f"  missing/failed: {', '.join(missing)}")
    wide = pd.DataFrame(out).sort_index()
    wide.index.name = "date"
    return wide


def save(df, fname):
    path = MANUAL / fname
    df.to_csv(path)
    print(f"  wrote {path.name}  ({df.shape[0]} rows x {df.shape[1]} cols)")


def main():
    test = "--test" in sys.argv
    print(f"lseg-data {getattr(ld, '__version__', '?')}  |  opening session...")
    ld.open_session()
    print("session open.")

    if test:
        df = fetch("EUR=")
        if df is not None:
            print("EUR= columns:", list(df.columns))
            print(df.tail(3))
        ld.close_session()
        return

    save(build_wide(SPOT_RICS, "mid", "Block 1 spot"), "lseg_fx_spot.csv")
    save(build_wide(FWD_RICS, "mid", "Block 2 forwards (points)"),
         "lseg_fx_fwd1m_points.csv")

    print("\nBlock 3/4 single series")
    for ric, fname, col, kind in SINGLES:
        df = fetch(ric)
        s = pick_series(df, kind) if df is not None else None
        if s is None or s.dropna().empty:
            print(f"  ! {ric}: skipped")
            continue
        out = pd.DataFrame({col: s}).sort_index()
        out.index.name = "date"
        save(out, fname)

    ld.close_session()
    print("\nDone. Next:  python3 \"Code/get_data.py\"")
    print("NOTE: forwards are POINTS -> set FWD_IS_POINTS=True in build_fx_factors.py")


if __name__ == "__main__":
    main()
