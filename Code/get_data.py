"""
get_data.py — downloads ALL publicly available thesis data in one call.
=======================================================================
Data policy: FRED first, LSEG second, no Yahoo Finance.

  - FRED: macro block (USD indices, yields, TIPS, VIX, oil, gas, S&P 500)
          + ~21 daily bilateral FX rates (H.10 release)
  - LSEG (from Monday): everything FRED lacks — see Data/manual/LSEG_TODO.md
          which this script (re)generates. Export from LSEG as CSV/XLSX into
          Data/manual/ with a 'date' column; files named lseg_*.csv|xlsx are
          picked up and merged automatically on the next run.
  - GPR / EPU / TPU indices: auto-download from the authors' sites.
  - Course data (Project 2/3): copied over from Main/Data (old)/.

Run:   pip install pandas requests openpyxl xlrd
       python3 "Code/get_data.py"

FRED routes: fredgraph.csv (no key) with fallback to the official API —
get a free key at https://fred.stlouisfed.org/docs/api/api_key.html and
   export FRED_API_KEY=yourkey

Output:
  Data/raw/         one file per source, untouched
  Data/manual/      LSEG exports, course data, manual downloads
  Data/processed/   daily_panel.csv  (merged levels, business days)
  Data/sources_log.csv   (series, source, retrieval time -> Appendix C)
"""

import os
import shutil
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
START = "2019-01-01"
FREEZE = "2026-06-30"      # sample freeze — cite in Ch. 4; change once, rerun

ROOT = Path(__file__).resolve().parent.parent          # thesis folder
RAW, MANUAL, PROC = ROOT / "Data/raw", ROOT / "Data/manual", ROOT / "Data/processed"
OLD_RAW = ROOT / "Main/Data (old)/data/raw"            # old artefacts to reuse
for p in (RAW, MANUAL, PROC):
    p.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")      # or paste key here

# --- FRED macro block --------------------------------------------------------
FRED_SERIES = {
    "DTWEXBGS":     "Nominal broad USD index",
    "DTWEXAFEGS":   "Nominal advanced-economies USD index",
    "DTWEXEMEGS":   "Nominal emerging-markets USD index",
    "DGS10":        "10y Treasury constant-maturity yield",
    "DGS5":         "5y Treasury constant-maturity yield",
    "DFII5":        "5y TIPS yield",
    "T5YIE":        "5y breakeven inflation",
    "VIXCLS":       "CBOE VIX",
    "DCOILBRENTEU": "Brent crude spot",
    "DCOILWTICO":   "WTI crude spot",
    "DHHNGSP":      "Henry Hub natural gas spot",
    "SP500":        "S&P 500 index",
}

# --- FRED daily FX (H.10 release) -------------------------------------------
# code -> (label, invert). H.10 mixes quoting conventions; invert=True where
# FRED quotes USD per foreign unit. After inversion EVERYTHING is
# FOREIGN PER 1 USD (positive return = USD appreciates) — state in Ch. 4.
FRED_FX = {
    "DEXUSEU": ("EUR", True),  "DEXUSUK": ("GBP", True),
    "DEXUSAL": ("AUD", True),  "DEXUSNZ": ("NZD", True),
    "DEXJPUS": ("JPY", False), "DEXSZUS": ("CHF", False),
    "DEXCAUS": ("CAD", False), "DEXSDUS": ("SEK", False),
    "DEXNOUS": ("NOK", False), "DEXDNUS": ("DKK", False),
    "DEXMXUS": ("MXN", False), "DEXBZUS": ("BRL", False),
    "DEXSFUS": ("ZAR", False), "DEXINUS": ("INR", False),
    "DEXKOUS": ("KRW", False), "DEXTAUS": ("TWD", False),
    "DEXTHUS": ("THB", False), "DEXSIUS": ("SGD", False),
    "DEXMAUS": ("MYR", False), "DEXCHUS": ("CNY", False),
    "DEXHKUS": ("HKD", False),
}

# --- What FRED lacks -> LSEG on Monday ---------------------------------------
LSEG_TODO = """\
# LSEG / DATASTREAM DOWNLOAD LIST  (clean, full re-download)
# =========================================================
# Settings for EVERY request:
#   - Daily frequency, 2019-01-01 to 2026-06-30  (sample freeze)
#   - Export to Excel/CSV with a 'date' column
#   - Save into Data/manual/ named  lseg_<block>.csv  (get_data.py auto-merges)
#   - Source: WM/Refinitiv (WMR), datatype = Exchange Rate Middle (ER) where it
#     applies -> one consistent, citable source; FRED stays as a cross-check
#   - FX: NO currency conversion (the RIC fixes the quote). Assets: native ccy.
#
# ------------------------------------------------------------------
# BLOCK 1 — FX SPOT, full 32-currency panel          -> lseg_fx_spot.csv
# ------------------------------------------------------------------
# Majors, quoted USD per 1 FC:
#     EUR=  GBP=  AUD=  NZD=
# Quoted FC per 1 USD:
#     JPY=  CHF=  CAD=  SEK=  NOK=  DKK=
#     MXN=  BRL=  ZAR=  TRY=  INR=  KRW=  TWD=  THB=  SGD=  MYR=
#     IDR=  COP=  CLP=  PEN=  PLN=  HUF=  CZK=  ILS=  CNY=  HKD=
# Pegs / context:
#     SAR=  KWD=
#
# ------------------------------------------------------------------
# BLOCK 2 — 1M FX FORWARDS, same panel  -> lseg_fx_fwd1m_outright.csv
# ------------------------------------------------------------------
# Feeds build_fx_factors.py, which rebuilds the Dollar (DOL) + Carry (HML)
# factors through 2026 (the course factors are monthly and end in 2017).
# Forward discount = interest differential via CIP (Lustig-Roussanov-
# Verdelhan 2011).
#   PREFERRED — Datastream OUTRIGHT: search "<CURRENCY> 1 MONTH FORWARD"
#     (WMR or Barclays), datatype ER (middle). Gives the rate directly.
#     Save as  lseg_fx_fwd1m_outright.csv  (script default).
#   FALLBACK — Workspace points: RIC  <CCY>1M=  returns forward POINTS, not
#     the outright (outright = spot + points / 10^k; k=2 for JPY, else 4).
#     Save as  lseg_fx_fwd1m_points.csv  and set FWD_IS_POINTS=True in
#     build_fx_factors.py.
#   Same 32 currencies as Block 1. DKK, HKD, SAR, KWD, CNY are pegged/managed
#   -> pull them, but build_fx_factors.py drops them from the carry sort.
#
# ------------------------------------------------------------------
# BLOCK 3 — series NOT on FRED  (one file each, native currency)
# ------------------------------------------------------------------
#   XAU=         Gold spot, USD/oz                  -> lseg_gold.csv   (USD)
#   .STOXX50E    Euro Stoxx 50 price index          -> lseg_stoxx.csv  (EUR, not USD)
#   TRNLTTFMc1   TTF nat-gas front month, EUR/MWh   -> lseg_ttf.csv    (EUR, not USD)
#   # Native quote on purpose: keeps the EUR asset return free of the
#   # EUR/USD leg, which is analysed separately as the dollar.
#
# ------------------------------------------------------------------
# BLOCK 4 — optional backups (cross-check FRED, cheap while you're there)
# ------------------------------------------------------------------
#   .VIX   CBOE VIX           (cross-check FRED VIXCLS)
#   LCOc1  Brent front month  (cross-check FRED Brent spot)
#   .SPX   S&P 500            (cross-check FRED SP500)
"""

# GPR / EPU / TPU
GPR_URLS = {
    "gpr_daily":   ["https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
                    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xlsx"],
    "gpr_monthly": ["https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
                    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx"],
}
EPU_DAILY_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
TPU_URLS = ["https://www.matteoiacoviello.com/tpu_files/tpu_daily_data.csv",
            "https://www.matteoiacoviello.com/tpu_files/tpu_web_daily.xlsx"]

CARRY_OVER = ["factors_course.csv", "liberation_day_data.xlsx", "epu_categorical.xlsx"]

HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36")}
LOG = []

# Force IPv4 — fixes the classic "works in browser, Python read-timeout"
# problem caused by broken IPv6 routes (browsers fall back automatically,
# Python's requests does not).
import urllib3.util.connection as _conn
import socket as _socket
_conn.allowed_gai_family = lambda: _socket.AF_INET


def stamp(series, desc, source):
    LOG.append({"series": series, "description": desc, "source": source,
                "retrieved": datetime.now().strftime("%Y-%m-%d %H:%M")})


def header(txt):
    print(f"\n{'=' * 64}\n{txt}\n{'=' * 64}")


def http_get(url, tries=3, timeout=60):
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


# ----------------------------------------------------------------------------
# FRED (two routes)
# ----------------------------------------------------------------------------
FRED_GRAPH_OK = None     # set once by probe_fred()


def probe_fred():
    """One quick test request so 33 series don't each wait out a timeout."""
    global FRED_GRAPH_OK
    try:
        http_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
                 tries=1, timeout=15)
        FRED_GRAPH_OK = True
        print("  fredgraph reachable.")
    except Exception:
        FRED_GRAPH_OK = False
        print("  fredgraph NOT reachable" +
              (" — using API route." if FRED_API_KEY else
               " and no FRED_API_KEY set — skipping FRED entirely. "
               "Get a free key: fred.stlouisfed.org/docs/api/api_key.html"))


def fred_via_graph(code):
    if not FRED_GRAPH_OK:
        raise ConnectionError("fredgraph unreachable (probe failed)")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = http_get(url, tries=1, timeout=30)
    return (pd.read_csv(BytesIO(r.content),
                        parse_dates=["observation_date"], na_values=".")
              .rename(columns={"observation_date": "date"})
              .set_index("date")[code])


def fred_via_api(code):
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={code}&api_key={FRED_API_KEY}&file_type=json"
           f"&observation_start={START}&observation_end={FREEZE}")
    obs = http_get(url, tries=2).json()["observations"]
    s = pd.Series({o["date"]: o["value"] for o in obs}, name=code)
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return pd.to_numeric(s, errors="coerce")


def fred_series(code):
    try:
        return fred_via_graph(code), "fredgraph"
    except Exception:
        if FRED_API_KEY:
            return fred_via_api(code), "API"
        raise


def get_fred_macro():
    header("1/6  FRED — macro block")
    probe_fred()
    if not FRED_GRAPH_OK and not FRED_API_KEY:
        return pd.DataFrame()
    frames = []
    for code, desc in FRED_SERIES.items():
        try:
            s, route = fred_series(code)
            frames.append(s)
            stamp(code, desc, f"FRED ({route})")
            print(f"  OK  {code:14s} {desc} [{route}]")
        except Exception as e:
            print(f"  !!  {code:14s} FAILED: {e}")
    if not frames:
        print("  !!  FRED unreachable — rerun later (set FRED_API_KEY for fallback).")
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).loc[START:FREEZE]
    df.to_csv(RAW / "fred_macro.csv")
    return df


def get_fred_fx():
    header("2/6  FRED — daily FX (H.10)")
    if not FRED_GRAPH_OK and not FRED_API_KEY:
        print("  --  skipped (FRED unreachable, no API key)")
        return pd.DataFrame()
    series = {}
    for code, (label, invert) in FRED_FX.items():
        try:
            s, route = fred_series(code)
            series[label] = (1.0 / s) if invert else s
            stamp(label, f"FX, foreign per USD (FRED {code})", f"FRED ({route})")
        except Exception as e:
            print(f"  !!  {label} ({code}) FAILED: {e}")
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series).loc[START:FREEZE]
    df.index.name = "date"
    df.to_csv(RAW / "fred_fx.csv")
    print(f"  OK  {len(df.columns)} currencies "
          f"(rest comes from LSEG — see Data/manual/LSEG_TODO.md)")
    return df


# ----------------------------------------------------------------------------
# GPR / EPU / TPU
# ----------------------------------------------------------------------------
def fetch_table(urls):
    for url in urls:
        try:
            r = http_get(url)
            if url.endswith(".csv"):
                return pd.read_csv(BytesIO(r.content)), url
            try:
                return pd.read_excel(BytesIO(r.content), engine="openpyxl"), url
            except Exception:
                return pd.read_excel(BytesIO(r.content), engine="xlrd"), url
        except Exception:
            continue
    return None, None


def get_indices():
    header("3/6  GPR / EPU / TPU indices")
    for name, urls in GPR_URLS.items():
        df, url = fetch_table(urls)
        if df is not None:
            df.to_csv(RAW / f"{name}.csv", index=False)
            stamp(name.upper(), "Geopolitical Risk index (Caldara & Iacoviello)", url)
            print(f"  OK  {name} ({len(df)} rows)")
        else:
            print(f"  !!  {name}: get manually from matteoiacoviello.com/gpr.htm")
    df, url = fetch_table([EPU_DAILY_URL])
    if df is not None:
        df.to_csv(RAW / "epu_daily.csv", index=False)
        stamp("EPU_daily", "US Economic Policy Uncertainty (daily)", url)
        print(f"  OK  epu_daily ({len(df)} rows)")
    else:
        print("  !!  epu_daily: get from policyuncertainty.com -> Data/raw/epu_daily.csv")
    df, url = fetch_table(TPU_URLS)
    if df is not None:
        df.to_csv(RAW / "tpu_daily.csv", index=False)
        stamp("TPU_daily", "Trade Policy Uncertainty (daily)", url)
        print(f"  OK  tpu_daily ({len(df)} rows)")
    else:
        print("  !!  tpu_daily: auto-download failed.")
        old = OLD_RAW / "tpu_daily.csv"
        if old.exists() and not (RAW / "tpu_daily.csv").exists():
            shutil.copy(old, RAW / "tpu_daily.csv")
            stamp("TPU_daily", "TPU daily (old file, ends Mar 2026 — REFRESH "
                  "from matteoiacoviello.com/tpu.htm before freeze)", str(old))
            print("      -> used old file for now (ends 2026-03-16)")


# ----------------------------------------------------------------------------
# Course data + LSEG
# ----------------------------------------------------------------------------
def carry_over_course_files():
    header("4/6  Course data (Project 2/3)")
    for fname in CARRY_OVER:
        src, dst = OLD_RAW / fname, MANUAL / fname
        if dst.exists():
            print(f"  OK  {fname} already in Data/manual/")
        elif src.exists():
            shutil.copy(src, dst)
            stamp(fname, "Course dataset (Dalgic)", "ILIAS / old pipeline")
            print(f"  OK  copied {fname}")
        else:
            print(f"  !!  {fname} missing — get from ILIAS -> Data/manual/")


def load_lseg_exports():
    """Merge any lseg_*.csv / lseg_*.xlsx the user exported into Data/manual/."""
    header("5/6  LSEG exports")
    (MANUAL / "LSEG_TODO.md").write_text(LSEG_TODO)
    frames = []
    files = sorted(list(MANUAL.glob("lseg_*.csv")) + list(MANUAL.glob("lseg_*.xlsx")))
    # Forward points/outrights are inputs to build_fx_factors.py, NOT panel
    # series — they share bare currency names with spot and would collide.
    files = [f for f in files if not any(k in f.name.lower()
             for k in ("fwd", "forward"))]
    if not files:
        print("  --  none yet. Shopping list written to Data/manual/LSEG_TODO.md")
        return pd.DataFrame()
    for f in files:
        try:
            df = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f)
            datecol = next(c for c in df.columns if c.lower() in ("date", "day"))
            df[datecol] = pd.to_datetime(df[datecol])
            df = df.set_index(datecol)
            df.index.name = "date"
            # LSEG quotes the majors USD per FC; the panel is FC per USD.
            if "fx_spot" in f.name.lower():
                for _c in ("EUR", "GBP", "AUD", "NZD"):
                    if _c in df.columns:
                        df[_c] = 1.0 / df[_c]
            frames.append(df)
            stamp(f.name, f"LSEG export ({', '.join(map(str, df.columns))})",
                  "LSEG Workspace")
            print(f"  OK  {f.name} ({df.shape[1]} series)")
        except Exception as e:
            print(f"  !!  {f.name}: could not parse ({e}) — needs a 'date' column")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------
def build_panel(parts):
    header("6/6  Merge")
    parts = [p for p in parts if p is not None and not p.empty]
    gpr_f = RAW / "gpr_daily.csv"
    if gpr_f.exists():
        g = pd.read_csv(gpr_f, parse_dates=["date"]).set_index("date")
        cols = [c for c in ("GPRD", "GPRD_ACT", "GPRD_THREAT") if c in g.columns]
        parts.append(g[cols])
    if not parts:
        print("  !!  nothing to merge")
        return
    panel = pd.concat(parts, axis=1).sort_index().loc[START:FREEZE]
    panel = panel[panel.index.dayofweek < 5]
    panel.to_csv(PROC / "daily_panel.csv")
    pd.DataFrame(LOG).to_csv(ROOT / "Data/sources_log.csv", index=False)
    print(f"  Panel: {panel.shape[0]} days x {panel.shape[1]} series "
          f"({panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d})")
    print("  Saved Data/processed/daily_panel.csv + Data/sources_log.csv")


if __name__ == "__main__":
    print(f"Thesis data download | FRED + LSEG | sample {START} -> {FREEZE}")
    macro = get_fred_macro()
    fx = get_fred_fx()                        # saved to Data/raw/fred_fx.csv (cross-check)
    get_indices()
    carry_over_course_files()
    lseg = load_lseg_exports()
    # FX source: prefer the consistent LSEG WMR panel (32 ccy) when present; FRED
    # H.10 then serves only as a cross-check. Else fall back to FRED's ~21.
    if (MANUAL / "lseg_fx_spot.csv").exists():
        print("  FX source: LSEG (lseg_fx_spot.csv); FRED FX kept as cross-check.")
        build_panel([macro, lseg])
    else:
        print("  FX source: FRED H.10 (no lseg_fx_spot.csv yet).")
        build_panel([macro, fx, lseg])
    print("\nDone. LSEG list: Data/manual/LSEG_TODO.md "
          "(drop exports in Data/manual/, rerun this script).")
