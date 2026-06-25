"""
01_data_acquisition.py
======================
Downloads all raw data needed for the thesis:
  - G10 + EM currency pairs (yfinance)
  - VIX, Oil (Brent & WTI), Gold, S&P 500, Bitcoin, Yields (yfinance)
  - GPR daily & monthly (Caldara & Iacoviello 2022) — auto-attempt, manual fallback

Run from the Data folder:
    cd Data
    python3 scripts/01_data_acquisition.py

Outputs go to data/raw/
"""

import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# 1. CURRENCY PAIRS — yfinance
# ---------------------------------------------------------------------------
# We want everything as units of foreign currency per 1 USD (foreign/USD).
# For XXXUSD=X pairs (e.g. EURUSD=X gives USD per 1 EUR), we INVERT.
# For USDXXX=X pairs (e.g. USDJPY=X gives JPY per 1 USD), no inversion.

CURRENCY_PAIRS = {
    # G10 — major pairs
    "EURUSD=X":  ("EUR", True),   # yf gives USD/EUR → invert
    "GBPUSD=X":  ("GBP", True),   # yf gives USD/GBP → invert
    "AUDUSD=X":  ("AUD", True),   # yf gives USD/AUD → invert
    "NZDUSD=X":  ("NZD", True),   # yf gives USD/NZD → invert
    "USDJPY=X":  ("JPY", False),  # yf gives JPY/USD → already foreign/USD
    "USDCHF=X":  ("CHF", False),
    "USDCAD=X":  ("CAD", False),
    "USDSEK=X":  ("SEK", False),
    "USDNOK=X":  ("NOK", False),
    "USDDKK=X":  ("DKK", False),
    # EM — commodity / oil linked
    "USDBRL=X":  ("BRL", False),
    "USDMXN=X":  ("MXN", False),
    "USDTRY=X":  ("TRY", False),
    "USDZAR=X":  ("ZAR", False),
    "USDINR=X":  ("INR", False),
    "USDKRW=X":  ("KRW", False),
    "USDTWD=X":  ("TWD", False),
    "USDTHB=X":  ("THB", False),
    "USDPHP=X":  ("PHP", False),
    "USDSGD=X":  ("SGD", False),
    "USDMYR=X":  ("MYR", False),
    "USDIDR=X":  ("IDR", False),
    "USDCOP=X":  ("COP", False),
    "USDCLP=X":  ("CLP", False),
    "USDPEN=X":  ("PEN", False),
    "USDPLN=X":  ("PLN", False),
    "USDHUF=X":  ("HUF", False),
    "USDCZK=X":  ("CZK", False),
    "USDRON=X":  ("RON", False),
    "USDILS=X":  ("ILS", False),
    # Oil exporters (EM)
    "USDRUB=X":  ("RUB", False),  # may be unavailable post-sanctions
    "USDSAR=X":  ("SAR", False),
    "USDKWD=X":  ("KWD", False),
    "USDNGN=X":  ("NGN", False),
}


def download_fx_data():
    """Download FX spot rates from yfinance."""
    import yfinance as yf

    print("\n" + "=" * 60)
    print("STEP 1: Downloading FX spot rates from Yahoo Finance")
    print("=" * 60)

    all_series = {}
    failed = []

    for ticker, (label, invert) in CURRENCY_PAIRS.items():
        try:
            print(f"  Downloading {label} ({ticker})...", end=" ")
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("Empty dataframe")

            # Use Close price
            close = df["Close"].squeeze()
            if invert:
                close = 1.0 / close  # convert to foreign/USD

            all_series[label] = close
            print(f"OK — {len(close)} obs from {close.index[0].date()} to {close.index[-1].date()}")
        except Exception as e:
            failed.append((label, ticker, str(e)))
            print(f"FAILED ({e})")

    if all_series:
        fx_df = pd.DataFrame(all_series)
        fx_df.index.name = "date"
        fx_df.to_csv(os.path.join(RAW_DIR, "fx_spot_daily.csv"))
        print(f"\n  → Saved {len(fx_df)} rows, {len(fx_df.columns)} currencies to fx_spot_daily.csv")

    if failed:
        print(f"\n  ⚠ Failed pairs ({len(failed)}):")
        for label, ticker, err in failed:
            print(f"    {label} ({ticker}): {err}")

    return fx_df if all_series else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. FINANCIAL MARKET DATA — yfinance
# ---------------------------------------------------------------------------
# NOTE: DXY (DX-Y.NYB) removed — delisted on Yahoo Finance.
# A dollar index is constructed from the 34 currency pairs in script 02.

MARKET_TICKERS = {
    "^VIX":      "VIX",
    "^GSPC":     "SP500",
    "BZ=F":      "Brent",
    "CL=F":      "WTI",
    "GC=F":      "Gold",
    "BTC-USD":   "Bitcoin",
    "^TNX":      "UST10Y",       # 10-year yield
    "^FVX":      "UST5Y",        # 5-year yield
    "^IRX":      "UST3M",        # 3-month T-bill
}


def download_market_data():
    """Download VIX, oil, gold, S&P500, BTC, yields."""
    import yfinance as yf

    print("\n" + "=" * 60)
    print("STEP 2: Downloading market data from Yahoo Finance")
    print("=" * 60)

    all_series = {}
    for ticker, label in MARKET_TICKERS.items():
        try:
            print(f"  Downloading {label} ({ticker})...", end=" ")
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("Empty dataframe")
            close = df["Close"].squeeze()
            all_series[label] = close
            print(f"OK — {len(close)} obs")
        except Exception as e:
            print(f"FAILED ({e})")

    if all_series:
        mkt_df = pd.DataFrame(all_series)
        mkt_df.index.name = "date"
        mkt_df.to_csv(os.path.join(RAW_DIR, "market_daily.csv"))
        print(f"\n  → Saved market_daily.csv ({len(mkt_df)} rows, {len(mkt_df.columns)} series)")
        return mkt_df

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. GPR INDEX — Caldara & Iacoviello (2022)
# ---------------------------------------------------------------------------
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

GPR_DAILY_ALT = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xlsx"
GPR_MONTHLY_ALT = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx"


def download_gpr():
    """Try to auto-download GPR; flag for manual download if it fails."""
    print("\n" + "=" * 60)
    print("STEP 3: Downloading GPR Index (Caldara & Iacoviello 2022)")
    print("=" * 60)

    for label, urls, fname in [
        ("GPR Daily",   [GPR_DAILY_URL, GPR_DAILY_ALT],   "gpr_daily.csv"),
        ("GPR Monthly", [GPR_MONTHLY_URL, GPR_MONTHLY_ALT], "gpr_monthly.csv"),
    ]:
        outpath = os.path.join(RAW_DIR, fname)
        if os.path.exists(outpath):
            print(f"  ✓ {label}: already exists, skipping.")
            continue

        success = False
        for url in urls:
            try:
                print(f"  Trying {label} from {url}...", end=" ")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()

                # Try reading as Excel
                try:
                    df = pd.read_excel(resp.content, engine="openpyxl")
                except Exception:
                    from io import BytesIO
                    df = pd.read_excel(BytesIO(resp.content), engine="xlrd")

                df.to_csv(outpath, index=False)
                print(f"OK — saved {len(df)} rows")
                success = True
                break
            except Exception as e:
                print(f"FAILED ({e})")

        if not success:
            print(f"\n  ⚠ MANUAL DOWNLOAD REQUIRED for {label}:")
            print(f"    1. Go to https://www.matteoiacoviello.com/gpr.htm")
            print(f"    2. Download the {label.lower()} file")
            print(f"    3. Save as: {outpath}")


# ---------------------------------------------------------------------------
# 4. CHECK ALL FILES
# ---------------------------------------------------------------------------
def check_all_files():
    """Check that all required files are present."""
    print("\n" + "=" * 60)
    print("STEP 4: Checking all required files")
    print("=" * 60)

    # Files produced by this script
    auto_files = {
        "fx_spot_daily.csv": "FX spot rates (34 currencies)",
        "market_daily.csv":  "Market data (VIX, S&P, oil, gold, BTC, yields)",
        "gpr_daily.csv":     "GPR Daily Index (Caldara & Iacoviello)",
        "gpr_monthly.csv":   "GPR Monthly Index (Caldara & Iacoviello)",
    }

    # Files you downloaded manually
    manual_files = {
        "epu_categorical.xlsx": (
            "EPU + Trade Policy Uncertainty (monthly categories)",
            "https://www.policyuncertainty.com/categorical_epu.html"
        ),
        "tpu_daily.csv": (
            "Trade Policy Uncertainty (daily)",
            "https://www.policyuncertainty.com/trade_uncertainty.html"
        ),
        "factors_course.csv": (
            "Project 2 dataset — 30 currencies + Dollar Risk + Carry Trade Risk",
            "Your course data from Prof. Dalgic"
        ),
        "liberation_day_data.xlsx": (
            "Project 3 dataset — yields, TIPS, EUR/USD around Liberation Day",
            "Your course data from Prof. Dalgic"
        ),
    }

    all_ok = True

    print("\n  Auto-downloaded:")
    for fname, desc in auto_files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    ✓ {fname} ({size:,} bytes)")
        else:
            all_ok = False
            print(f"    ✗ {fname} MISSING — {desc}")

    print("\n  Manual downloads:")
    for fname, (desc, source) in manual_files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    ✓ {fname} ({size:,} bytes)")
        else:
            all_ok = False
            print(f"    ✗ {fname} MISSING")
            print(f"      What: {desc}")
            print(f"      From: {source}")
            print(f"      Save to: {path}")
            print()

    return all_ok


# ---------------------------------------------------------------------------
# 5. SUMMARY REPORT
# ---------------------------------------------------------------------------
def print_summary():
    """Print what's in data/raw/."""
    print("\n" + "=" * 60)
    print("DATA INVENTORY — data/raw/")
    print("=" * 60)

    files = sorted([f for f in os.listdir(RAW_DIR) if not f.startswith(".")])
    total_size = 0
    for f in files:
        path = os.path.join(RAW_DIR, f)
        size = os.path.getsize(path)
        total_size += size
        rows = ""
        if f.endswith(".csv"):
            try:
                with open(path) as fh:
                    rows = f" ({sum(1 for _ in fh) - 1:,} rows)"
            except:
                pass
        print(f"  {f:35s} {size:>10,} bytes{rows}")

    print(f"\n  Total: {len(files)} files, {total_size:,} bytes")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     THESIS DATA ACQUISITION PIPELINE                   ║")
    print("║     Geopolitical Risk & FX — Raw Data Download         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nDate range: {START_DATE} to {END_DATE}")
    print(f"Output dir: {os.path.abspath(RAW_DIR)}")

    # Step 1: FX
    download_fx_data()

    # Step 2: Market data
    download_market_data()

    # Step 3: GPR
    download_gpr()

    # Step 4: File check (replaces old FRED + manual check steps)
    all_ok = check_all_files()

    # Summary
    print_summary()

    if all_ok:
        print("\n✓ All files present. Ready to run:")
        print("    python3 scripts/02_data_cleaning.py")
    else:
        print("\n⚠ Some files are missing. See instructions above.")
        print("  Add the missing files, then run:")
        print("    python3 scripts/02_data_cleaning.py")


if __name__ == "__main__":
    main()
