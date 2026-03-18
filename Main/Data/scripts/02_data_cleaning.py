"""
02_data_cleaning.py
===================
Merges all raw data into a single master_daily.csv ready for analysis.

Steps:
  1. Load FX spot, market, GPR, EPU/TPU, course factors, liberation day data
  2. Compute log returns for FX and market series
  3. Classify currencies: safe vs risky (Lustig et al. framework)
  4. Classify currencies: oil exporter vs oil importer
  5. Build equal-weighted portfolios (safe, risky, oil-exp, oil-imp)
  6. Create event dummies (Liberation Day, Iran strikes, Hormuz closure)
  7. Merge everything and output master_daily.csv

Run from the Data folder:
    cd Data
    python3 scripts/02_data_cleaning.py

Input files (all in data/raw/):
    fx_spot_daily.csv          — from 01_data_acquisition.py
    market_daily.csv           — from 01_data_acquisition.py
    gpr_daily.csv              — from matteoiacoviello.com
    gpr_monthly.csv            — from matteoiacoviello.com
    epu_categorical.xlsx       — from policyuncertainty.com (contains EPU + TPU monthly)
    tpu_daily.csv              — from policyuncertainty.com
    factors_course.csv         — Project 2 dataset (30 currencies + Dollar/Carry factors)
    liberation_day_data.xlsx   — Project 3 dataset (yields, TIPS, EUR/USD around Apr 2025)

Output:
    data/processed/master_daily.csv
    data/processed/master_monthly.csv
    data/processed/fx_returns_daily.csv
    data/processed/portfolios_daily.csv
    data/processed/factors_course_clean.csv
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS — run from the Data/ folder
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# CURRENCY CLASSIFICATION
# ---------------------------------------------------------------------------
# Based on Lustig, Roussanov & Verdelhan (2011) carry-trade sorting and
# standard safe-haven literature. "Safe" = low interest rate, appreciates
# in global downturns. "Risky" = high interest rate, depreciates in crises.
# This is a STARTING classification — refine with your Project 2 betas.

SAFE_CURRENCIES = [
    "JPY",   # Classic safe haven
    "CHF",   # Classic safe haven
    "EUR",   # Low rates, reserve currency
    "DKK",   # Pegged to EUR
    "SEK",   # Low rates, Scandinavian
    "SGD",   # Low rates, strong fundamentals
    "ILS",   # Current account surplus
    "TWD",   # Current account surplus
    "KRW",   # Moderate — can move to risky with your betas
]

RISKY_CURRENCIES = [
    "AUD",   # Commodity currency, high carry historically
    "NZD",   # High carry
    "GBP",   # Moderate-high carry
    "NOK",   # Oil-linked, higher carry
    "CAD",   # Commodity/oil linked
    "BRL",   # High carry EM
    "MXN",   # High carry EM
    "TRY",   # Very high carry EM
    "ZAR",   # High carry EM
    "INR",   # EM
    "THB",   # EM
    "PHP",   # EM
    "MYR",   # EM
    "IDR",   # EM
    "COP",   # EM + oil
    "CLP",   # EM + copper
    "PEN",   # EM
    "PLN",   # EM
    "HUF",   # EM
    "CZK",   # EM
    "RON",   # EM
    "RUB",   # EM + oil (limited data post-2022)
    "NGN",   # EM + oil
]

# Oil classification based on net trade position
OIL_EXPORTERS = [
    "NOK",   # Norway — major North Sea producer
    "CAD",   # Canada — oil sands
    "RUB",   # Russia — major exporter
    "COP",   # Colombia — significant oil exports
    "MXN",   # Mexico — Pemex
    "BRL",   # Brazil — pre-salt offshore
    "SAR",   # Saudi Arabia (pegged, but useful for comparison)
    "KWD",   # Kuwait (pegged)
    "NGN",   # Nigeria
    "MYR",   # Malaysia — net oil/LNG exporter
]

OIL_IMPORTERS = [
    "JPY",   # Japan — massive oil importer
    "EUR",   # Eurozone — net importer
    "INR",   # India — huge importer
    "KRW",   # South Korea — major importer
    "TRY",   # Turkey — almost entirely dependent on imports
    "THB",   # Thailand
    "PHP",   # Philippines
    "ZAR",   # South Africa
    "PLN",   # Poland
    "HUF",   # Hungary
    "CZK",   # Czech Republic
    "TWD",   # Taiwan — importer
    "IDR",   # Indonesia (shifted to net importer)
]


# ---------------------------------------------------------------------------
# EVENT DATES
# ---------------------------------------------------------------------------
EVENTS = {
    "liberation_day": {
        "date": "2025-04-02",
        "description": "US announces tariffs on almost all countries",
        "window_start": "2025-03-26",
        "window_end": "2025-04-11",
    },
    "iran_strikes_jun2025": {
        "date": "2025-06-15",          # PLACEHOLDER — adjust to actual date
        "description": "June 2025 Iran military strikes",
        "window_start": "2025-06-08",
        "window_end": "2025-06-25",
    },
    "hormuz_closure_feb2026": {
        "date": "2026-02-15",          # PLACEHOLDER — adjust to actual date
        "description": "Feb 2026 Strait of Hormuz closure/disruption",
        "window_start": "2026-02-08",
        "window_end": "2026-02-28",
    },
}


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def load_csv_safe(path, index_col=0, parse_dates=True):
    """Load a CSV, return empty DataFrame if missing."""
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
        print(f"  ✓ Loaded {os.path.basename(path)}: {df.shape}")
        return df
    else:
        print(f"  ✗ Missing: {path}")
        return pd.DataFrame()


def compute_log_returns(df, prefix=""):
    """Compute log returns: r_t = ln(P_t / P_{t-1})."""
    returns = np.log(df / df.shift(1))
    if prefix:
        returns.columns = [f"{prefix}_{c}" for c in returns.columns]
    return returns


def build_portfolio(returns_df, currencies, name):
    """
    Build equal-weighted portfolio return from available currencies.
    For FX returns where positive = USD appreciation (foreign depreciation),
    the portfolio tracks the average move of the group vs USD.
    """
    available = [c for c in currencies if c in returns_df.columns]
    if not available:
        print(f"  ⚠ No currencies available for portfolio '{name}'")
        return pd.Series(dtype=float, name=name)

    portfolio = returns_df[available].mean(axis=1)
    portfolio.name = name
    n = len(available)
    print(f"  Portfolio '{name}': {n} currencies — {', '.join(available)}")
    return portfolio


def create_event_dummies(index):
    """Create dummy variables for geopolitical events."""
    dummies = pd.DataFrame(index=index)

    for event_name, event_info in EVENTS.items():
        event_date = pd.Timestamp(event_info["date"])
        window_start = pd.Timestamp(event_info["window_start"])
        window_end = pd.Timestamp(event_info["window_end"])

        # Point dummy: 1 on the event date (or nearest trading day)
        dummies[f"d_{event_name}"] = 0
        if event_date in index:
            dummies.loc[event_date, f"d_{event_name}"] = 1
        else:
            nearby = index[(index >= event_date - pd.Timedelta(days=3)) &
                           (index <= event_date + pd.Timedelta(days=3))]
            if len(nearby) > 0:
                closest = nearby[np.argmin(np.abs(nearby - event_date))]
                dummies.loc[closest, f"d_{event_name}"] = 1

        # Window dummy: 1 during the entire event window
        dummies[f"w_{event_name}"] = (
            (index >= window_start) & (index <= window_end)
        ).astype(int)

        # Post dummy: 1 from event date onward (for structural break analysis)
        dummies[f"post_{event_name}"] = (index >= event_date).astype(int)

    return dummies


# ---------------------------------------------------------------------------
# LOAD GPR DATA
# ---------------------------------------------------------------------------
def load_gpr():
    """Load and standardize GPR data (daily and monthly)."""
    print("\n  Loading GPR data...")
    gpr_frames = {}

    # --- Daily GPR ---
    path = os.path.join(RAW_DIR, "gpr_daily.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)

            # Find date column
            date_col = None
            for c in df.columns:
                if "date" in c.lower() or "day" in c.lower():
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)

            # Keep GPR columns
            gpr_cols = [c for c in df.columns if "gpr" in c.lower() or "GPR" in c]
            if gpr_cols:
                gpr_frames["daily"] = df[gpr_cols]
                print(f"    ✓ GPR daily: {len(df)} rows, columns: {gpr_cols[:5]}...")
        except Exception as e:
            print(f"    ⚠ Error reading gpr_daily.csv: {e}")
    else:
        print(f"    ✗ gpr_daily.csv not found")

    # --- Monthly GPR ---
    path = os.path.join(RAW_DIR, "gpr_monthly.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)

            date_col = None
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)

            gpr_cols = [c for c in df.columns if "gpr" in c.lower() or "GPR" in c]
            if gpr_cols:
                gpr_frames["monthly"] = df[gpr_cols]
                print(f"    ✓ GPR monthly: {len(df)} rows, columns: {gpr_cols[:5]}...")
        except Exception as e:
            print(f"    ⚠ Error reading gpr_monthly.csv: {e}")
    else:
        print(f"    ✗ gpr_monthly.csv not found")

    return gpr_frames


# ---------------------------------------------------------------------------
# LOAD EPU / TPU — from epu_categorical.xlsx and tpu_daily.csv
# ---------------------------------------------------------------------------
def load_epu_tpu():
    """
    Load EPU and TPU data.
    - epu_categorical.xlsx: monthly EPU + Trade Policy Uncertainty as category columns
    - tpu_daily.csv: daily Trade Policy Uncertainty
    """
    print("\n  Loading EPU/TPU data...")
    frames = {}

    # --- Categorical EPU (monthly, contains TPU as a column) ---
    path = os.path.join(RAW_DIR, "epu_categorical.xlsx")
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
            print(f"    Raw columns in epu_categorical.xlsx: {list(df.columns)}")

            # Build date from Year + Month columns
            if "Year" in df.columns and "Month" in df.columns:
                # Convert to numeric first — drops footer rows with text
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
                df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
                df = df.dropna(subset=["Year", "Month"])
                df["date"] = pd.to_datetime(
                    df["Year"].astype(int).astype(str) + "-" +
                    df["Month"].astype(int).astype(str).str.zfill(2) + "-01",
                    errors="coerce"
                )
                df = df.dropna(subset=["date"])
                df = df.set_index("date")
                df = df.drop(columns=["Year", "Month"], errors="ignore")

                # Identify overall EPU and trade policy columns
                epu_col = None
                tpu_col = None
                for c in df.columns:
                    cl = c.lower()
                    if "trade" in cl and ("poli" in cl or "uncert" in cl):
                        tpu_col = c
                    elif ("news" in cl and "poli" in cl) or "epu" in cl.replace("_", ""):
                        epu_col = c

                if epu_col:
                    frames["EPU"] = df[[epu_col]].rename(columns={epu_col: "EPU"})
                    print(f"    ✓ EPU monthly: {len(df)} rows (column: '{epu_col}')")
                else:
                    # Fall back: keep all numeric columns
                    frames["EPU_all"] = df.select_dtypes(include=[np.number])
                    print(f"    ✓ EPU categorical (all columns): {len(df)} rows")

                if tpu_col:
                    frames["TPU_monthly"] = df[[tpu_col]].rename(columns={tpu_col: "TPU_monthly"})
                    print(f"    ✓ TPU monthly: {len(df)} rows (column: '{tpu_col}')")
                else:
                    print(f"    ⚠ Could not auto-identify Trade Policy column.")
                    print(f"      Available columns: {list(df.columns)}")
                    print(f"      → You may need to set the column name manually in the script")

            else:
                print(f"    ⚠ No Year/Month columns. Columns: {list(df.columns)}")

        except Exception as e:
            print(f"    ⚠ Error reading epu_categorical.xlsx: {e}")
    else:
        print(f"    ✗ epu_categorical.xlsx not found")

    # --- Daily TPU ---
    path = os.path.join(RAW_DIR, "tpu_daily.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)

            # Find date column
            date_col = None
            for c in df.columns:
                if "date" in c.lower() or "day" in c.lower():
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)

            # Find TPU value columns
            tpu_cols = [c for c in df.columns if "tpu" in c.lower() or "trade" in c.lower()
                        or "uncert" in c.lower()]
            if tpu_cols:
                frames["TPU_daily"] = df[tpu_cols]
            else:
                frames["TPU_daily"] = df.select_dtypes(include=[np.number])

            print(f"    ✓ TPU daily: {len(df)} rows, columns: {list(frames['TPU_daily'].columns)}")

        except Exception as e:
            print(f"    ⚠ Error reading tpu_daily.csv: {e}")
    else:
        print(f"    ✗ tpu_daily.csv not found")

    return frames


# ---------------------------------------------------------------------------
# LOAD LIBERATION DAY DATA (Project 3)
# ---------------------------------------------------------------------------
def load_liberation_day_data():
    """
    Load Project 3 dataset: daily US 5Y yield, 5Y TIPS, US 1Y rate,
    Germany 1Y rate, EUR/USD exchange rate.
    """
    print("\n  Loading Liberation Day data (Project 3)...")

    path = os.path.join(RAW_DIR, "liberation_day_data.xlsx")
    if not os.path.exists(path):
        print(f"    ✗ liberation_day_data.xlsx not found")
        return pd.DataFrame()

    try:
        df = pd.read_excel(path)
        print(f"    Raw columns: {list(df.columns)}")

        # Find date column
        date_col = None
        for c in df.columns:
            if "date" in c.lower() or "day" in c.lower():
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)

        # Prefix columns to avoid clashes with market data
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if "tip" in cl or "real" in cl or ("inflation" in cl and "protect" in cl):
                rename_map[c] = "P3_TIPS_5Y"
            elif "eur" in cl or "eurusd" in cl or ("dollar" in cl and "euro" in cl):
                rename_map[c] = "P3_EURUSD"
            elif "german" in cl or "de_" in cl or "bund" in cl:
                rename_map[c] = "P3_DE_1Y"
            elif "5" in c and ("yield" in cl or "rate" in cl or "yr" in cl or "year" in cl):
                rename_map[c] = "P3_US_5Y"
            elif "1" in c and ("yield" in cl or "rate" in cl or "yr" in cl or "year" in cl):
                rename_map[c] = "P3_US_1Y"

        # If auto-rename didn't catch everything, keep original with P3_ prefix
        for c in df.columns:
            if c not in rename_map:
                rename_map[c] = f"P3_{c}"

        df = df.rename(columns=rename_map)
        print(f"    ✓ Liberation Day data: {len(df)} rows, columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"    ⚠ Error reading liberation_day_data.xlsx: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# LOAD COURSE FACTORS (Project 2)
# ---------------------------------------------------------------------------
def load_course_factors():
    """
    Load Project 2 dataset: 30 currencies, Dollar Risk, Carry Trade Risk.
    Monthly frequency.
    """
    print("\n  Loading course factors (Project 2)...")

    path = os.path.join(RAW_DIR, "factors_course.csv")
    if not os.path.exists(path):
        print(f"    ✗ factors_course.csv not found")
        return pd.DataFrame()

    try:
        # Try UTF-8 first, then latin-1 (common with European Excel exports)
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except UnicodeDecodeError:
            df = pd.read_csv(path, index_col=0, parse_dates=True, encoding="latin-1")
        print(f"    ✓ Course factors: {df.shape}")
        print(f"    Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        return df
    except Exception as e:
        print(f"    ⚠ Error reading factors_course.csv: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     THESIS DATA CLEANING PIPELINE                      ║")
    print("║     Merge, Transform, Build Portfolios                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # A. LOAD RAW DATA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    fx_spot = load_csv_safe(os.path.join(RAW_DIR, "fx_spot_daily.csv"))
    market = load_csv_safe(os.path.join(RAW_DIR, "market_daily.csv"))

    gpr_data = load_gpr()
    epu_data = load_epu_tpu()
    liberation_data = load_liberation_day_data()
    factors_course = load_course_factors()

    # ------------------------------------------------------------------
    # B. COMPUTE LOG RETURNS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPUTING LOG RETURNS")
    print("=" * 60)

    # FX log returns: positive = foreign currency depreciation vs USD
    if not fx_spot.empty:
        fx_ret = compute_log_returns(fx_spot)
        print(f"  FX returns: {fx_ret.shape}")
        fx_ret.to_csv(os.path.join(PROC_DIR, "fx_returns_daily.csv"))
        print(f"  → Saved fx_returns_daily.csv")
    else:
        fx_ret = pd.DataFrame()

    # Market returns
    if not market.empty:
        price_cols = [c for c in market.columns
                      if c not in ["VIX", "UST10Y", "UST5Y", "UST3M"]]
        yield_cols = [c for c in market.columns
                      if c in ["VIX", "UST10Y", "UST5Y", "UST3M"]]

        mkt_ret = pd.DataFrame(index=market.index)
        if price_cols:
            mkt_ret = compute_log_returns(market[price_cols], prefix="ret")
        if yield_cols:
            for c in yield_cols:
                mkt_ret[f"chg_{c}"] = market[c].diff()
            for c in yield_cols:
                mkt_ret[f"lvl_{c}"] = market[c]
        for c in price_cols:
            mkt_ret[f"lvl_{c}"] = market[c]

        print(f"  Market returns/changes: {mkt_ret.shape}")
    else:
        mkt_ret = pd.DataFrame()

    # ------------------------------------------------------------------
    # C. BUILD PORTFOLIOS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING CURRENCY PORTFOLIOS")
    print("=" * 60)

    portfolios = pd.DataFrame(index=fx_ret.index if not fx_ret.empty else [])

    if not fx_ret.empty:
        # Safe vs Risky
        p_safe = build_portfolio(fx_ret, SAFE_CURRENCIES, "port_safe")
        p_risky = build_portfolio(fx_ret, RISKY_CURRENCIES, "port_risky")

        if not p_safe.empty and not p_risky.empty:
            portfolios["port_safe"] = p_safe
            portfolios["port_risky"] = p_risky
            portfolios["port_RmS"] = p_risky - p_safe
            print(f"  Risky-minus-Safe spread computed")

        # Oil exporter vs importer
        p_oil_exp = build_portfolio(fx_ret, OIL_EXPORTERS, "port_oil_exp")
        p_oil_imp = build_portfolio(fx_ret, OIL_IMPORTERS, "port_oil_imp")

        if not p_oil_exp.empty and not p_oil_imp.empty:
            portfolios["port_oil_exp"] = p_oil_exp
            portfolios["port_oil_imp"] = p_oil_imp
            portfolios["port_ExpMinusImp"] = p_oil_exp - p_oil_imp
            print(f"  Exporter-minus-Importer spread computed")

        # Dollar basket (equal-weight all available currencies)
        all_ccy = fx_ret.columns.tolist()
        p_dollar = build_portfolio(fx_ret, all_ccy, "port_dollar_basket")
        if not p_dollar.empty:
            portfolios["port_dollar_basket"] = p_dollar

        portfolios.to_csv(os.path.join(PROC_DIR, "portfolios_daily.csv"))
        print(f"  → Saved portfolios_daily.csv")

    # ------------------------------------------------------------------
    # D. EVENT DUMMIES
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CREATING EVENT DUMMIES")
    print("=" * 60)

    if not fx_ret.empty:
        master_index = fx_ret.index
    elif not market.empty:
        master_index = market.index
    else:
        master_index = pd.DatetimeIndex([])

    event_dummies = create_event_dummies(master_index)
    for col in event_dummies.columns:
        n_ones = event_dummies[col].sum()
        print(f"  {col}: {n_ones} days flagged")

    # ------------------------------------------------------------------
    # E. MERGE INTO MASTER (DAILY)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MERGING INTO MASTER DATASET")
    print("=" * 60)

    # Start with FX returns
    master = fx_ret.copy() if not fx_ret.empty else pd.DataFrame()

    # Add portfolios
    if not portfolios.empty:
        master = master.join(portfolios, how="outer")
        print(f"  + portfolios: {master.shape}")

    # Add market returns
    if not mkt_ret.empty:
        master = master.join(mkt_ret, how="outer", rsuffix="_mkt")
        print(f"  + market: {master.shape}")

    # Add daily GPR
    if "daily" in gpr_data:
        gpr_daily = gpr_data["daily"]
        gpr_daily.index = pd.to_datetime(gpr_daily.index)
        gpr_daily.columns = [f"GPR_{c}" if not c.upper().startswith("GPR")
                              else c for c in gpr_daily.columns]
        master = master.join(gpr_daily, how="left")
        print(f"  + GPR daily: {master.shape}")

    # Add daily TPU
    if "TPU_daily" in epu_data:
        tpu_daily = epu_data["TPU_daily"]
        tpu_daily.index = pd.to_datetime(tpu_daily.index)
        tpu_daily.columns = [f"TPU_{c}" if not c.upper().startswith("TPU")
                              else c for c in tpu_daily.columns]
        master = master.join(tpu_daily, how="left")
        print(f"  + TPU daily: {master.shape}")

    # Add Liberation Day data (Project 3)
    if not liberation_data.empty:
        liberation_data.index = pd.to_datetime(liberation_data.index)
        master = master.join(liberation_data, how="left")
        print(f"  + Liberation Day data: {master.shape}")

    # Add event dummies
    master = master.join(event_dummies, how="left")
    print(f"  + event dummies: {master.shape}")

    # ------------------------------------------------------------------
    # F. SAVE MONTHLY DATA SEPARATELY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING MONTHLY MASTER")
    print("=" * 60)

    monthly_frames = []

    if "EPU" in epu_data:
        monthly_frames.append(epu_data["EPU"])
    if "EPU_all" in epu_data:
        monthly_frames.append(epu_data["EPU_all"])
    if "TPU_monthly" in epu_data:
        monthly_frames.append(epu_data["TPU_monthly"])
    if "monthly" in gpr_data:
        gpr_m = gpr_data["monthly"]
        gpr_m.columns = [f"GPR_{c}" if not c.upper().startswith("GPR")
                          else c for c in gpr_m.columns]
        monthly_frames.append(gpr_m)

    monthly_path = None
    if monthly_frames:
        monthly_master = pd.concat(monthly_frames, axis=1, join="outer")
        monthly_master = monthly_master.sort_index()
        monthly_path = os.path.join(PROC_DIR, "master_monthly.csv")
        monthly_master.to_csv(monthly_path)
        print(f"  → Saved master_monthly.csv: {monthly_master.shape[0]} rows × {monthly_master.shape[1]} columns")
        print(f"    Columns: {list(monthly_master.columns)}")
    else:
        print("  ⚠ No monthly data to merge")

    # Save course factors separately (monthly, different structure)
    if not factors_course.empty:
        factors_course.to_csv(os.path.join(PROC_DIR, "factors_course_clean.csv"))
        print(f"  → Saved factors_course_clean.csv: {factors_course.shape}")

    # ------------------------------------------------------------------
    # G. CLEAN UP MASTER
    # ------------------------------------------------------------------
    master = master.sort_index()
    master = master.dropna(how="all")

    # Forward-fill level/index data on weekends (different calendars)
    # but do NOT forward-fill FX returns
    ffill_cols = [c for c in master.columns
                  if any(x in c for x in ["GPR", "TPU", "lvl_", "P3_"])]
    if ffill_cols:
        master[ffill_cols] = master[ffill_cols].ffill(limit=5)

    # ------------------------------------------------------------------
    # H. SAVE MASTER
    # ------------------------------------------------------------------
    outpath = os.path.join(PROC_DIR, "master_daily.csv")
    master.to_csv(outpath)
    print(f"\n  → Saved master_daily.csv: {master.shape[0]:,} rows × {master.shape[1]} columns")

    # ------------------------------------------------------------------
    # I. SUMMARY STATISTICS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUICK SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\n  Date range: {master.index[0].date()} to {master.index[-1].date()}")
    print(f"  Trading days: {len(master):,}")
    print(f"  Columns: {len(master.columns)}")

    # FX return columns
    fx_cols = [c for c in master.columns if c in fx_ret.columns] if not fx_ret.empty else []
    print(f"\n  FX currencies: {len(fx_cols)}")

    # Portfolio columns
    port_cols = [c for c in master.columns if c.startswith("port_")]
    if port_cols:
        print(f"  Portfolios: {', '.join(port_cols)}")

        print(f"\n  {'Portfolio':<25s} {'Mean(ann%)':<12s} {'Vol(ann%)':<12s} {'Sharpe':<8s}")
        print("  " + "-" * 57)
        for pc in port_cols:
            s = master[pc].dropna()
            if len(s) > 100:
                ann_mean = s.mean() * 252 * 100
                ann_vol = s.std() * np.sqrt(252) * 100
                sharpe = ann_mean / ann_vol if ann_vol > 0 else 0
                print(f"  {pc:<25s} {ann_mean:>10.2f}  {ann_vol:>10.2f}  {sharpe:>6.2f}")

    # Event coverage
    event_cols = [c for c in master.columns if c.startswith(("d_", "w_", "post_"))]
    if event_cols:
        print(f"\n  Event dummies: {len(event_cols)}")
        future_events = []
        for ec in event_cols:
            if ec.startswith("d_") and master[ec].sum() == 0:
                future_events.append(ec.replace("d_", ""))
        if future_events:
            print(f"  ⚠ Future/unmatched events (no data yet): {', '.join(future_events)}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Daily master:   {os.path.abspath(outpath)}")
    if monthly_path:
        print(f"  Monthly master: {os.path.abspath(monthly_path)}")
    print("  Ready for analysis scripts.\n")


if __name__ == "__main__":
    main()
