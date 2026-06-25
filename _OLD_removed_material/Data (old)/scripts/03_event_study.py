"""
03_event_study.py
=================
Event study analysis around three geopolitical shocks:
  1. Liberation Day (Apr 2, 2025) — US tariffs on almost all countries
  2. Iran strikes (Jun 2025) — military escalation
  3. Hormuz closure (Feb 2026) — Strait of Hormuz disruption

Methodology:
  - Estimation window: [-120, -11] trading days before event
  - Event window: [-5, +10] trading days around event
  - Abnormal returns = actual - mean(estimation window)
  - Cumulative abnormal returns (CAR) with t-tests
  - Cross-sectional: safe vs risky, oil exporters vs importers

Run from the Data folder:
    cd Data
    python3 scripts/03_event_study.py

Input:  data/processed/master_daily.csv
Output: output/figures/event_study_*.png
        output/tables/event_study_*.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROC_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join("output", "figures")
TAB_DIR = os.path.join("output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Event definitions
EVENTS = {
    "liberation_day": {
        "date": "2025-04-02",
        "label": "Liberation Day (Apr 2, 2025)",
        "est_window": (-120, -11),   # estimation window relative to event
        "evt_window": (-5, 10),      # event window
    },
    "iran_strikes": {
        "date": "2025-06-15",
        "label": "Iran Strikes (Jun 2025)",
        "est_window": (-120, -11),
        "evt_window": (-5, 10),
    },
    "hormuz_closure": {
        "date": "2026-02-15",
        "label": "Hormuz Closure (Feb 2026)",
        "est_window": (-120, -11),
        "evt_window": (-5, 10),
    },
}

# Series to study
PORTFOLIO_SERIES = [
    "port_safe", "port_risky", "port_RmS",
    "port_oil_exp", "port_oil_imp", "port_ExpMinusImp",
    "port_dollar_basket",
]

MARKET_SERIES = [
    "ret_Brent", "ret_WTI", "ret_Gold", "ret_SP500", "ret_Bitcoin",
]

# Individual currencies of special interest
KEY_CURRENCIES = [
    "JPY", "CHF", "EUR",       # safe havens
    "NOK", "CAD", "MXN",       # oil exporters
    "TRY", "INR", "BRL",       # EM / oil importers
]

# Plot style
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def find_event_index(dates, event_date_str):
    """Find the index position of the event date (or nearest trading day)."""
    event_date = pd.Timestamp(event_date_str)
    if event_date in dates:
        return dates.get_loc(event_date)

    # Find nearest trading day within 5 days
    mask = (dates >= event_date - pd.Timedelta(days=5)) & \
           (dates <= event_date + pd.Timedelta(days=5))
    nearby = dates[mask]
    if len(nearby) == 0:
        return None
    closest = nearby[np.argmin(np.abs(nearby - event_date))]
    return dates.get_loc(closest)


def compute_car(series, event_idx, est_window, evt_window):
    """
    Compute Cumulative Abnormal Returns.

    Parameters
    ----------
    series : pd.Series — daily returns
    event_idx : int — position of event in the index
    est_window : tuple — (start, end) relative to event_idx
    evt_window : tuple — (start, end) relative to event_idx

    Returns
    -------
    dict with AR, CAR, t-stat, p-value, estimation stats
    """
    est_start = event_idx + est_window[0]
    est_end = event_idx + est_window[1]
    evt_start = event_idx + evt_window[0]
    evt_end = event_idx + evt_window[1]

    # Bounds check
    if est_start < 0 or evt_end >= len(series):
        return None

    # Estimation window: compute mean and std of normal returns
    est_returns = series.iloc[est_start:est_end + 1].dropna()
    if len(est_returns) < 30:
        return None

    mu = est_returns.mean()
    sigma = est_returns.std()

    # Event window: compute abnormal returns
    evt_returns = series.iloc[evt_start:evt_end + 1]
    evt_dates = series.index[evt_start:evt_end + 1]

    ar = evt_returns - mu                    # abnormal returns
    car = ar.cumsum()                        # cumulative abnormal returns

    # t-test: is CAR significantly different from 0?
    # Under H0, CAR(t1,t2) ~ N(0, (t2-t1+1) * sigma^2)
    n_days = len(ar.dropna())
    car_final = car.iloc[-1] if not np.isnan(car.iloc[-1]) else np.nan
    car_std = sigma * np.sqrt(n_days)
    t_stat = car_final / car_std if car_std > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(est_returns) - 1))

    # Relative days for plotting
    rel_days = list(range(evt_window[0], evt_window[0] + len(ar)))

    return {
        "ar": ar,
        "car": car,
        "rel_days": rel_days,
        "dates": evt_dates,
        "car_final": car_final,
        "t_stat": t_stat,
        "p_value": p_value,
        "mu_est": mu,
        "sigma_est": sigma,
        "n_est": len(est_returns),
        "n_evt": n_days,
    }


# ---------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------
def run_single_event_study(master, event_name, event_info, series_list, series_type="portfolio"):
    """Run event study for one event across multiple series."""
    print(f"\n  Event: {event_info['label']}")

    dates = master.index
    event_idx = find_event_index(dates, event_info["date"])

    if event_idx is None:
        print(f"    ⚠ Event date {event_info['date']} not found in data")
        return pd.DataFrame()

    actual_date = dates[event_idx]
    print(f"    Event date in data: {actual_date.date()}")

    results = []
    car_curves = {}

    for col in series_list:
        if col not in master.columns:
            continue

        series = master[col].copy()
        res = compute_car(series, event_idx, event_info["est_window"], event_info["evt_window"])

        if res is None:
            print(f"    {col}: insufficient data")
            continue

        results.append({
            "series": col,
            "event": event_name,
            "event_date": actual_date.date(),
            "CAR_pct": res["car_final"] * 100,
            "t_stat": res["t_stat"],
            "p_value": res["p_value"],
            "significant_5pct": "***" if res["p_value"] < 0.01 else
                                "**" if res["p_value"] < 0.05 else
                                "*" if res["p_value"] < 0.10 else "",
            "AR_day0_pct": res["ar"].iloc[abs(event_info["evt_window"][0])] * 100
                           if abs(event_info["evt_window"][0]) < len(res["ar"]) else np.nan,
            "est_mean_pct": res["mu_est"] * 100,
            "est_std_pct": res["sigma_est"] * 100,
        })

        car_curves[col] = {
            "rel_days": res["rel_days"],
            "car": res["car"].values * 100,  # convert to percent
        }

        stars = results[-1]["significant_5pct"]
        print(f"    {col:30s} CAR={res['car_final']*100:+7.2f}%  t={res['t_stat']:+6.2f}  {stars}")

    results_df = pd.DataFrame(results)
    return results_df, car_curves


def plot_car_curves(car_curves, event_info, event_name, group_label):
    """Plot CAR curves for a set of series around one event."""
    if not car_curves:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for label, data in car_curves.items():
        ax.plot(data["rel_days"][:len(data["car"])], data["car"], label=label, linewidth=2)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(x=0, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label="Event day")

    ax.set_xlabel("Trading days relative to event")
    ax.set_ylabel("Cumulative Abnormal Return (%)")
    ax.set_title(f"Event Study: {event_info['label']}\n{group_label}")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fname = f"event_study_{event_name}_{group_label.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Saved {fname}")


def plot_car_comparison(all_results, series_name, events_info):
    """Plot same series across different events for comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for event_name, (car_curves, event_info) in events_info.items():
        if series_name in car_curves:
            data = car_curves[series_name]
            ax.plot(data["rel_days"][:len(data["car"])], data["car"],
                    label=event_info["label"], linewidth=2)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axvline(x=0, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xlabel("Trading days relative to event")
    ax.set_ylabel("Cumulative Abnormal Return (%)")
    ax.set_title(f"Event Comparison: {series_name}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = f"event_comparison_{series_name}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_project3_analysis(master):
    """
    Replicate Project 3 analysis: plot yields, TIPS, EUR/USD around Liberation Day.
    Uses the P3_ columns from liberation_day_data.xlsx.
    """
    print("\n" + "=" * 60)
    print("PROJECT 3 REPLICATION: Liberation Day Variables")
    print("=" * 60)

    p3_cols = [c for c in master.columns if c.startswith("P3_")]
    if not p3_cols:
        print("  ⚠ No Project 3 data found (P3_ columns)")
        return

    # Filter to the date range with P3 data
    p3_data = master[p3_cols].dropna(how="all")
    if p3_data.empty:
        print("  ⚠ Project 3 columns are all NaN")
        return

    print(f"  P3 data: {len(p3_data)} rows, columns: {p3_cols}")

    liberation_date = pd.Timestamp("2025-04-02")

    # --- Plot 1: US yields and TIPS ---
    yield_cols = [c for c in p3_cols if "US_5Y" in c or "US_1Y" in c or "TIPS" in c or "DE_1Y" in c]
    if yield_cols:
        fig, ax = plt.subplots(figsize=(12, 7))
        for col in yield_cols:
            series = p3_data[col].dropna()
            label = col.replace("P3_", "")
            ax.plot(series.index, series.values, label=label, linewidth=1.8)

        ax.axvline(x=liberation_date, color="red", linewidth=1.5, linestyle="--",
                    alpha=0.8, label="Liberation Day")
        ax.set_ylabel("Yield (%)")
        ax.set_title("Interest Rates Around Liberation Day")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "p3_yields_liberation_day.png"), dpi=150)
        plt.close(fig)
        print("  → Saved p3_yields_liberation_day.png")

    # --- Plot 2: Yield curve slope (5Y - 1Y) ---
    if "P3_US_5Y" in p3_cols and "P3_US_1Y" in p3_cols:
        slope = p3_data["P3_US_5Y"] - p3_data["P3_US_1Y"]
        slope = slope.dropna()
        if not slope.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(slope.index, slope.values, color="darkblue", linewidth=1.8)
            ax.axhline(y=0, color="gray", linewidth=0.8, linestyle=":")
            ax.axvline(x=liberation_date, color="red", linewidth=1.5, linestyle="--",
                        alpha=0.8, label="Liberation Day")
            ax.fill_between(slope.index, slope.values, 0, alpha=0.15,
                            where=slope.values > 0, color="blue", label="Upward sloping")
            ax.fill_between(slope.index, slope.values, 0, alpha=0.15,
                            where=slope.values < 0, color="red", label="Inverted")
            ax.set_ylabel("5Y − 1Y Spread (%)")
            ax.set_title("US Yield Curve Slope Around Liberation Day")
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "p3_yield_curve_slope.png"), dpi=150)
            plt.close(fig)
            print("  → Saved p3_yield_curve_slope.png")

    # --- Plot 3: EUR/USD ---
    if "P3_EURUSD" in p3_cols:
        eurusd = p3_data["P3_EURUSD"].dropna()
        if not eurusd.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(eurusd.index, eurusd.values, color="darkgreen", linewidth=1.8)
            ax.axvline(x=liberation_date, color="red", linewidth=1.5, linestyle="--",
                        alpha=0.8, label="Liberation Day")
            ax.set_ylabel("USD per EUR")
            ax.set_title("EUR/USD Exchange Rate Around Liberation Day")
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "p3_eurusd_liberation_day.png"), dpi=150)
            plt.close(fig)
            print("  → Saved p3_eurusd_liberation_day.png")

    # --- Table: changes around Liberation Day ---
    if liberation_date in p3_data.index or any(p3_data.index > liberation_date):
        # Find pre and post values
        pre = p3_data[p3_data.index < liberation_date].iloc[-1] if any(p3_data.index < liberation_date) else None
        # 1 day, 1 week, 2 weeks after
        post_dates = {}
        for label, delta in [("1d", 1), ("1w", 5), ("2w", 10)]:
            target = liberation_date + pd.Timedelta(days=delta)
            post_mask = p3_data.index >= target
            if post_mask.any():
                post_dates[label] = p3_data[post_mask].iloc[0]

        if pre is not None and post_dates:
            changes = pd.DataFrame({"Pre-event": pre})
            for label, post in post_dates.items():
                changes[f"Post {label}"] = post
                changes[f"Chg {label}"] = post - pre

            changes.to_csv(os.path.join(TAB_DIR, "p3_liberation_day_changes.csv"))
            print("  → Saved p3_liberation_day_changes.csv")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     EVENT STUDY ANALYSIS                               ║")
    print("║     Geopolitical Shocks & Currency Markets             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load data
    master_path = os.path.join(PROC_DIR, "master_daily.csv")
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"\n  Loaded master_daily.csv: {master.shape}")
    print(f"  Date range: {master.index[0].date()} to {master.index[-1].date()}")

    # ======================================================================
    # A. EVENT STUDIES FOR EACH EVENT
    # ======================================================================
    all_results = []
    all_car_curves = {}

    for event_name, event_info in EVENTS.items():
        print("\n" + "=" * 60)
        print(f"EVENT STUDY: {event_info['label']}")
        print("=" * 60)

        # --- Portfolios ---
        res_df, car_curves = run_single_event_study(
            master, event_name, event_info, PORTFOLIO_SERIES, "portfolio"
        )
        if not res_df.empty:
            all_results.append(res_df)
            plot_car_curves(car_curves, event_info, event_name, "Portfolios")

        # --- Market series ---
        res_df_mkt, car_curves_mkt = run_single_event_study(
            master, event_name, event_info, MARKET_SERIES, "market"
        )
        if not res_df_mkt.empty:
            all_results.append(res_df_mkt)
            plot_car_curves(car_curves_mkt, event_info, event_name, "Market")

        # --- Individual currencies ---
        res_df_ccy, car_curves_ccy = run_single_event_study(
            master, event_name, event_info, KEY_CURRENCIES, "currency"
        )
        if not res_df_ccy.empty:
            all_results.append(res_df_ccy)
            plot_car_curves(car_curves_ccy, event_info, event_name, "Key Currencies")

        # Store for cross-event comparison
        all_car_curves[event_name] = (
            {**car_curves, **car_curves_mkt, **car_curves_ccy},
            event_info
        )

    # ======================================================================
    # B. CROSS-EVENT COMPARISONS
    # ======================================================================
    print("\n" + "=" * 60)
    print("CROSS-EVENT COMPARISONS")
    print("=" * 60)

    comparison_series = ["port_RmS", "port_ExpMinusImp", "ret_Brent", "port_dollar_basket"]
    for s in comparison_series:
        has_data = any(s in curves for curves, _ in all_car_curves.values())
        if has_data:
            plot_car_comparison(all_results, s, all_car_curves)
            print(f"  → Saved event_comparison_{s}.png")

    # ======================================================================
    # C. COMBINED RESULTS TABLE
    # ======================================================================
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(["event", "series"])

        # Save full table
        combined.to_csv(os.path.join(TAB_DIR, "event_study_results.csv"), index=False)
        print(f"\n  → Saved event_study_results.csv ({len(combined)} rows)")

        # Print summary
        print("\n  SUMMARY OF SIGNIFICANT RESULTS (p < 0.10):")
        print("  " + "-" * 70)
        sig = combined[combined["p_value"] < 0.10]
        if not sig.empty:
            for _, row in sig.iterrows():
                print(f"  {row['event']:25s} {row['series']:25s} "
                      f"CAR={row['CAR_pct']:+7.2f}%  t={row['t_stat']:+6.2f}  {row['significant_5pct']}")
        else:
            print("  No statistically significant results at 10% level.")

    # ======================================================================
    # D. PROJECT 3 REPLICATION
    # ======================================================================
    run_project3_analysis(master)

    # ======================================================================
    # DONE
    # ======================================================================
    print("\n" + "=" * 60)
    print("EVENT STUDY COMPLETE")
    print("=" * 60)
    print(f"  Figures: {os.path.abspath(FIG_DIR)}")
    print(f"  Tables:  {os.path.abspath(TAB_DIR)}")


if __name__ == "__main__":
    main()
