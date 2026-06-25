"""
04_var_analysis.py
==================
Vector Autoregression analysis of geopolitical risk and currency markets.

Models:
  1. Baseline VAR: GPR → currency portfolios (safe, risky, RmS)
  2. Oil channel VAR: GPR → oil → oil-exp/imp currencies
  3. Extended VAR with EPU/TPU controls
  4. Granger causality tests
  5. Impulse response functions
  6. Forecast error variance decomposition
  7. Pre/post Liberation Day structural break

Run from the Data folder:
    cd Data
    python3 scripts/04_var_analysis.py

Input:  data/processed/master_daily.csv
        data/processed/master_monthly.csv
Output: output/figures/var_*.png
        output/tables/var_*.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROC_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join("output", "figures")
TAB_DIR = os.path.join("output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

MAX_IRF_PERIODS = 24   # months for impulse responses
MAX_VAR_LAGS = 6       # max lags for VAR selection

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def resample_to_monthly(daily_df, columns, method="last"):
    """Resample daily data to monthly frequency."""
    subset = daily_df[columns].dropna(how="all")
    if method == "last":
        monthly = subset.resample("ME").last()
    elif method == "mean":
        monthly = subset.resample("ME").mean()
    elif method == "sum":
        monthly = subset.resample("ME").sum()
    return monthly.dropna()


def check_stationarity(df, significance=0.05):
    """Run ADF test on each column, report results."""
    results = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 20:
            continue
        try:
            adf_stat, p_value, lags, nobs, crit, _ = adfuller(series, autolag="AIC")
            results.append({
                "variable": col,
                "ADF_stat": adf_stat,
                "p_value": p_value,
                "lags": lags,
                "nobs": nobs,
                "stationary": p_value < significance,
            })
        except Exception:
            pass
    return pd.DataFrame(results)


def run_granger_tests(df, cause_col, effect_cols, max_lag=6):
    """Run Granger causality tests: does cause_col Granger-cause each effect?"""
    results = []
    for effect_col in effect_cols:
        pair = df[[effect_col, cause_col]].dropna()
        if len(pair) < max_lag * 3:
            continue
        try:
            test = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
            # Extract minimum p-value across lags
            min_p = min(test[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
            best_lag = min(range(1, max_lag + 1),
                          key=lambda l: test[l][0]["ssr_ftest"][1])
            f_stat = test[best_lag][0]["ssr_ftest"][0]

            results.append({
                "cause": cause_col,
                "effect": effect_col,
                "best_lag": best_lag,
                "F_stat": f_stat,
                "p_value": min_p,
                "significant_5pct": "***" if min_p < 0.01 else
                                    "**" if min_p < 0.05 else
                                    "*" if min_p < 0.10 else "",
            })
        except Exception as e:
            pass

    return pd.DataFrame(results)


def estimate_var_and_irf(df, var_name, n_periods=24, save_prefix=""):
    """
    Estimate VAR, compute IRFs and FEVD.

    Parameters
    ----------
    df : DataFrame — columns are the endogenous variables
    var_name : str — label for the model
    n_periods : int — IRF horizon
    save_prefix : str — file prefix for output

    Returns
    -------
    var_result, irf_result
    """
    print(f"\n  --- VAR Model: {var_name} ---")
    print(f"  Variables: {list(df.columns)}")
    print(f"  Observations: {len(df)}")

    # Estimate VAR with AIC lag selection
    model = VAR(df)
    try:
        lag_order = model.select_order(maxlags=min(MAX_VAR_LAGS, len(df) // 5))
        selected_lag = lag_order.aic
        print(f"  Selected lag (AIC): {selected_lag}")

        # Ensure at least 1 lag
        if selected_lag < 1:
            selected_lag = 1

        result = model.fit(selected_lag)
        print(f"  AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")

    except Exception as e:
        print(f"  ⚠ Lag selection failed ({e}), using 2 lags")
        result = model.fit(2)

    # --- Impulse Response Functions ---
    irf = result.irf(n_periods)

    # Plot IRFs
    n_vars = len(df.columns)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(4 * n_vars, 3.5 * n_vars))
    if n_vars == 1:
        axes = np.array([[axes]])
    elif n_vars > 1 and axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            irf_vals = irf.irfs[:, i, j]
            lower = irf.ci[:, i, j, 0] if irf.ci is not None else None
            upper = irf.ci[:, i, j, 1] if irf.ci is not None else None

            ax.plot(range(n_periods + 1), irf_vals, "b-", linewidth=1.5)
            if lower is not None and upper is not None:
                ax.fill_between(range(n_periods + 1), lower, upper, alpha=0.15, color="blue")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_title(f"{df.columns[j]} → {df.columns[i]}", fontsize=9)

            if i == n_vars - 1:
                ax.set_xlabel("Months")

    plt.suptitle(f"Impulse Response Functions: {var_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    fname = f"var_irf_{save_prefix}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {fname}")

    # --- Forecast Error Variance Decomposition ---
    fevd = result.fevd(n_periods)

    # Save FEVD table
    fevd_data = []
    for i, target in enumerate(df.columns):
        decomp = fevd.decomp[i]  # shape: (n_periods, n_vars)
        for h in [1, 6, 12, 24]:
            if h <= n_periods:
                row = {"target": target, "horizon": h}
                for j, source in enumerate(df.columns):
                    row[f"pct_{source}"] = decomp[h - 1, j] * 100
                fevd_data.append(row)

    fevd_df = pd.DataFrame(fevd_data)
    fname_fevd = f"var_fevd_{save_prefix}.csv"
    fevd_df.to_csv(os.path.join(TAB_DIR, fname_fevd), index=False)
    print(f"  → Saved {fname_fevd}")

    # Print FEVD summary for GPR
    gpr_col = [c for c in df.columns if "GPR" in c.upper()]
    if gpr_col:
        print(f"\n  FEVD — Share explained by {gpr_col[0]}:")
        for _, row in fevd_df[fevd_df["horizon"] == 12].iterrows():
            gpr_share = row.get(f"pct_{gpr_col[0]}", 0)
            print(f"    {row['target']:25s} at h=12: {gpr_share:5.1f}%")

    return result, irf


def structural_break_test(df, break_date, var_cols, var_name):
    """
    Test for structural break around an event (Chow-style).
    Compares VAR coefficients pre vs post break date.
    """
    print(f"\n  --- Structural Break Test: {var_name} (break: {break_date}) ---")

    pre = df[df.index < break_date].dropna()
    post = df[df.index >= break_date].dropna()

    print(f"  Pre-break:  {len(pre)} obs ({pre.index[0].date()} to {pre.index[-1].date()})")
    print(f"  Post-break: {len(post)} obs ({post.index[0].date()} to {post.index[-1].date()})")

    if len(pre) < 30 or len(post) < 12:
        print("  ⚠ Insufficient observations for structural break test")
        return

    # Compare means
    results = []
    for col in var_cols:
        if col not in df.columns:
            continue
        pre_mean = pre[col].mean()
        post_mean = post[col].mean()
        pre_std = pre[col].std()
        post_std = post[col].std()

        # Welch's t-test for difference in means
        t_stat, p_val = stats.ttest_ind(
            pre[col].dropna(), post[col].dropna(), equal_var=False
        )

        results.append({
            "variable": col,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "diff": post_mean - pre_mean,
            "pre_std": pre_std,
            "post_std": post_std,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": "***" if p_val < 0.01 else "**" if p_val < 0.05 else
                           "*" if p_val < 0.10 else "",
        })

    results_df = pd.DataFrame(results)
    fname = f"var_structural_break_{var_name}.csv"
    results_df.to_csv(os.path.join(TAB_DIR, fname), index=False)
    print(f"  → Saved {fname}")

    for _, row in results_df.iterrows():
        print(f"    {row['variable']:25s} pre={row['pre_mean']:+8.4f}  "
              f"post={row['post_mean']:+8.4f}  diff={row['diff']:+8.4f}  "
              f"t={row['t_stat']:+6.2f} {row['significant']}")

    return results_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     VAR ANALYSIS                                       ║")
    print("║     Geopolitical Risk → Currency Markets               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load data
    daily = pd.read_csv(os.path.join(PROC_DIR, "master_daily.csv"),
                         index_col=0, parse_dates=True)
    print(f"\n  Loaded master_daily.csv: {daily.shape}")

    monthly_path = os.path.join(PROC_DIR, "master_monthly.csv")
    if os.path.exists(monthly_path):
        monthly_raw = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
        print(f"  Loaded master_monthly.csv: {monthly_raw.shape}")
    else:
        monthly_raw = pd.DataFrame()

    # ======================================================================
    # A. BUILD MONTHLY VAR DATASET
    # ======================================================================
    print("\n" + "=" * 60)
    print("BUILDING MONTHLY VAR DATASET")
    print("=" * 60)

    # Resample portfolio returns to monthly (sum of daily log returns = monthly log return)
    port_cols = [c for c in daily.columns if c.startswith("port_")]
    if port_cols:
        monthly_ports = resample_to_monthly(daily, port_cols, method="sum")
        print(f"  Monthly portfolio returns: {monthly_ports.shape}")
    else:
        monthly_ports = pd.DataFrame()

    # Resample market returns to monthly
    mkt_ret_cols = [c for c in daily.columns if c.startswith("ret_")]
    if mkt_ret_cols:
        monthly_mkt = resample_to_monthly(daily, mkt_ret_cols, method="sum")
        print(f"  Monthly market returns: {monthly_mkt.shape}")
    else:
        monthly_mkt = pd.DataFrame()

    # Monthly VIX (average level)
    vix_cols = [c for c in daily.columns if "VIX" in c and "lvl" in c]
    if vix_cols:
        monthly_vix = resample_to_monthly(daily, vix_cols, method="mean")
    else:
        monthly_vix = pd.DataFrame()

    # GPR monthly (already in monthly_raw)
    gpr_monthly_col = None
    if not monthly_raw.empty:
        gpr_candidates = [c for c in monthly_raw.columns if c in ["GPR", "GPRD"]]
        if gpr_candidates:
            gpr_monthly_col = gpr_candidates[0]

    # Combine monthly data
    monthly_frames = [monthly_ports, monthly_mkt, monthly_vix]
    if gpr_monthly_col and not monthly_raw.empty:
        monthly_frames.append(monthly_raw[[gpr_monthly_col]].rename(
            columns={gpr_monthly_col: "GPR"}
        ))

    # EPU from monthly master
    if not monthly_raw.empty:
        epu_cols = [c for c in monthly_raw.columns if "epu" in c.lower() or c == "EPU"]
        if epu_cols:
            monthly_frames.append(monthly_raw[epu_cols[:1]].rename(
                columns={epu_cols[0]: "EPU"}
            ))
        tpu_cols = [c for c in monthly_raw.columns
                    if "trade" in c.lower() or "tpu" in c.lower() or c == "TPU_monthly"]
        if tpu_cols:
            monthly_frames.append(monthly_raw[tpu_cols[:1]].rename(
                columns={tpu_cols[0]: "TPU"}
            ))

    monthly_all = pd.concat([f for f in monthly_frames if not f.empty], axis=1, join="outer")
    monthly_all = monthly_all.sort_index()
    print(f"  Combined monthly: {monthly_all.shape}")
    print(f"  Columns: {list(monthly_all.columns)}")

    # ======================================================================
    # B. STATIONARITY TESTS
    # ======================================================================
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS (ADF)")
    print("=" * 60)

    stationarity = check_stationarity(monthly_all)
    if not stationarity.empty:
        stationarity.to_csv(os.path.join(TAB_DIR, "var_adf_tests.csv"), index=False)
        for _, row in stationarity.iterrows():
            status = "✓ I(0)" if row["stationary"] else "✗ non-stationary"
            print(f"  {row['variable']:25s} ADF={row['ADF_stat']:+7.3f}  p={row['p_value']:.4f}  {status}")

        # Transform non-stationary series (first-difference GPR/EPU levels)
        non_stat = stationarity[~stationarity["stationary"]]["variable"].tolist()
        if non_stat:
            print(f"\n  Differencing non-stationary series: {non_stat}")
            for col in non_stat:
                if col in monthly_all.columns:
                    monthly_all[f"d_{col}"] = monthly_all[col].diff()
            # Re-check
            new_cols = [f"d_{c}" for c in non_stat if f"d_{c}" in monthly_all.columns]
            if new_cols:
                recheck = check_stationarity(monthly_all[new_cols])
                for _, row in recheck.iterrows():
                    status = "✓ I(0)" if row["stationary"] else "✗ still non-stationary"
                    print(f"    {row['variable']:25s} ADF={row['ADF_stat']:+7.3f}  p={row['p_value']:.4f}  {status}")

    # ======================================================================
    # C. GRANGER CAUSALITY
    # ======================================================================
    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY TESTS")
    print("=" * 60)

    # Does GPR Granger-cause currency portfolios?
    gpr_col_use = "d_GPR" if "d_GPR" in monthly_all.columns else "GPR"
    if gpr_col_use in monthly_all.columns:
        target_cols = [c for c in monthly_all.columns
                       if c.startswith("port_") or c.startswith("ret_")]

        granger_results = run_granger_tests(
            monthly_all, gpr_col_use, target_cols, max_lag=MAX_VAR_LAGS
        )
        if not granger_results.empty:
            granger_results.to_csv(os.path.join(TAB_DIR, "var_granger_gpr.csv"), index=False)
            print(f"\n  GPR → Currency/Market (Granger causality):")
            for _, row in granger_results.iterrows():
                print(f"    GPR → {row['effect']:25s} lag={row['best_lag']}  "
                      f"F={row['F_stat']:6.2f}  p={row['p_value']:.4f}  {row['significant_5pct']}")
        print(f"  → Saved var_granger_gpr.csv")
    else:
        print("  ⚠ GPR column not found in monthly data")

    # ======================================================================
    # D. VAR MODEL 1: BASELINE (GPR → safe, risky, RmS)
    # ======================================================================
    print("\n" + "=" * 60)
    print("VAR MODEL 1: BASELINE")
    print("=" * 60)

    var1_cols = []
    if gpr_col_use in monthly_all.columns:
        var1_cols.append(gpr_col_use)
    for c in ["port_safe", "port_risky", "port_RmS"]:
        if c in monthly_all.columns:
            var1_cols.append(c)

    if len(var1_cols) >= 2:
        var1_data = monthly_all[var1_cols].dropna()
        if len(var1_data) >= 50:
            estimate_var_and_irf(var1_data, "Baseline: GPR → Safe/Risky",
                                 n_periods=MAX_IRF_PERIODS, save_prefix="baseline")

    # ======================================================================
    # E. VAR MODEL 2: OIL CHANNEL (GPR → oil → exp/imp currencies)
    # ======================================================================
    print("\n" + "=" * 60)
    print("VAR MODEL 2: OIL CHANNEL")
    print("=" * 60)

    var2_cols = []
    if gpr_col_use in monthly_all.columns:
        var2_cols.append(gpr_col_use)
    for c in ["ret_Brent", "port_oil_exp", "port_oil_imp", "port_ExpMinusImp"]:
        if c in monthly_all.columns:
            var2_cols.append(c)

    if len(var2_cols) >= 3:
        var2_data = monthly_all[var2_cols].dropna()
        if len(var2_data) >= 50:
            estimate_var_and_irf(var2_data, "Oil Channel: GPR → Brent → Oil FX",
                                 n_periods=MAX_IRF_PERIODS, save_prefix="oil_channel")

    # ======================================================================
    # F. VAR MODEL 3: EXTENDED WITH EPU/TPU
    # ======================================================================
    print("\n" + "=" * 60)
    print("VAR MODEL 3: EXTENDED (with EPU/TPU controls)")
    print("=" * 60)

    var3_cols = []
    if gpr_col_use in monthly_all.columns:
        var3_cols.append(gpr_col_use)
    for c in ["EPU", "d_EPU", "TPU", "d_TPU"]:
        if c in monthly_all.columns:
            var3_cols.append(c)
            break  # use first available EPU variant
    for c in ["port_RmS", "ret_Brent", "port_dollar_basket"]:
        if c in monthly_all.columns:
            var3_cols.append(c)

    if len(var3_cols) >= 3:
        var3_data = monthly_all[var3_cols].dropna()
        if len(var3_data) >= 50:
            estimate_var_and_irf(var3_data, "Extended: GPR + EPU → FX",
                                 n_periods=MAX_IRF_PERIODS, save_prefix="extended")

    # ======================================================================
    # G. STRUCTURAL BREAK: PRE/POST LIBERATION DAY
    # ======================================================================
    print("\n" + "=" * 60)
    print("STRUCTURAL BREAK ANALYSIS")
    print("=" * 60)

    break_date = pd.Timestamp("2025-04-02")
    break_cols = [c for c in monthly_all.columns
                  if c.startswith("port_") or c in [gpr_col_use, "ret_Brent"]]

    if break_cols:
        structural_break_test(monthly_all, break_date, break_cols, "liberation_day")

    # ======================================================================
    # H. CORRELATION ANALYSIS (connects to Project 2)
    # ======================================================================
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    corr_cols = []
    for c in [gpr_col_use, "port_safe", "port_risky", "port_RmS",
              "ret_SP500", "ret_Gold", "ret_Brent", "ret_Bitcoin",
              "port_dollar_basket"]:
        if c in monthly_all.columns:
            corr_cols.append(c)

    if len(corr_cols) >= 3:
        corr_data = monthly_all[corr_cols].dropna()
        corr_matrix = corr_data.corr()
        corr_matrix.to_csv(os.path.join(TAB_DIR, "var_correlation_matrix.csv"))
        print(f"  → Saved var_correlation_matrix.csv")

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr_cols, fontsize=9)

        # Add correlation values
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_matrix.iloc[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Monthly Correlation Matrix: GPR, Portfolios, Markets")
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "var_correlation_heatmap.png"), dpi=150)
        plt.close(fig)
        print("  → Saved var_correlation_heatmap.png")

        # Print key correlations
        if gpr_col_use in corr_matrix.columns:
            print(f"\n  Correlations with {gpr_col_use}:")
            for c in corr_cols:
                if c != gpr_col_use:
                    print(f"    {c:25s} {corr_matrix.loc[gpr_col_use, c]:+.3f}")

    # ======================================================================
    # DONE
    # ======================================================================
    print("\n" + "=" * 60)
    print("VAR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures: {os.path.abspath(FIG_DIR)}")
    print(f"  Tables:  {os.path.abspath(TAB_DIR)}")


if __name__ == "__main__":
    main()
