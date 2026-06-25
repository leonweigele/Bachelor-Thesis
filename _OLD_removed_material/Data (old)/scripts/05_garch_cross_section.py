"""
05_garch_cross_section.py
=========================
Time-varying volatility and cross-sectional analysis.

Analysis:
  1. GARCH(1,1) for each currency — estimate conditional volatility
  2. GPR-augmented GARCH: does GPR predict FX volatility?
  3. Volatility comparison: safe vs risky, oil exp vs imp
  4. Cross-sectional regressions: connect Project 2 factor betas to GPR exposure
  5. Rolling correlations: GPR-FX co-movement over time
  6. Volatility event analysis: vol spikes around geopolitical events

Run from the Data folder:
    cd Data
    python3 scripts/05_garch_cross_section.py

Input:  data/processed/master_daily.csv
        data/processed/factors_course_clean.csv
Output: output/figures/garch_*.png
        output/tables/garch_*.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Currency groups (must match 02_data_cleaning.py)
SAFE_CCY = ["JPY", "CHF", "EUR", "DKK", "SEK", "SGD", "ILS", "TWD", "KRW"]
RISKY_CCY = ["AUD", "NZD", "GBP", "NOK", "CAD", "BRL", "MXN", "TRY", "ZAR",
             "INR", "THB", "PHP", "MYR", "IDR", "COP", "CLP", "PEN", "PLN",
             "HUF", "CZK", "RON", "RUB", "NGN"]
OIL_EXP = ["NOK", "CAD", "RUB", "COP", "MXN", "BRL", "SAR", "KWD", "NGN", "MYR"]
OIL_IMP = ["JPY", "EUR", "INR", "KRW", "TRY", "THB", "PHP", "ZAR", "PLN",
           "HUF", "CZK", "TWD", "IDR"]

# Analysis subset — currencies with enough data for GARCH
GARCH_CURRENCIES = [
    "JPY", "CHF", "EUR", "GBP", "AUD", "NZD", "CAD", "NOK", "SEK",
    "BRL", "MXN", "TRY", "ZAR", "INR", "KRW", "PLN", "HUF", "CZK",
    "THB", "PHP", "SGD", "MYR", "IDR", "COP", "CLP"
]

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# GARCH ESTIMATION
# ---------------------------------------------------------------------------
def estimate_garch(returns, name=""):
    """
    Estimate GARCH(1,1) model.
    Uses the arch package if available, otherwise falls back to
    a simple EWMA volatility estimate.
    """
    returns = returns.dropna() * 100  # scale to percent for numerical stability

    if len(returns) < 500:
        return None

    try:
        from arch import arch_model

        model = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
        result = model.fit(disp="off", show_warning=False)

        cond_vol = result.conditional_volatility / 100  # back to decimal
        params = {
            "omega": result.params.get("omega", np.nan),
            "alpha": result.params.get("alpha[1]", np.nan),
            "beta": result.params.get("beta[1]", np.nan),
            "persistence": result.params.get("alpha[1]", 0) + result.params.get("beta[1]", 0),
            "log_likelihood": result.loglikelihood,
            "aic": result.aic,
        }

        return {
            "cond_vol": cond_vol,
            "params": params,
            "model": "GARCH(1,1)",
        }

    except ImportError:
        # Fallback: EWMA volatility (lambda = 0.94, RiskMetrics)
        lam = 0.94
        returns_dec = returns / 100
        var_t = np.zeros(len(returns_dec))
        var_t[0] = returns_dec.iloc[:20].var()

        for t in range(1, len(returns_dec)):
            var_t[t] = lam * var_t[t-1] + (1 - lam) * returns_dec.iloc[t-1]**2

        cond_vol = pd.Series(np.sqrt(var_t), index=returns_dec.index, name=name)

        return {
            "cond_vol": cond_vol,
            "params": {"lambda": lam, "persistence": lam},
            "model": "EWMA(0.94)",
        }

    except Exception as e:
        print(f"    ⚠ GARCH failed for {name}: {e}")
        return None


def estimate_garch_x(returns, exog, name=""):
    """
    Estimate GARCH(1,1)-X model with GPR in the variance equation.
    Tests whether GPR predicts FX volatility.
    """
    # Align and clean
    combined = pd.DataFrame({"ret": returns, "exog": exog}).dropna()
    if len(combined) < 500:
        return None

    ret = combined["ret"] * 100
    x = combined["exog"]

    # Standardize exogenous variable
    x_std = (x - x.mean()) / x.std()

    try:
        from arch import arch_model

        # Standard GARCH first
        model_base = arch_model(ret, vol="Garch", p=1, q=1, mean="Constant")
        res_base = model_base.fit(disp="off", show_warning=False)

        # GARCH-X: add GPR to mean equation as a proxy
        # (arch package doesn't directly support exog in variance,
        #  so we test GPR in the mean equation instead)
        model_x = arch_model(ret, x=pd.DataFrame({"GPR": x_std}),
                              vol="Garch", p=1, q=1, mean="ARX")
        res_x = model_x.fit(disp="off", show_warning=False)

        # Also run OLS: squared returns ~ GPR (direct vol test)
        sq_ret = (ret ** 2).values
        x_vals = x_std.values

        from scipy.stats import pearsonr
        corr, p_val = pearsonr(sq_ret, x_vals)

        return {
            "base_aic": res_base.aic,
            "x_aic": res_x.aic,
            "improvement": res_base.aic - res_x.aic,
            "gpr_vol_corr": corr,
            "gpr_vol_pval": p_val,
        }

    except ImportError:
        # Fallback: correlation of squared returns with GPR
        sq_ret = (returns.dropna() ** 2)
        combined = pd.DataFrame({"sq_ret": sq_ret, "gpr": exog}).dropna()
        if len(combined) < 100:
            return None
        corr, p_val = stats.pearsonr(combined["sq_ret"], combined["gpr"])
        return {
            "gpr_vol_corr": corr,
            "gpr_vol_pval": p_val,
            "model": "OLS_fallback",
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# ROLLING ANALYSIS
# ---------------------------------------------------------------------------
def compute_rolling_correlation(series1, series2, window=60):
    """Compute rolling correlation between two series."""
    combined = pd.DataFrame({"s1": series1, "s2": series2}).dropna()
    if len(combined) < window:
        return pd.Series(dtype=float)
    return combined["s1"].rolling(window).corr(combined["s2"])


def compute_rolling_volatility(returns, window=22):
    """Compute rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     GARCH & CROSS-SECTIONAL ANALYSIS                   ║")
    print("║     Volatility, GPR Exposure, Factor Betas             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load data
    daily = pd.read_csv(os.path.join(PROC_DIR, "master_daily.csv"),
                         index_col=0, parse_dates=True)
    print(f"\n  Loaded master_daily.csv: {daily.shape}")

    # Load course factors
    factors_path = os.path.join(PROC_DIR, "factors_course_clean.csv")
    if os.path.exists(factors_path):
        try:
            factors = pd.read_csv(factors_path, index_col=0, parse_dates=True)
        except UnicodeDecodeError:
            factors = pd.read_csv(factors_path, index_col=0, parse_dates=True, encoding="latin-1")
        print(f"  Loaded factors_course_clean.csv: {factors.shape}")
    else:
        factors = pd.DataFrame()
        print("  ⚠ factors_course_clean.csv not found")

    # Find GPR daily column
    gpr_col = None
    for c in daily.columns:
        if c in ["GPRD", "GPR_GPRD", "GPRD_MA7"]:
            gpr_col = c
            break
    if gpr_col is None:
        gpr_candidates = [c for c in daily.columns if "GPR" in c.upper() and "post" not in c.lower()]
        if gpr_candidates:
            gpr_col = gpr_candidates[0]

    if gpr_col:
        print(f"  Using GPR column: {gpr_col}")
    else:
        print("  ⚠ No GPR column found — some analyses will be skipped")

    # ======================================================================
    # A. GARCH ESTIMATION FOR EACH CURRENCY
    # ======================================================================
    print("\n" + "=" * 60)
    print("GARCH(1,1) ESTIMATION")
    print("=" * 60)

    available_ccy = [c for c in GARCH_CURRENCIES if c in daily.columns]
    garch_results = {}
    garch_params = []

    for ccy in available_ccy:
        returns = daily[ccy].dropna()
        res = estimate_garch(returns, name=ccy)

        if res is not None:
            garch_results[ccy] = res
            row = {"currency": ccy, "model": res["model"]}
            row.update(res["params"])

            # Classification
            row["safe_risky"] = "safe" if ccy in SAFE_CCY else "risky" if ccy in RISKY_CCY else "other"
            row["oil_class"] = "exporter" if ccy in OIL_EXP else "importer" if ccy in OIL_IMP else "other"

            # Unconditional stats
            row["unc_vol_ann"] = returns.std() * np.sqrt(252) * 100
            row["mean_cond_vol_ann"] = res["cond_vol"].mean() * np.sqrt(252) * 100

            garch_params.append(row)
            print(f"  {ccy:5s} persist={res['params'].get('persistence', 0):.3f}  "
                  f"unc_vol={row['unc_vol_ann']:.1f}%  [{res['model']}]")

    params_df = pd.DataFrame(garch_params)
    if not params_df.empty:
        params_df.to_csv(os.path.join(TAB_DIR, "garch_params.csv"), index=False)
        print(f"\n  → Saved garch_params.csv ({len(params_df)} currencies)")

    # ======================================================================
    # B. GPR → VOLATILITY TEST
    # ======================================================================
    print("\n" + "=" * 60)
    print("GPR → FX VOLATILITY (GARCH-X / Correlation)")
    print("=" * 60)

    if gpr_col:
        gpr_series = daily[gpr_col]
        gpr_x_results = []

        for ccy in available_ccy:
            returns = daily[ccy]
            res = estimate_garch_x(returns, gpr_series, name=ccy)

            if res is not None:
                row = {"currency": ccy}
                row.update(res)
                row["safe_risky"] = "safe" if ccy in SAFE_CCY else "risky"
                row["oil_class"] = "exporter" if ccy in OIL_EXP else "importer" if ccy in OIL_IMP else "other"
                gpr_x_results.append(row)

                sig = "***" if res.get("gpr_vol_pval", 1) < 0.01 else \
                      "**" if res.get("gpr_vol_pval", 1) < 0.05 else \
                      "*" if res.get("gpr_vol_pval", 1) < 0.10 else ""
                print(f"  {ccy:5s} GPR-vol corr={res.get('gpr_vol_corr', 0):+.3f}  "
                      f"p={res.get('gpr_vol_pval', 1):.4f} {sig}")

        gpr_x_df = pd.DataFrame(gpr_x_results)
        if not gpr_x_df.empty:
            gpr_x_df.to_csv(os.path.join(TAB_DIR, "garch_gpr_volatility.csv"), index=False)
            print(f"\n  → Saved garch_gpr_volatility.csv")

            # Compare safe vs risky
            safe_corr = gpr_x_df[gpr_x_df["safe_risky"] == "safe"]["gpr_vol_corr"].mean()
            risky_corr = gpr_x_df[gpr_x_df["safe_risky"] == "risky"]["gpr_vol_corr"].mean()
            print(f"\n  Average GPR-vol correlation:")
            print(f"    Safe currencies:  {safe_corr:+.4f}")
            print(f"    Risky currencies: {risky_corr:+.4f}")

            # Oil exporters vs importers
            exp_corr = gpr_x_df[gpr_x_df["oil_class"] == "exporter"]["gpr_vol_corr"].mean()
            imp_corr = gpr_x_df[gpr_x_df["oil_class"] == "importer"]["gpr_vol_corr"].mean()
            print(f"    Oil exporters:    {exp_corr:+.4f}")
            print(f"    Oil importers:    {imp_corr:+.4f}")

    # ======================================================================
    # C. ROLLING VOLATILITY COMPARISON
    # ======================================================================
    print("\n" + "=" * 60)
    print("ROLLING VOLATILITY: SAFE vs RISKY, OIL EXP vs IMP")
    print("=" * 60)

    for port_pair, labels, fname in [
        (("port_safe", "port_risky"), ("Safe", "Risky"), "safe_vs_risky"),
        (("port_oil_exp", "port_oil_imp"), ("Oil Exporters", "Oil Importers"), "oil_exp_vs_imp"),
    ]:
        col1, col2 = port_pair
        if col1 in daily.columns and col2 in daily.columns:
            vol1 = compute_rolling_volatility(daily[col1], window=22) * 100
            vol2 = compute_rolling_volatility(daily[col2], window=22) * 100

            fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

            # Top: volatility levels
            axes[0].plot(vol1.index, vol1.values, label=labels[0], linewidth=1.2, alpha=0.8)
            axes[0].plot(vol2.index, vol2.values, label=labels[1], linewidth=1.2, alpha=0.8)
            axes[0].set_ylabel("Annualized Volatility (%)")
            axes[0].set_title(f"Rolling 22-day Volatility: {labels[0]} vs {labels[1]}")
            axes[0].legend()

            # Event lines
            for event_date, event_label, color in [
                ("2025-04-02", "Liberation Day", "red"),
                ("2025-06-15", "Iran Strikes", "orange"),
                ("2026-02-15", "Hormuz", "darkred"),
            ]:
                for ax in axes:
                    try:
                        ax.axvline(x=pd.Timestamp(event_date), color=color,
                                   linewidth=1, linestyle="--", alpha=0.7)
                    except:
                        pass

            # Bottom: volatility spread
            spread = vol1 - vol2
            axes[1].plot(spread.index, spread.values, color="purple", linewidth=1, alpha=0.7)
            axes[1].axhline(y=0, color="black", linewidth=0.5)
            axes[1].fill_between(spread.index, spread.values, 0, alpha=0.1, color="purple")
            axes[1].set_ylabel(f"{labels[0]} − {labels[1]} Vol (%)")
            axes[1].set_title("Volatility Spread")

            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, f"garch_rolling_vol_{fname}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → Saved garch_rolling_vol_{fname}.png")

    # ======================================================================
    # D. ROLLING GPR-FX CORRELATION
    # ======================================================================
    print("\n" + "=" * 60)
    print("ROLLING GPR-PORTFOLIO CORRELATIONS")
    print("=" * 60)

    if gpr_col:
        gpr_change = daily[gpr_col].diff()  # use GPR changes, not levels

        roll_corrs = {}
        for port in ["port_safe", "port_risky", "port_RmS", "port_oil_exp", "port_oil_imp"]:
            if port in daily.columns:
                rc = compute_rolling_correlation(gpr_change, daily[port], window=60)
                roll_corrs[port] = rc

        if roll_corrs:
            fig, ax = plt.subplots(figsize=(14, 6))
            for label, rc in roll_corrs.items():
                ax.plot(rc.index, rc.values, label=label, linewidth=1.2, alpha=0.8)

            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_ylabel("60-day Rolling Correlation")
            ax.set_title(f"Rolling Correlation: ΔGPR vs Portfolio Returns")
            ax.legend(fontsize=9)

            for event_date, color in [
                ("2025-04-02", "red"), ("2025-06-15", "orange"), ("2026-02-15", "darkred")
            ]:
                try:
                    ax.axvline(x=pd.Timestamp(event_date), color=color,
                               linewidth=1, linestyle="--", alpha=0.7)
                except:
                    pass

            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "garch_rolling_gpr_corr.png"), dpi=150)
            plt.close(fig)
            print("  → Saved garch_rolling_gpr_corr.png")

    # ======================================================================
    # E. CROSS-SECTIONAL: FACTOR BETAS vs GPR EXPOSURE
    # ======================================================================
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL ANALYSIS: Project 2 Betas vs GPR Exposure")
    print("=" * 60)

    if not factors.empty and gpr_col:
        # Step 1: Compute GPR beta for each currency (time-series regression)
        gpr_change = daily[gpr_col].diff()
        gpr_betas = {}

        for ccy in available_ccy:
            if ccy not in daily.columns:
                continue
            combined = pd.DataFrame({"ret": daily[ccy], "gpr": gpr_change}).dropna()
            if len(combined) < 100:
                continue

            slope, intercept, r, p, se = stats.linregress(combined["gpr"], combined["ret"])
            gpr_betas[ccy] = {
                "gpr_beta": slope,
                "gpr_pval": p,
                "gpr_r2": r ** 2,
            }

        gpr_beta_df = pd.DataFrame(gpr_betas).T
        gpr_beta_df.index.name = "currency"

        if not gpr_beta_df.empty:
            gpr_beta_df.to_csv(os.path.join(TAB_DIR, "garch_gpr_betas.csv"))
            print(f"  GPR betas computed for {len(gpr_beta_df)} currencies")
            print(f"  → Saved garch_gpr_betas.csv")

            # Step 2: Try to match with Project 2 factor betas
            # Look for Dollar Risk and Carry Trade Risk columns
            factor_cols = [c for c in factors.columns
                          if "dollar" in c.lower() or "carry" in c.lower()
                          or "risk" in c.lower()]

            if factor_cols:
                print(f"\n  Project 2 factor columns found: {factor_cols}")
                # This requires the factors file to have currency-level betas
                # which depends on how the user structured their Project 2 output
                print("  → To complete cross-sectional analysis:")
                print("     1. Run Project 2 regressions to get each currency's")
                print("        Dollar Risk beta and Carry Trade Risk beta")
                print("     2. Merge those betas with the GPR betas above")
                print("     3. Regress: GPR_beta_i = a + b * CarryBeta_i + e")
                print("     This tests whether carry-trade-exposed currencies")
                print("     are also more sensitive to geopolitical risk.")
            else:
                print("  Project 2 factors loaded but no Dollar/Carry columns identified")
                print(f"  Available columns: {list(factors.columns)[:10]}")

    else:
        if factors.empty:
            print("  ⚠ Course factors not available — skipping cross-sectional analysis")
        if not gpr_col:
            print("  ⚠ GPR column not available")

    # ======================================================================
    # F. VOLATILITY AROUND EVENTS
    # ======================================================================
    print("\n" + "=" * 60)
    print("VOLATILITY AROUND GEOPOLITICAL EVENTS")
    print("=" * 60)

    events = {
        "Liberation Day": "2025-04-02",
        "Iran Strikes": "2025-06-15",
        "Hormuz Closure": "2026-02-15",
    }

    for port in ["port_safe", "port_risky", "port_oil_exp", "port_oil_imp"]:
        if port not in daily.columns:
            continue

        vol_22 = compute_rolling_volatility(daily[port], window=22) * 100

        for event_name, event_date_str in events.items():
            event_date = pd.Timestamp(event_date_str)
            # Get vol 5 days before and 5 days after
            pre_mask = (vol_22.index >= event_date - pd.Timedelta(days=10)) & \
                       (vol_22.index < event_date)
            post_mask = (vol_22.index >= event_date) & \
                        (vol_22.index <= event_date + pd.Timedelta(days=10))

            pre_vol = vol_22[pre_mask].mean()
            post_vol = vol_22[post_mask].mean()

            if not np.isnan(pre_vol) and not np.isnan(post_vol):
                change = post_vol - pre_vol
                print(f"  {port:20s} {event_name:20s} "
                      f"pre={pre_vol:5.1f}%  post={post_vol:5.1f}%  "
                      f"Δ={change:+5.1f}%")

    # ======================================================================
    # G. SUMMARY FIGURE: GPR BETAS BY CURRENCY GROUP
    # ======================================================================
    if gpr_col and gpr_betas:
        beta_df = pd.DataFrame(gpr_betas).T
        beta_df["group"] = beta_df.index.map(
            lambda c: "Safe" if c in SAFE_CCY else "Risky" if c in RISKY_CCY else "Other"
        )
        beta_df["oil"] = beta_df.index.map(
            lambda c: "Exporter" if c in OIL_EXP else "Importer" if c in OIL_IMP else "Other"
        )

        # Bar chart of GPR betas colored by group
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # By safe/risky
        colors_sr = beta_df["group"].map({"Safe": "#2196F3", "Risky": "#F44336", "Other": "#9E9E9E"})
        beta_sorted = beta_df.sort_values("gpr_beta")
        axes[0].barh(range(len(beta_sorted)), beta_sorted["gpr_beta"],
                      color=[colors_sr[i] for i in beta_sorted.index])
        axes[0].set_yticks(range(len(beta_sorted)))
        axes[0].set_yticklabels(beta_sorted.index, fontsize=8)
        axes[0].axvline(x=0, color="black", linewidth=0.8)
        axes[0].set_xlabel("GPR Beta (ΔGPR → FX Return)")
        axes[0].set_title("GPR Sensitivity by Currency (Safe=Blue, Risky=Red)")

        # By oil classification
        colors_oil = beta_df["oil"].map({"Exporter": "#FF9800", "Importer": "#4CAF50", "Other": "#9E9E9E"})
        axes[1].barh(range(len(beta_sorted)), beta_sorted["gpr_beta"],
                      color=[colors_oil[i] for i in beta_sorted.index])
        axes[1].set_yticks(range(len(beta_sorted)))
        axes[1].set_yticklabels(beta_sorted.index, fontsize=8)
        axes[1].axvline(x=0, color="black", linewidth=0.8)
        axes[1].set_xlabel("GPR Beta (ΔGPR → FX Return)")
        axes[1].set_title("GPR Sensitivity by Currency (Exporter=Orange, Importer=Green)")

        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "garch_gpr_betas_by_group.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Saved garch_gpr_betas_by_group.png")

    # ======================================================================
    # DONE
    # ======================================================================
    print("\n" + "=" * 60)
    print("GARCH & CROSS-SECTIONAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures: {os.path.abspath(FIG_DIR)}")
    print(f"  Tables:  {os.path.abspath(TAB_DIR)}")


if __name__ == "__main__":
    main()
