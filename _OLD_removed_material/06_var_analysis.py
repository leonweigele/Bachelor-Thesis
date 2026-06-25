"""
06_var_analysis.py — baseline daily VAR: geopolitical risk -> dollar & baskets
==============================================================================
Chapter 6 (SVAR) starting point, ported to the current panel. Builds a small
recursive (Cholesky) VAR on stationary daily series and reports the two
headline objects: Granger-causality of geopolitical risk for the dollar and the
currency baskets, and the impulse responses to a GPR shock.

System (Cholesky order = most exogenous first):
    d_GPRD  ->  r_DCOILBRENTEU  ->  d_VIXCLS  ->  SAFE  ->  RISKY  ->  r_DTWEXBGS
(GPR is most exogenous; the dollar is most endogenous.)

Inputs : Data/processed/returns_daily.csv, Data/processed/portfolios_daily.csv
Outputs: Output/tables/var_lag_selection.csv, var_granger.csv, var_irf_gpr.csv
         Output/figures/var_irf_gpr.(png|pdf)

NOTE: baseline only. Bootstrapped IRF bands, FEVD, and the tariff-vs-Iran
subsample split (see Section 4.4 / Chapter 6 plan) are the next additions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
FIGS = ROOT / "Output/figures"
TABS = ROOT / "Output/tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

ORDER = ["d_GPRD", "r_DCOILBRENTEU", "d_VIXCLS", "SAFE", "RISKY", "r_DTWEXBGS"]
SHOCK = "d_GPRD"
MAXLAGS = 10
IRF_H = 15


def load_system():
    rets = pd.read_csv(PROC / "returns_daily.csv", parse_dates=["date"], index_col="date")
    ports = pd.read_csv(PROC / "portfolios_daily.csv", parse_dates=["date"], index_col="date")
    df = rets.join(ports[[c for c in ports.columns if c not in rets.columns]], how="outer")
    missing = [c for c in ORDER if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns for the VAR: {missing}. Run 02_build_returns.py.")
    return df[ORDER].dropna()


def main():
    Y = load_system()
    print(f"VAR system: {Y.shape[0]} obs x {Y.shape[1]} vars "
          f"({Y.index.min():%Y-%m-%d} -> {Y.index.max():%Y-%m-%d})")

    # stationarity (ADF) — all series are returns/first differences, expect I(0)
    adf = {c: adfuller(Y[c].dropna(), autolag="AIC")[1] for c in ORDER}
    print("  ADF p-values:", {c: round(p, 4) for c, p in adf.items()})

    model = VAR(Y)
    sel = model.select_order(MAXLAGS)
    pd.DataFrame({k: sel.ics[k] for k in sel.ics}).to_csv(TABS / "var_lag_selection.csv")
    p = sel.aic or 1
    print(f"  lag order (AIC): {p}")

    res = model.fit(p)

    # --- Granger causality: does each driver cause the dollar / baskets? -------
    rows = []
    for caused in ["r_DTWEXBGS", "SAFE", "RISKY"]:
        for causing in ["d_GPRD", "r_DCOILBRENTEU", "d_VIXCLS"]:
            if causing == caused:
                continue
            t = res.test_causality(caused, [causing], kind="f")
            rows.append({"caused": caused, "causing": causing,
                         "F": round(t.test_statistic, 2), "p": round(t.pvalue, 4),
                         "sig": "***" if t.pvalue < .01 else "**" if t.pvalue < .05
                         else "*" if t.pvalue < .10 else ""})
    gc = pd.DataFrame(rows)
    gc.to_csv(TABS / "var_granger.csv", index=False)
    print("\nGranger causality (driver -> target):")
    print(gc.to_string(index=False))

    # --- IRFs with bootstrapped 90% bands -------------------------------------
    irf = res.irf(IRF_H)
    cum = irf.orth_cum_effects                       # [h, response, shock], cumulative
    se = irf.cum_effect_stderr(orth=True)            # asymptotic s.e. of cumulative IRF
    z = 1.645
    lo, hi = cum - z * se, cum + z * se
    band = "90% asymptotic"

    iD = ORDER.index("r_DTWEXBGS")
    drivers = {"d_GPRD": "geopolitical-risk", "d_VIXCLS": "VIX (risk-sentiment)",
               "r_DCOILBRENTEU": "oil"}
    print(f"\nCumulative dollar response (r_DTWEXBGS) to a 1 s.d. shock, % [{band}]:")
    rows = []
    for s, lab in drivers.items():
        iS = ORDER.index(s)
        for h in (5, 10, IRF_H):
            sigstar = "" if (lo[h, iD, iS] <= 0 <= hi[h, iD, iS]) else "*"
            rows.append({"shock": s, "h": h, "cum_resp_%": round(cum[h, iD, iS] * 100, 3),
                         "lo_%": round(lo[h, iD, iS] * 100, 3),
                         "hi_%": round(hi[h, iD, iS] * 100, 3), "sig10": sigstar})
        print(f"  to {lab:22s} h={IRF_H}: {cum[IRF_H, iD, iS]*100:+.2f}"
              f"  [{lo[IRF_H, iD, iS]*100:+.2f}, {hi[IRF_H, iD, iS]*100:+.2f}]"
              f"{' (sig)' if rows[-1]['sig10'] else ' (n.s.)'}")
    pd.DataFrame(rows).to_csv(TABS / "var_irf_dollar.csv", index=False)

    # --- FEVD: share of dollar-return variance from each shock ----------------
    fevd = res.fevd(IRF_H + 1)
    dec = fevd.decomp[iD]                             # [horizon, shock]
    fevd_rows = [{"h": h, **{s: round(dec[h, ORDER.index(s)] * 100, 1) for s in ORDER}}
                 for h in (1, 5, 10, IRF_H)]
    pd.DataFrame(fevd_rows).to_csv(TABS / "var_fevd_dollar.csv", index=False)
    print(f"\nFEVD of the dollar at h={IRF_H} (% of variance from):")
    print("  " + "  ".join(f"{s}:{dec[IRF_H, ORDER.index(s)]*100:.1f}"
                           for s in ["d_GPRD", "r_DCOILBRENTEU", "d_VIXCLS"]))

    # --- figure: dollar response to GPR vs VIX shock, with bands --------------
    try:
        import matplotlib.pyplot as plt
        hh = range(IRF_H + 1)
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
        for ax, s, ttl in [(axes[0], "d_GPRD", "Geopolitical-risk shock"),
                           (axes[1], "d_VIXCLS", "VIX shock")]:
            iS = ORDER.index(s)
            ax.fill_between(hh, lo[:, iD, iS] * 100, hi[:, iD, iS] * 100,
                            color="tab:blue", alpha=.18)
            ax.plot(hh, cum[:, iD, iS] * 100, color="tab:blue", lw=1.6)
            ax.axhline(0, color="k", lw=.6)
            ax.set_title(ttl, fontsize=9.5); ax.set_xlabel("days")
        axes[0].set_ylabel("cumulative USD response, %")
        fig.suptitle(f"Broad dollar response to orthogonalised shocks ({band} bands)",
                     fontsize=9.5)
        fig.tight_layout()
        fig.savefig(FIGS / "var_irf_gpr.png", dpi=150)
        fig.savefig(FIGS / "var_irf_gpr.pdf")
        print("\nSaved Output/figures/var_irf_gpr.(png|pdf) + var_irf_dollar.csv + var_fevd_dollar.csv")
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
