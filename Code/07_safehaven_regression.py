"""
07_safehaven_regression.py — OLS dummy-interaction test of the safe-haven breakdown
===================================================================================
The simple, human-readable alternative to a DCC-GARCH. Tests whether the dollar's
safe-haven sensitivity to risk (the VIX) changed during the tariff shock vs the
Hormuz / Ukraine shocks.

Model (Newey-West / HAC standard errors, to handle daily heteroskedasticity):

  r^USD_t = a + b1 dVIX_t
              + b2 (dVIX_t x D_tariff_t)
              + b3 (dVIX_t x D_hormuz_t)
              + b4 (dVIX_t x D_ukraine_t)
              + controls (the level dummies) + e_t

Reading the coefficients (USD up = appreciation; dVIX up = risk-off):
  b1            normal safe-haven sensitivity         -> expect > 0 (risk-off -> USD up)
  b1 + b2       sensitivity during the TARIFF window   -> breakdown if << b1 (b2 < 0)
  b1 + b3       sensitivity during the HORMUZ window   -> haven holds if ~ b1
  b1 + b4       sensitivity during the UKRAINE window  -> haven holds if ~ b1

Crisis windows: event day .. +20 trading days (same horizon as the event study).

Inputs : Data/processed/returns_daily.csv, Data/processed/events.csv
Outputs: Output/tables/safehaven_regression.csv  (+ console summary)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
TABS = ROOT / "Output/tables"
TABS.mkdir(parents=True, exist_ok=True)
WIN = 20  # trading days after each event that count as the "crisis" window

rets = pd.read_csv(PROC / "returns_daily.csv", parse_dates=["date"], index_col="date")
ev = pd.read_csv(PROC / "events.csv")

d = rets[["r_DTWEXBGS", "d_VIXCLS"]].dropna().copy()
d.columns = ["usd", "vix"]
bdays = d.index


def window_dummy(date):
    pos = bdays.searchsorted(pd.Timestamp(date))
    dum = pd.Series(0.0, index=bdays)
    if pos < len(bdays):
        dum.iloc[pos:min(len(bdays), pos + WIN + 1)] = 1.0
    return dum.values


def edate(name):
    return ev.loc[ev.event == name, "date"].iloc[0]


for tag, name in [("tariff", "liberation_day"), ("hormuz", "hormuz"), ("ukraine", "ukraine")]:
    d[f"D_{tag}"] = window_dummy(edate(name))
    d[f"vix_{tag}"] = d["vix"] * d[f"D_{tag}"]

model = smf.ols(
    "usd ~ vix + vix_tariff + vix_hormuz + vix_ukraine + D_tariff + D_hormuz + D_ukraine",
    data=d,
).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

b, p = model.params, model.pvalues
print(model.summary())
print("\n================ SAFE-HAVEN SENSITIVITY OF THE DOLLAR TO A VIX SHOCK ================")
print(f"  normal (non-crisis)   b1           = {b['vix']:+.4f}  (p={p['vix']:.3f})")
print(f"  tariff  interaction   b2           = {b['vix_tariff']:+.4f}  (p={p['vix_tariff']:.3f})")
print(f"  hormuz  interaction   b3           = {b['vix_hormuz']:+.4f}  (p={p['vix_hormuz']:.3f})")
print(f"  ukraine interaction   b4           = {b['vix_ukraine']:+.4f}  (p={p['vix_ukraine']:.3f})")
print(f"  -> tariff-window sensitivity  b1+b2 = {b['vix']+b['vix_tariff']:+.4f}")
print(f"  -> hormuz-window sensitivity  b1+b3 = {b['vix']+b['vix_hormuz']:+.4f}")
print(f"  -> ukraine-window sensitivity b1+b4 = {b['vix']+b['vix_ukraine']:+.4f}")
print(f"  R^2 = {model.rsquared:.3f},  N = {int(model.nobs)}")

pd.DataFrame({"coef": b, "std_err": model.bse, "t": model.tvalues, "p": p}).round(4) \
    .to_csv(TABS / "safehaven_regression.csv")
print("\nSaved Output/tables/safehaven_regression.csv")
