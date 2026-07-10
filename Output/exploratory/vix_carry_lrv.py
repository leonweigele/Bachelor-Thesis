# EXPLORATORY — not thesis material yet.
# LRV (2011) global-risk check: correlation of carry/safe/risky portfolio
# returns with daily VIX changes, full sample and event windows.
# t-stat: r*sqrt((n-2)/(1-r^2)); |t|>~2 => sig. at 5%.
# Refs: Lustig, Roussanov & Verdelhan (2011, RFS); Brunnermeier, Nagel &
# Pedersen (2008). Run from Data/processed/.

import numpy as np
import pandas as pd

PROC = "."  # Data/processed
OUT = "../../Output/exploratory"
VARS = ["CARRY_HML", "SAFE", "RISKY", "USD_EW"]

p = pd.read_csv(f"{PROC}/portfolios_daily.csv", parse_dates=["date"])
r = pd.read_csv(f"{PROC}/returns_daily.csv", parse_dates=["date"])
df = p.merge(r[["date", "d_VIXCLS"]], on="date").dropna(
    subset=VARS + ["d_VIXCLS"]
)

ev = pd.read_csv(f"{PROC}/events.csv", parse_dates=["win_start", "win_end"])
win = {e: (s, t) for e, s, t in zip(ev.event, ev.win_start, ev.win_end)}

lib = df.date.between(*win["liberation_day"])
hor = df.date.between(*win["hormuz"])

samples = {
    "full_sample": df,
    "excl_both_windows": df[~lib & ~hor],
    "liberation_day_win": df[lib],
    "hormuz_win": df[hor],
}


def tstat(rho, n):
    return rho * np.sqrt((n - 2) / (1 - rho**2))


rows = []
for name, d in samples.items():
    n = len(d)
    for var in VARS:
        pr = d[var].corr(d["d_VIXCLS"])
        # Spearman via rank-Pearson (pandas delegates spearman to scipy)
        sr = d[var].rank().corr(d["d_VIXCLS"].rank())
        rows.append(dict(sample=name, var=var, n=n,
                         pearson=round(pr, 3), t_pearson=round(tstat(pr, n), 2),
                         spearman=round(sr, 3), t_spearman=round(tstat(sr, n), 2)))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/vix_carry_lrv_results.csv", index=False)
print(res.to_string(index=False))
print("\nsample ranges:")
for name, d in samples.items():
    print(f"  {name}: {d.date.min().date()} .. {d.date.max().date()}")
