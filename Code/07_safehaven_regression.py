"""
07_safehaven_regression.py — dummy-interaction tests of the safe-haven breakdown
================================================================================
The simple, human-readable alternative to a DCC-GARCH. Tests whether the
dollar's sensitivity to risk changed during the tariff shock vs the Hormuz /
Ukraine shocks — now across four risk measures (supervisor feedback,
Dalgic email 2026-07-22: joint VIX+GPR as lead suggestion; VXY and TPU as
alternatives).

Specs (all: level dummies included, Newey-West / HAC 5-lag standard errors):
  (A) dVIX only                  — original baseline spec
  (B) dVIX + dGPR jointly        — each interacted with the crisis dummies;
                                   VIX carries the trade shock, GPR the
                                   military/oil shocks
  (C) dVXY only                  — JPM VXY Global FX implied vol (mid quotes,
                                   manual Workspace export; bridges both types)
  (D) dGPR + dTPU jointly        — geopolitical + trade-policy news indices

Units: dollar return in PERCENT; every risk shock standardized to unit
variance over the full sample, so a coefficient reads "percent dollar move on
a one-standard-deviation risk-off day". t-stats are invariant to this scaling
(the raw-unit baseline b1 is printed for continuity with older drafts).

TPU: authors' currently published daily index (TPUD_index from
tpu_web_latest.xlsx, downloaded 2026-07-23). NOTE the vintage: this series is
a different construction from the pre-2026 daily file (corr ~0.6) — cite the
download date, as the authors request.

Crisis windows: event day .. +20 trading days (same horizon as the event
study).

Inputs : Data/processed/returns_daily.csv, Data/processed/events.csv,
         Data/raw/tpu_daily.csv
Outputs: Output/tables/safehaven_regression.csv   (long format, all specs)
         Main/LaTeX Thesis/content/tab_safehaven_regression.tex
         console summary + collinearity diagnostics
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"
TABS = ROOT / "Output/tables"
TEX = ROOT / "Main/LaTeX Thesis/content/tab_safehaven_regression.tex"
TABS.mkdir(parents=True, exist_ok=True)
WIN = 20                       # trading days after each event = "crisis" window

# ---------------------------------------------------------------- data
rets = pd.read_csv(PROC / "returns_daily.csv", parse_dates=["date"],
                   index_col="date")
ev = pd.read_csv(PROC / "events.csv")

tpu = pd.read_csv(ROOT / "Data/raw/tpu_daily.csv")
tpu["date"] = pd.to_datetime(dict(year=tpu.year, month=tpu.month, day=tpu.day))
tpu = tpu.set_index("date")["daily_tpu_index"].sort_index()
d_tpu = tpu.reindex(rets.index).ffill().diff()      # align to trading days

d = pd.DataFrame({
    "usd": rets["r_DTWEXBGS"] * 100,               # percent
    "vix": rets["d_VIXCLS"],
    "gpr": rets["d_GPRD"],
    "vxy": rets["d_VXY_Global"],
    "tpu": d_tpu,
}).dropna(subset=["usd", "vix", "gpr"])

MEASURES = ["vix", "gpr", "vxy", "tpu"]
NICE = {"vix": "$\\Delta$VIX", "gpr": "$\\Delta$GPR",
        "vxy": "$\\Delta$VXY", "tpu": "$\\Delta$TPU"}

RAW_VIX_SD = d["vix"].std()
for m in MEASURES:                                  # unit variance
    d[m] = d[m] / d[m].std()

bdays = d.index


def window_dummy(date):
    pos = bdays.searchsorted(pd.Timestamp(date))
    dum = pd.Series(0.0, index=bdays)
    if pos < len(bdays):
        dum.iloc[pos:min(len(bdays), pos + WIN + 1)] = 1.0
    return dum


EVENTS = [("tariff", "liberation_day"), ("hormuz", "hormuz"),
          ("ukraine", "ukraine")]
for tag, name in EVENTS:
    date = ev.loc[ev.event == name, "date"].iloc[0]
    d[f"D_{tag}"] = window_dummy(date)
    for m in MEASURES:
        d[f"{m}_{tag}"] = d[m] * d[f"D_{tag}"]

# ---------------------------------------------------------------- diagnostics
print("=" * 74)
print("Collinearity of the standardized daily risk shocks (full sample):")
corr = d[MEASURES].corr().round(2)
print(corr.to_string())
print("(rule of thumb: joint specs are unproblematic below ~0.8)")

# ---------------------------------------------------------------- specs
SPECS = {
    "A: VIX":       ["vix"],
    "B: VIX + GPR": ["vix", "gpr"],
    "C: VXY":       ["vxy"],
    "D: GPR + TPU": ["gpr", "tpu"],
}


def fit(measures):
    terms = []
    for m in measures:
        terms += [m] + [f"{m}_{t}" for t, _ in EVENTS]
    terms += [f"D_{t}" for t, _ in EVENTS]
    sub = d.dropna(subset=measures)
    model = smf.ols("usd ~ " + " + ".join(terms), data=sub) \
               .fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model


rows = []
results = {}
for label, ms in SPECS.items():
    model = fit(ms)
    results[label] = model
    for term in model.params.index:
        rows.append({"spec": label, "term": term,
                     "coef": model.params[term], "se": model.bse[term],
                     "t": model.tvalues[term], "p": model.pvalues[term]})
    rows.append({"spec": label, "term": "_stats", "coef": model.rsquared,
                 "se": np.nan, "t": np.nan, "p": model.nobs})

pd.DataFrame(rows).round(5).to_csv(TABS / "safehaven_regression.csv",
                                   index=False)

# ---------------------------------------------------------------- console
print("\n" + "=" * 74)
print("Sensitivity of the daily broad-dollar return (%) to a 1-SD risk shock")
print("=" * 74)
for label, model in results.items():
    b, t = model.params, model.tvalues
    ms = SPECS[label]
    print(f"\n--- Spec {label}   (N={int(model.nobs)}, R2={model.rsquared:.3f})")
    for m in ms:
        line = f"  d{m.upper():5s} normal {b[m]:+.4f} (t={t[m]:+.1f})"
        for tag, _ in EVENTS:
            k = f"{m}_{tag}"
            line += f" | x{tag} {b[k]:+.4f} (t={t[k]:+.1f})"
        print(line)
        wins = "   ".join(f"{tag}: {b[m] + b[f'{m}_{tag}']:+.4f}"
                          for tag, _ in EVENTS)
        print(f"         within-window slopes -> {wins}")

bA = results["A: VIX"].params
print(f"\nContinuity check, spec A in raw units: b1 = "
      f"{bA['vix'] / RAW_VIX_SD / 100:+.5f} per VIX point "
      f"(old draft quoted +0.0003)")

# ---------------------------------------------------------------- tex table


def stars(p):
    return "^{***}" if p < .01 else "^{**}" if p < .05 else \
           "^{*}" if p < .10 else ""


def cell(label, term):
    model = results[label]
    if term not in model.params.index:
        return "---"
    return (f"${model.params[term]:+.3f}{stars(model.pvalues[term])}$"
            f"\\;({model.tvalues[term]:+.1f})")


cols = list(SPECS)
lines = [
    "% Auto-generated by Code/07_safehaven_regression.py — do not edit by hand.",
    r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}",
    r"\caption{Did the dollar's safe-haven reflex survive each shock?"
    r" (OLS, HAC standard errors)}",
    r"\label{tab:safehaven_reg}", r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{l cccc}", r"\toprule",
    " & " + " & ".join(f"({c.split(':')[0]}) {c.split(': ')[1]}"
                       for c in cols) + r" \\",
    r"\midrule",
]
for m in MEASURES:
    block = [
        (f"{NICE[m]} (normal)", m),
        (r"\quad $\times$ Tariff", f"{m}_tariff"),
        (r"\quad $\times$ Hormuz", f"{m}_hormuz"),
        (r"\quad $\times$ Ukraine", f"{m}_ukraine"),
    ]
    if all(cell(c, m) == "---" for c in cols):
        continue
    for name, term in block:
        lines.append(name + " & " +
                     " & ".join(cell(c, term) for c in cols) + r" \\")
    lines.append(r"\addlinespace")
lines += [
    r"\midrule",
    "Window dummies & " + " & ".join(["Yes"] * len(cols)) + r" \\",
    "$N$ & " + " & ".join(str(int(results[c].nobs)) for c in cols) + r" \\",
    "$R^2$ & " + " & ".join(f"{results[c].rsquared:.3f}"
                            for c in cols) + r" \\",
    r"\bottomrule", r"\end{tabular}",
    r"\begin{tablenotes}[flushleft]\footnotesize",
    r"\item \textit{Notes.} OLS of the daily broad-dollar return (percent) on"
    r" daily risk shocks, their interactions with crisis-window dummies, and"
    r" the window (level) dummies; Newey--West (HAC, 5 lags) standard errors,"
    r" $t$-statistics in parentheses. Each risk shock is standardized to unit"
    r" variance, so coefficients read as the percent dollar move on a"
    r" one-standard-deviation risk-off day. Crisis windows run from the event"
    r" day to $+20$ trading days. VIX: CBOE (FRED \texttt{VIXCLS}); GPR:"
    r" \textcite{caldara_iacoviello_2022} daily index; VXY: J.P.\,Morgan VXY"
    r" Global FX implied volatility (LSEG \texttt{.JPMVXYGL}, mid); TPU:"
    r" \textcite{caldara_etal_2020_tpu} daily index, current vintage"
    r" (downloaded 23~July 2026). A positive slope is safe-haven behaviour"
    r" (risk up $\rightarrow$ dollar up); a negative interaction weakens it."
    r" $^{*}/^{**}/^{***}$: 10/5/1\%.",
    r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", "",
]
TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {TEX.relative_to(ROOT)} and Output/tables/safehaven_regression.csv")
