"""
regression_sensitivity.py -- diagnostic companion to Code/regression/07_safehaven_regression.py
=====================================================================================
Read-only diagnostic. Reads the three FROZEN inputs of the thesis regression from
./inputs/ and writes everything to ./out/. It never touches the thesis repository.

Inputs (byte-identical copies of the repository files, hashes recorded in out/):
  inputs/returns_daily.csv           Data/processed/returns_daily.csv   (2026-08-11)
  inputs/events.csv                  Data/processed/events.csv          (2026-09-16, hormuz_closure removed)
  inputs/tpu_daily.csv               Data/raw/tpu_daily.csv             (2026-07-23)
  inputs/safehaven_regression.csv    Output/tables/safehaven_regression.csv (shipped)
  inputs/tab_safehaven_regression.tex content/tab_safehaven_regression.tex  (shipped)

Blocks
  1  Reproduce specs A-D exactly as 07_safehaven_regression.py builds them
     (same dropna, same standardisation, same window dummies, HAC(5)) and tie
     every coefficient/SE/t/p to the shipped CSV and every printed cell to the
     shipped .tex table.
  2  Covariance estimators on the IDENTICAL point estimates: classical, HC0-HC3,
     Newey-West with 0..21 lags and the automatic 4(T/100)^(2/9) rule. Includes
     the mechanism check (autocovariances of the interaction score inside the
     21-day window) and the direct 21-observation within-window regression.
  3  Influence: DFBETAS / Cook's D from the full OLS, leave-one-out over every
     tariff-window day, drop-two and drop-three, robust (Huber) within-window
     slope, rank correlation inside vs outside the window. Coefficient changes
     are reported separately from significance changes.
  4  Placebo windows: the tariff window is moved to every 21-day stretch of the
     sample that does not overlap the three modelled windows (Hormuz and Ukraine
     stay in the model). Coefficient-based randomisation p-values (SE-free) and
     nominal rejection rates for each SE estimator. Same exercise for the Hormuz
     and Ukraine windows. Sub-designs: non-overlapping blocks, quiet-days set
     (other dated episodes excluded), volatility-matched set, COVID excluded.
  5  Summary table: what changes the estimated relationship, what changes only
     its stars.

Run:  python regression_sensitivity.py        (about 40 seconds)

Placed inside the repository as Code/regression/<name>.py with no inputs/ folder next to it, the
script reads the frozen inputs at their repository paths and writes to
Output/tables/regression_sensitivity/ instead. It never writes anything else.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
if (HERE / "inputs").is_dir():
    # temporary-folder mode: byte copies of the frozen inputs sit next to the script
    INP = HERE / "inputs"
    OUT = HERE / "out"
    PATHS = {f: INP / f for f in ["returns_daily.csv", "events.csv", "tpu_daily.csv",
                                  "safehaven_regression.csv", "tab_safehaven_regression.tex"]}
else:
    # repository mode (script placed in Code/regression/): read the frozen inputs in place and
    # write under Output/, never next to the code
    ROOT = HERE.parents[1]
    OUT = ROOT / "Output/tables/regression_sensitivity"
    PATHS = {"returns_daily.csv": ROOT / "Data/processed/returns_daily.csv",
             "events.csv": ROOT / "Data/processed/events.csv",
             "tpu_daily.csv": ROOT / "Data/raw/tpu_daily.csv",
             "safehaven_regression.csv": ROOT / "Output/tables/safehaven_regression.csv",
             "tab_safehaven_regression.tex": ROOT / "Main/LaTeX Thesis/content/tab_safehaven_regression.tex"}
OUT.mkdir(parents=True, exist_ok=True)

WIN = 20                          # as in 07: event day .. +20 trading days
HAC_LAGS = 5                      # as shipped
MEASURES = ["vix", "gpr", "vxy", "tpu"]
SPECS = {"A: VIX": ["vix"], "B: VIX + GPR": ["vix", "gpr"],
         "C: VXY": ["vxy"], "D: GPR + TPU": ["gpr", "tpu"]}
EVENTS = [("tariff", "liberation_day"), ("hormuz", "hormuz"),
          ("ukraine", "ukraine")]
TAGS = [t for t, _ in EVENTS]

log_lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    log_lines.append(s)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- 0
log("=" * 78)
log("0. Provenance")
log("=" * 78)
prov = {"python": sys.version.split()[0], "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "statsmodels": sm.__version__ if hasattr(sm, "__version__") else
        __import__("statsmodels").__version__}
for f in ["returns_daily.csv", "events.csv", "tpu_daily.csv",
          "safehaven_regression.csv", "tab_safehaven_regression.tex"]:
    prov[f] = sha(PATHS[f])
    log(f"  {f:32s} sha256 {prov[f]}")
log(f"  python {prov['python']}, pandas {prov['pandas']}, numpy {prov['numpy']}, "
    f"statsmodels {prov['statsmodels']}")
(OUT / "provenance.json").write_text(json.dumps(prov, indent=2))

# --------------------------------------------------------------------------- 1
log()
log("=" * 78)
log("1. Reproduce the shipped specifications from the frozen inputs")
log("=" * 78)

# --- data block copied line for line from 07_safehaven_regression.py --------
rets = pd.read_csv(PATHS["returns_daily.csv"], parse_dates=["date"],
                   index_col="date")
ev = pd.read_csv(PATHS["events.csv"])
tpu = pd.read_csv(PATHS["tpu_daily.csv"])
tpu["date"] = pd.to_datetime(dict(year=tpu.year, month=tpu.month, day=tpu.day))
tpu = tpu.set_index("date")["daily_tpu_index"].sort_index()
d_tpu = tpu.reindex(rets.index).ffill().diff()

d = pd.DataFrame({
    "usd": rets["r_DTWEXBGS"] * 100,
    "vix": rets["d_VIXCLS"],
    "gpr": rets["d_GPRD"],
    "vxy": rets["d_VXY_Global"],
    "tpu": d_tpu,
}).dropna(subset=["usd", "vix", "gpr"])
RAW_SD = {m: d[m].std() for m in MEASURES}
for m in MEASURES:
    d[m] = d[m] / d[m].std()
bdays = d.index


def window_dummy(date):
    pos = bdays.searchsorted(pd.Timestamp(date))
    dum = pd.Series(0.0, index=bdays)
    if pos < len(bdays):
        dum.iloc[pos:min(len(bdays), pos + WIN + 1)] = 1.0
    return dum


WPOS = {}
for tag, name in EVENTS:
    date = ev.loc[ev.event == name, "date"].iloc[0]
    d[f"D_{tag}"] = window_dummy(date)
    WPOS[tag] = int(bdays.searchsorted(pd.Timestamp(date)))
    for m in MEASURES:
        d[f"{m}_{tag}"] = d[m] * d[f"D_{tag}"]
# --- end of copied block -----------------------------------------------------

N_ALL = len(d)
log(f"  rows after dropna(usd, vix, gpr): {N_ALL}  "
    f"({bdays[0]:%Y-%m-%d} .. {bdays[-1]:%Y-%m-%d})")
for tag in TAGS:
    p = WPOS[tag]
    log(f"  window {tag:8s}: rows {p}..{p + WIN} = "
        f"{bdays[p]:%Y-%m-%d} .. {bdays[p + WIN]:%Y-%m-%d}  "
        f"({int(d[f'D_{tag}'].sum())} days)")
log("  raw SD of the risk shocks (unit of the standardisation): " +
    ", ".join(f"{m} {RAW_SD[m]:.4f}" for m in MEASURES))


def terms_for(measures):
    terms = []
    for m in measures:
        terms += [m] + [f"{m}_{t}" for t in TAGS]
    terms += [f"D_{t}" for t in TAGS]
    return terms


def fit_formula(measures, cov_type="HAC", cov_kwds=None):
    """Exactly the 07 fit()."""
    terms = terms_for(measures)
    sub = d.dropna(subset=measures)
    if cov_type == "HAC":
        return smf.ols("usd ~ " + " + ".join(terms), data=sub).fit(
            cov_type="HAC", cov_kwds=cov_kwds or {"maxlags": HAC_LAGS})
    if cov_type == "nonrobust":
        return smf.ols("usd ~ " + " + ".join(terms), data=sub).fit()
    return smf.ols("usd ~ " + " + ".join(terms), data=sub).fit(cov_type=cov_type)


def design(measures, frame=None):
    frame = d if frame is None else frame
    sub = frame.dropna(subset=measures)
    X = sm.add_constant(sub[terms_for(measures)], has_constant="add")
    return sub["usd"], X


shipped = pd.read_csv(PATHS["safehaven_regression.csv"])
base = {}
rows = []
maxdev = {"coef": 0, "se": 0, "t": 0, "p": 0}
for label, ms in SPECS.items():
    res = fit_formula(ms)
    base[label] = res
    s = shipped[shipped.spec == label].set_index("term")
    for term in res.params.index:
        rec = {"spec": label, "term": term, "coef": res.params[term],
               "se": res.bse[term], "t": res.tvalues[term], "p": res.pvalues[term],
               "coef_shipped": s.loc[term, "coef"], "se_shipped": s.loc[term, "se"],
               "t_shipped": s.loc[term, "t"], "p_shipped": s.loc[term, "p"]}
        for k in ("coef", "se", "t", "p"):
            rec[f"dev_{k}"] = abs(round(rec[k], 5) - rec[f"{k}_shipped"])
            maxdev[k] = max(maxdev[k], rec[f"dev_{k}"])
        rows.append(rec)
    rec = {"spec": label, "term": "_stats", "coef": res.rsquared, "p": res.nobs,
           "coef_shipped": s.loc["_stats", "coef"], "p_shipped": s.loc["_stats", "p"]}
    rows.append(rec)
    log(f"  {label:14s} N={int(res.nobs)}  R2={res.rsquared:.5f}  "
        f"(shipped {s.loc['_stats','coef']:.5f}, N {int(s.loc['_stats','p'])})")
repro = pd.DataFrame(rows)
repro.to_csv(OUT / "01_baseline_reproduction.csv", index=False)
log(f"  max |deviation| vs shipped CSV after rounding to 5 dp: "
    + ", ".join(f"{k} {v:.0e}" for k, v in maxdev.items()))
REPRO_OK = all(v <= 1.5e-5 for v in maxdev.values())
log(f"  -> baseline reproduction {'EXACT (to CSV precision)' if REPRO_OK else 'DEVIATES'}")

# tie the printed .tex cells: coef 3 dp, t 2 dp, stars from p
tex = PATHS["tab_safehaven_regression.tex"].read_text()
cell_re = re.compile(r"\$([+-]?[\d.]+)(\^\{(\*+)\})?\$\\,\(\$([+-]?[\d.]+)\$\)")
phantom_re = re.compile(r"\\phantom\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
row_map = {r"$\Delta$VIX (normal)": "vix", r"$\Delta$GPR (normal)": "gpr",
           r"$\Delta$VXY (normal)": "vxy", r"$\Delta$TPU (normal)": "tpu"}
tex_checks = 0
tex_fail = []
current_m = None
cols = list(SPECS)
for line in tex.splitlines():
    for k, v in row_map.items():
        if line.startswith(k):
            current_m = v
    if current_m is None or " & " not in line:
        continue
    head, *cells = [c.strip() for c in line.rstrip(" \\").split("&")]
    if head in row_map:
        term_suffix = ""
    elif "Tariff" in head:
        term_suffix = "_tariff"
    elif "Hormuz" in head:
        term_suffix = "_hormuz"
    elif "Ukraine" in head:
        term_suffix = "_ukraine"
    else:
        continue
    term = current_m + term_suffix
    for col, cell in zip(cols, cells):
        if cell == "---":
            continue
        m = cell_re.search(phantom_re.sub("", cell))
        if not m:
            tex_fail.append((term, col, cell, "unparsed"))
            continue
        coef_txt, _, stars_txt, t_txt = m.groups()
        res = base[col]
        if term not in res.params.index:
            tex_fail.append((term, col, cell, "term missing"))
            continue
        p = res.pvalues[term]
        want_stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
        ok = (abs(float(coef_txt) - round(res.params[term], 3)) < 1e-9 and
              abs(float(t_txt) - round(res.tvalues[term], 2)) < 1e-9 and
              (stars_txt or "") == want_stars)
        tex_checks += 1
        if not ok:
            tex_fail.append((term, col, cell,
                             f"have {res.params[term]:+.3f}{want_stars} ({res.tvalues[term]:+.2f})"))
log(f"  .tex table cells tied: {tex_checks} checked, {len(tex_fail)} mismatches")
for f in tex_fail:
    log(f"     MISMATCH {f}")

# within-window slope identity (b1 + beta_k == subsample OLS slope)
log("  within-window slope identity (b1 + beta_k vs 21-day subsample OLS slope):")
for label, ms in [("A: VIX", "vix"), ("C: VXY", "vxy")]:
    res = base[label]
    for tag in TAGS:
        sub = d[d[f"D_{tag}"] == 1]
        slope = np.polyfit(sub[ms], sub["usd"], 1)[0]
        log(f"     {label} {tag:8s} b1+beta = {res.params[ms] + res.params[f'{ms}_{tag}']:+.5f}"
            f"   subsample slope = {slope:+.5f}")

# --------------------------------------------------------------------------- 2
log()
log("=" * 78)
log("2. Covariance estimators on the identical point estimates")
log("=" * 78)
T = N_ALL
auto_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
log(f"  automatic Newey-West rule 4(T/100)^(2/9) with T={T}: {auto_lag} lags")

cov_rows = []
KEY_TERMS = {"A: VIX": ["vix", "vix_tariff", "vix_hormuz", "vix_ukraine"],
             "B: VIX + GPR": ["vix", "vix_tariff", "vix_hormuz", "vix_ukraine",
                              "gpr", "gpr_tariff"],
             "C: VXY": ["vxy", "vxy_tariff", "vxy_hormuz", "vxy_ukraine"],
             "D: GPR + TPU": ["gpr", "gpr_tariff", "tpu_tariff"]}
for label, ms in SPECS.items():
    y, X = design(ms)
    ols = sm.OLS(y, X)
    fits = {"classical": ols.fit()}
    for hc in ["HC0", "HC1", "HC2", "HC3"]:
        fits[hc] = ols.fit(cov_type=hc)
    for L in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 21]:
        fits[f"HAC({L})"] = ols.fit(cov_type="HAC", cov_kwds={"maxlags": L})
    fits[f"HAC({auto_lag}) auto"] = ols.fit(cov_type="HAC", cov_kwds={"maxlags": auto_lag})
    for term in KEY_TERMS[label]:
        for est, r in fits.items():
            cov_rows.append({"spec": label, "term": term, "estimator": est,
                             "coef": r.params[term], "se": r.bse[term],
                             "t": r.tvalues[term], "p": r.pvalues[term]})
cov = pd.DataFrame(cov_rows)
cov.to_csv(OUT / "02_covariance_estimators.csv", index=False)

show_est = ["classical", "HC0", "HC1", "HC3", "HAC(0)", "HAC(2)", "HAC(5)",
            f"HAC({auto_lag}) auto", "HAC(10)", "HAC(21)"]
for label in SPECS:
    log(f"\n  {label}")
    log("  " + f"{'term':13s}{'coef':>9s}" + "".join(f"{e:>15s}" for e in show_est))
    for term in KEY_TERMS[label]:
        sub = cov[(cov.spec == label) & (cov.term == term)].set_index("estimator")
        line = f"  {term:13s}{sub['coef'].iloc[0]:+9.4f}"
        for e in show_est:
            line += f"{sub.loc[e, 't']:+8.2f}(p{sub.loc[e, 'p']:.3f})"[:15].rjust(15)
        log(line)
    # SE view for the tariff interaction
    tt = [t for t in KEY_TERMS[label] if t.endswith("_tariff") and not t.startswith("gpr") and not t.startswith("tpu")]
    for term in tt:
        sub = cov[(cov.spec == label) & (cov.term == term)].set_index("estimator")
        log(f"  {term} standard errors: " + ", ".join(
            f"{e} {sub.loc[e, 'se']:.4f}" for e in ["classical", "HC0", "HC3", "HAC(5)", "HAC(21)"]))

# lag path for the tariff interactions
log("\n  Newey-West t of the tariff interaction by lag length:")
for label, term in [("A: VIX", "vix_tariff"), ("B: VIX + GPR", "vix_tariff"),
                    ("C: VXY", "vxy_tariff")]:
    sub = cov[(cov.spec == label) & (cov.term == term)].set_index("estimator")
    path = [(L, sub.loc[f"HAC({L})", "t"]) for L in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 21]]
    mono = all(path[i + 1][1] <= path[i][1] for i in range(len(path) - 1))
    log(f"     {label:14s} {term:11s} " + " ".join(f"L{L}:{t:+.2f}" for L, t in path)
        + f"   monotone decreasing: {mono}")

# mechanism: autocovariances of the interaction score x_t e_t inside the window
log("\n  Mechanism check: sample autocovariances of the score s_t = x_t*e_t for the")
log("  tariff-interaction regressor (nonzero on 21 days only), Bartlett weights w_j=1-j/(L+1):")
mech_rows = []
for label, ms, term in [("A: VIX", ["vix"], "vix_tariff"), ("C: VXY", ["vxy"], "vxy_tariff")]:
    y, X = design(ms)
    r = sm.OLS(y, X).fit()
    s = (X[term] * r.resid).to_numpy()
    n = len(s)
    g0 = float(np.sum(s * s) / n)
    gam = [float(np.sum(s[j:] * s[:-j]) / n) for j in range(1, 22)]
    cum = g0
    line = f"     {label}: gamma_0={g0:.3e}"
    for j in [1, 2, 3, 4, 5]:
        line += f"  g{j}/g0={gam[j-1]/g0:+.3f}"
    log(line)
    for L in [0, 5, auto_lag, 21]:
        w = [1 - j / (L + 1) for j in range(1, L + 1)]
        lrv = g0 + 2 * sum(wj * gam[j] for j, wj in enumerate(w))
        mech_rows.append({"spec": label, "L": L, "lrv_over_gamma0": lrv / g0})
    log("       long-run-variance / gamma_0 at L=0,5,auto,21: " + ", ".join(
        f"{m['L']}:{m['lrv_over_gamma0']:.3f}" for m in mech_rows if m['spec'] == label))
pd.DataFrame(mech_rows).to_csv(OUT / "02b_hac_mechanism.csv", index=False)

# the honest yardstick: a 21-observation regression inside the window
log("\n  Direct within-window regression (21 observations, usd ~ risk shock):")
ww_rows = []
for label, m in [("A: VIX", "vix"), ("C: VXY", "vxy")]:
    for tag in TAGS:
        sub = d[d[f"D_{tag}"] == 1]
        Xw = sm.add_constant(sub[m])
        rw = sm.OLS(sub["usd"], Xw)
        f_cl, f_h3 = rw.fit(), rw.fit(cov_type="HC3")
        # normal-day slope for comparison
        out = d[(d.D_tariff + d.D_hormuz + d.D_ukraine) == 0]
        rn = sm.OLS(out["usd"], sm.add_constant(out[m])).fit(cov_type="HC3")
        diff = f_cl.params[m] - rn.params[m]
        se_diff_h3 = np.sqrt(f_h3.bse[m] ** 2 + rn.bse[m] ** 2)
        ww_rows.append({"spec": label, "window": tag, "n": len(sub),
                        "within_slope": f_cl.params[m], "se_classical": f_cl.bse[m],
                        "se_HC3": f_h3.bse[m], "normal_slope": rn.params[m],
                        "normal_se_HC3": rn.bse[m], "slope_diff": diff,
                        "t_diff_HC3": diff / se_diff_h3,
                        "hac5_se_interaction": base[label].bse[f"{m}_{tag}"]})
        log(f"     {label} {tag:8s}: within slope {f_cl.params[m]:+.4f} "
            f"(SE classical {f_cl.bse[m]:.4f}, HC3 {f_h3.bse[m]:.4f}); normal-day slope "
            f"{rn.params[m]:+.4f} (HC3 {rn.bse[m]:.4f}); difference {diff:+.4f}, "
            f"t(HC3) {diff / se_diff_h3:+.2f}; shipped HAC(5) SE of interaction "
            f"{base[label].bse[f'{m}_{tag}']:.4f}")
pd.DataFrame(ww_rows).to_csv(OUT / "02c_within_window_regressions.csv", index=False)

# --------------------------------------------------------------------------- 3
log()
log("=" * 78)
log("3. Influential tariff-window observations")
log("=" * 78)

infl_rows = []
for label, ms, term in [("A: VIX", ["vix"], "vix_tariff"),
                        ("B: VIX + GPR", ["vix", "gpr"], "vix_tariff"),
                        ("C: VXY", ["vxy"], "vxy_tariff")]:
    y, X = design(ms)
    r = sm.OLS(y, X).fit()
    inf = r.get_influence()
    k = list(X.columns).index(term)
    dfb = pd.Series(inf.dfbetas[:, k], index=y.index)
    cook = pd.Series(inf.cooks_distance[0], index=y.index)
    lev = pd.Series(inf.hat_matrix_diag, index=y.index)
    win = d[d.D_tariff == 1].index
    log(f"\n  {label}: DFBETAS for {term}, tariff-window days ranked by |DFBETAS|")
    log(f"     {'date':10s} {'day':>4s} {'usd%':>7s} {'shock(sd)':>9s} {'DFBETAS':>8s} {'CookD':>7s} {'lev':>6s}")
    tab = pd.DataFrame({"usd": d.loc[win, "usd"], "shock": d.loc[win, ms[0]],
                        "dfbetas": dfb[win], "cooks": cook[win], "leverage": lev[win]})
    tab["day"] = [int(bdays.get_loc(i)) - WPOS["tariff"] for i in tab.index]
    tab = tab.reindex(tab.dfbetas.abs().sort_values(ascending=False).index)
    for i, rr in tab.iterrows():
        log(f"     {i:%Y-%m-%d} {int(rr.day):>+4d} {rr.usd:+7.2f} {rr.shock:+9.2f} {rr.dfbetas:+8.3f} {rr.cooks:7.4f} {rr.leverage:6.3f}")
    # largest |DFBETAS| outside the window (should be small)
    outside = dfb.drop(win).abs().max()
    log(f"     largest |DFBETAS| outside the window: {outside:.3f}")
    tab.assign(spec=label, term=term).to_csv(OUT / f"03_dfbetas_{label[0]}.csv")

    # leave-one-out over the 21 window days (refit HAC(5) and HC3)
    full_b = base[label].params[term]
    full_t = base[label].tvalues[term]
    full_h3 = sm.OLS(y, X).fit(cov_type="HC3").tvalues[term]
    b1 = base[label].params[ms[0]]
    log(f"\n  Leave-one-out over the 21 tariff-window days ({label}): full beta {full_b:+.4f}, "
        f"t HAC(5) {full_t:+.2f}, t HC3 {full_h3:+.2f}, within-window slope {b1 + full_b:+.4f}")
    log(f"     {'dropped':10s} {'day':>4s} {'beta':>8s} {'%chg':>6s} {'slope':>8s} {'tHAC5':>7s} {'tHC3':>6s}")
    for day in win:
        yy, XX = y.drop(day), X.drop(day)
        m5 = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        m3 = sm.OLS(yy, XX).fit(cov_type="HC3")
        b = m5.params[term]
        infl_rows.append({"spec": label, "term": term, "dropped": day,
                          "day": int(bdays.get_loc(day)) - WPOS["tariff"],
                          "beta": b, "pct_change": 100 * (b - full_b) / full_b,
                          "within_slope": m5.params[ms[0]] + b,
                          "t_HAC5": m5.tvalues[term], "p_HAC5": m5.pvalues[term],
                          "t_HC3": m3.tvalues[term], "p_HC3": m3.pvalues[term]})
        rr = infl_rows[-1]
        flag = " <-- |t HAC5| < 1.96" if abs(rr["t_HAC5"]) < 1.96 else ""
        log(f"     {day:%Y-%m-%d} {rr['day']:>+4d} {b:+8.4f} {rr['pct_change']:+6.0f} {rr['within_slope']:+8.4f} "
            f"{rr['t_HAC5']:+7.2f} {rr['t_HC3']:+6.2f}{flag}")
    loo = pd.DataFrame([r for r in infl_rows if r["spec"] == label])
    log(f"     range of beta over LOO: {loo.beta.min():+.4f} .. {loo.beta.max():+.4f}; "
        f"days with |t HAC5| < 1.96: {(loo.t_HAC5.abs() < 1.96).sum()} of 21; "
        f"with |t HAC5| < 2.58: {(loo.t_HAC5.abs() < 2.58).sum()} of 21; "
        f"sign of beta always negative: {(loo.beta < 0).all()}")

    # drop the two / three most influential (by |DFBETAS|) and the two announcement days
    def refit_drop(days, note):
        yy, XX = y.drop(days), X.drop(days)
        m5 = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        m3 = sm.OLS(yy, XX).fit(cov_type="HC3")
        log(f"     drop {note:38s}: beta {m5.params[term]:+.4f}, within slope "
            f"{m5.params[ms[0]] + m5.params[term]:+.4f}, t HAC5 {m5.tvalues[term]:+.2f}, t HC3 {m3.tvalues[term]:+.2f}")
        return {"spec": label, "term": term, "dropped": note, "beta": m5.params[term],
                "within_slope": m5.params[ms[0]] + m5.params[term],
                "t_HAC5": m5.tvalues[term], "t_HC3": m3.tvalues[term]}
    top = list(tab.index[:3])
    multi = [refit_drop(top[:2], f"two largest |DFBETAS| ({top[0]:%d %b}, {top[1]:%d %b})"),
             refit_drop(top[:3], f"three largest |DFBETAS| (+{top[2]:%d %b})"),
             refit_drop([win[0], win[1]], "days 0 and +1 (2 and 3 Apr 2025)"),
             refit_drop([win[5]], "day +5 (9 Apr 2025, tariff pause)"),
             refit_drop([win[1], win[2]], "days +1 and +2 (3 and 4 Apr 2025)")]
    pd.DataFrame(multi).to_csv(OUT / f"03_dropsets_{label[0]}.csv", index=False)

    # robust and rank-based within-window checks (relationship, not stars)
    sub = d[d.D_tariff == 1]
    out = d[(d.D_tariff + d.D_hormuz + d.D_ukraine) == 0]
    hub_w = sm.RLM(sub["usd"], sm.add_constant(sub[ms[0]]), M=sm.robust.norms.HuberT()).fit()
    hub_n = sm.RLM(out["usd"], sm.add_constant(out[ms[0]]), M=sm.robust.norms.HuberT()).fit()
    rho_w = stats.spearmanr(sub[ms[0]], sub["usd"])
    rho_n = stats.spearmanr(out[ms[0]], out["usd"])
    log(f"     Huber within-window slope {hub_w.params[ms[0]]:+.4f} vs normal-day Huber slope "
        f"{hub_n.params[ms[0]]:+.4f}; Spearman rho within window {rho_w.statistic:+.3f} "
        f"(p={rho_w.pvalue:.2f}, n=21) vs normal days {rho_n.statistic:+.3f} (p={rho_n.pvalue:.1e})")
    infl_rows.append({"spec": label, "term": term, "dropped": "_huber_within_slope",
                      "beta": hub_w.params[ms[0]] - hub_n.params[ms[0]],
                      "within_slope": hub_w.params[ms[0]]})
pd.DataFrame(infl_rows).to_csv(OUT / "03_leave_one_out.csv", index=False)

# --------------------------------------------------------------------------- 4
log()
log("=" * 78)
log("4. Placebo windows")
log("=" * 78)
log("  Design: the window under test is removed from the model and replaced by a")
log("  pseudo-window of the same length (21 trading days) starting at every position s")
log("  of the sample such that the pseudo-window does not overlap ANY of the three")
log("  modelled windows. The two other real windows stay in the model with their own")
log("  level and interaction dummies. For each s the full regression is re-estimated")
log("  and the pseudo-interaction coefficient, its HAC(5) t, HC3 t and classical t")
log("  are stored. Consecutive pseudo-windows share 20 of 21 days, so the placebo")
log("  estimates are strongly dependent; non-overlapping-block versions are reported.")

real_pos = {tag: set(range(WPOS[tag], WPOS[tag] + WIN + 1)) for tag in TAGS}
all_real = set().union(*real_pos.values())
# other dated episodes in events.csv (kept as 'normal' days by the model)
other_pos = set()
for name in ["tariff_pause", "iran_12day", "hormuz_ceasefire", "us_strikes"]:
    date = ev.loc[ev.event == name, "date"].iloc[0]
    p = int(bdays.searchsorted(pd.Timestamp(date)))
    other_pos |= set(range(p, min(N_ALL, p + WIN + 1)))
covid_pos = set(np.where((bdays >= "2020-02-01") & (bdays <= "2020-06-30"))[0])


def placebo(label, m, target):
    """Move window `target` to every admissible start; other windows stay."""
    others = [t for t in TAGS if t != target]
    base_terms = [m] + [f"{m}_{t}" for t in others] + [f"D_{t}" for t in others]
    sub = d.dropna(subset=[m])
    idx = sub.index
    y = sub["usd"].to_numpy()
    Xb = sm.add_constant(sub[base_terms], has_constant="add").to_numpy()
    x = sub[m].to_numpy()
    n = len(y)
    out = []
    for s in range(0, n - WIN):
        win = set(range(s, s + WIN + 1))
        if win & all_real:
            continue
        D = np.zeros(n)
        D[s:s + WIN + 1] = 1.0
        X = np.column_stack([Xb, x * D, D])
        j = X.shape[1] - 2
        ols = sm.OLS(y, X)
        r0 = ols.fit()
        r5 = ols.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        r3 = ols.fit(cov_type="HC3")
        out.append({"start": s, "start_date": idx[s], "end_date": idx[s + WIN],
                    "beta": r0.params[j], "se_cl": r0.bse[j], "t_cl": r0.tvalues[j],
                    "se_hac5": r5.bse[j], "t_hac5": r5.tvalues[j],
                    "se_hc3": r3.bse[j], "t_hc3": r3.tvalues[j],
                    "quiet": not (win & other_pos), "covid": bool(win & covid_pos),
                    "shock_sd": float(np.std(x[s:s + WIN + 1], ddof=1)),
                    "usd_sd": float(np.std(y[s:s + WIN + 1], ddof=1))})
    return pd.DataFrame(out)


def placebo_fourth_window(m):
    """Alternative design: all three real windows stay in the model and the
    pseudo-window is ADDED as a fourth window, so the pseudo-interaction is measured
    against exactly the same normal-day slope as the real tariff interaction."""
    base_terms = [m] + [f"{m}_{t}" for t in TAGS] + [f"D_{t}" for t in TAGS]
    sub = d.dropna(subset=[m])
    y = sub["usd"].to_numpy()
    Xb = sm.add_constant(sub[base_terms], has_constant="add").to_numpy()
    x = sub[m].to_numpy()
    n = len(y)
    out = []
    for s in range(0, n - WIN):
        win = set(range(s, s + WIN + 1))
        if win & all_real:
            continue
        D = np.zeros(n)
        D[s:s + WIN + 1] = 1.0
        X = np.column_stack([Xb, x * D, D])
        j = X.shape[1] - 2
        ols = sm.OLS(y, X)
        r0, r5, r3 = ols.fit(), ols.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}), ols.fit(cov_type="HC3")
        out.append({"start": s, "beta": r0.params[j], "t_cl": r0.tvalues[j], "se_cl": r0.bse[j],
                    "t_hac5": r5.tvalues[j], "se_hac5": r5.bse[j], "t_hc3": r3.tvalues[j], "se_hc3": r3.bse[j],
                    "shock_sd": float(np.std(x[s:s + WIN + 1], ddof=1))})
    return pd.DataFrame(out)


def one_sided(series, actual):
    return (series <= actual).mean() if actual < 0 else (series >= actual).mean()


def summarise(pl, actual, note):
    """actual = dict(beta, t5, t3, tcl)."""
    n = len(pl)
    res = {
        "n_placebo": n,
        "rej5_hac5": (pl.t_hac5.abs() > 1.96).mean(), "rej1_hac5": (pl.t_hac5.abs() > 2.576).mean(),
        "rej5_hc3": (pl.t_hc3.abs() > 1.96).mean(), "rej1_hc3": (pl.t_hc3.abs() > 2.576).mean(),
        "rej5_cl": (pl.t_cl.abs() > 1.96).mean(), "rej1_cl": (pl.t_cl.abs() > 2.576).mean(),
        "sd_beta": pl.beta.std(), "med_se_hac5": pl.se_hac5.median(),
        "med_se_hc3": pl.se_hc3.median(), "med_se_cl": pl.se_cl.median(),
        "p_two_sided_beta": (pl.beta.abs() >= abs(actual["beta"])).mean(),
        "p_one_sided_beta": one_sided(pl.beta, actual["beta"]),
        "p_two_sided_t_hac5": (pl.t_hac5.abs() >= abs(actual["t5"])).mean(),
        "p_one_sided_t_hac5": one_sided(pl.t_hac5, actual["t5"]),
        "p_two_sided_t_hc3": (pl.t_hc3.abs() >= abs(actual["t3"])).mean(),
        "p_one_sided_t_hc3": one_sided(pl.t_hc3, actual["t3"]),
        "p_two_sided_t_cl": (pl.t_cl.abs() >= abs(actual["tcl"])).mean(),
        "p_one_sided_t_cl": one_sided(pl.t_cl, actual["tcl"]),
        "beta_q01": pl.beta.quantile(.01), "beta_q05": pl.beta.quantile(.05),
        "beta_q95": pl.beta.quantile(.95), "beta_q99": pl.beta.quantile(.99),
    }
    log(f"     [{note}] n={n}: nominal-5% rejection HAC(5) {res['rej5_hac5']:.1%}, HC3 {res['rej5_hc3']:.1%}, "
        f"classical {res['rej5_cl']:.1%}; nominal-1% HAC(5) {res['rej1_hac5']:.1%}, HC3 {res['rej1_hc3']:.1%}, classical {res['rej1_cl']:.1%}")
    log(f"        placebo SD of beta {res['sd_beta']:.4f} vs median SE: HAC(5) {res['med_se_hac5']:.4f}, "
        f"HC3 {res['med_se_hc3']:.4f}, classical {res['med_se_cl']:.4f}")
    log(f"        randomisation p two-sided/one-sided: coefficient {res['p_two_sided_beta']:.3f}/{res['p_one_sided_beta']:.3f}; "
        f"|t| HAC(5) {res['p_two_sided_t_hac5']:.3f}/{res['p_one_sided_t_hac5']:.3f}; "
        f"|t| HC3 {res['p_two_sided_t_hc3']:.3f}/{res['p_one_sided_t_hc3']:.3f}; "
        f"|t| classical {res['p_two_sided_t_cl']:.3f}/{res['p_one_sided_t_cl']:.3f}")
    log(f"        placebo beta quantiles 1/5/95/99%: {res['beta_q01']:+.4f} {res['beta_q05']:+.4f} "
        f"{res['beta_q95']:+.4f} {res['beta_q99']:+.4f}   (actual {actual['beta']:+.4f})")
    return res


log("\n  Within-window standard deviation of the standardised risk shock (identification strength):")
for m in ["vix", "vxy", "gpr"]:
    log(f"     {m}: " + ", ".join(f"{tag} {d.loc[d[f'D_{tag}'] == 1, m].std():.2f}" for tag in TAGS)
        + "  (full sample 1.00 by construction)")

pl_summ = []
placebo_store = {}
PLACEBO_RUNS = [("A: VIX", ["vix"], "vix", TAGS),
                ("B: VIX + GPR", ["vix", "gpr"], "vix", ["tariff"]),
                ("C: VXY", ["vxy"], "vxy", TAGS)]
for label, ms, m, targets in PLACEBO_RUNS:
    for target in targets:
        term = f"{m}_{target}"
        y0, X0 = design(ms)
        actual = {"beta": base[label].params[term], "t5": base[label].tvalues[term],
                  "t3": sm.OLS(y0, X0).fit(cov_type="HC3").tvalues[term],
                  "tcl": sm.OLS(y0, X0).fit().tvalues[term]}
        ab = actual["beta"]
        if label.startswith("B"):
            # spec B keeps the GPR block fixed; placebo moves only the VIX-tariff window
            others = [t for t in TAGS if t != target]
            sub = d.dropna(subset=ms)
            fixed = ["vix"] + [f"vix_{t}" for t in others] + ["gpr"] + [f"gpr_{t}" for t in TAGS] + [f"D_{t}" for t in others]
            yv = sub["usd"].to_numpy()
            Xb = sm.add_constant(sub[fixed], has_constant="add").to_numpy()
            xv = sub["vix"].to_numpy()
            nn = len(yv)
            recs = []
            for s in range(0, nn - WIN):
                win = set(range(s, s + WIN + 1))
                if win & all_real:
                    continue
                D = np.zeros(nn)
                D[s:s + WIN + 1] = 1.0
                X = np.column_stack([Xb, xv * D, D])
                j = X.shape[1] - 2
                ols = sm.OLS(yv, X)
                r0, r5, r3 = ols.fit(), ols.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}), ols.fit(cov_type="HC3")
                recs.append({"start": s, "start_date": sub.index[s], "end_date": sub.index[s + WIN],
                             "beta": r0.params[j], "se_cl": r0.bse[j], "t_cl": r0.tvalues[j],
                             "se_hac5": r5.bse[j], "t_hac5": r5.tvalues[j], "se_hc3": r3.bse[j],
                             "t_hc3": r3.tvalues[j], "quiet": not (win & other_pos),
                             "covid": bool(win & covid_pos),
                             "shock_sd": float(np.std(xv[s:s + WIN + 1], ddof=1)),
                             "usd_sd": float(np.std(yv[s:s + WIN + 1], ddof=1))})
            pl = pd.DataFrame(recs)
        else:
            pl = placebo(label, m, target)
        placebo_store[(label, target)] = pl
        pl.to_csv(OUT / f"04_placebo_{label[0]}_{target}.csv", index=False)
        sd_actual = d.loc[d[f"D_{target}"] == 1, m].std()
        pct = (pl.shock_sd < sd_actual).mean()
        log(f"\n  {label}, window moved: {target} (actual beta {ab:+.4f}, t HAC(5) {actual['t5']:+.2f}, "
            f"t HC3 {actual['t3']:+.2f}, t classical {actual['tcl']:+.2f}); within-window shock SD {sd_actual:.2f} "
            f"= percentile {pct:.1%} of placebo windows")
        variants = [("all admissible", pl)]
        if target == "tariff":
            variants += [
                ("quiet: other dated episodes excluded", pl[pl.quiet]),
                ("COVID Feb-Jun 2020 excluded", pl[~pl.covid]),
                ("shock SD >= half the tariff window's", pl[pl.shock_sd >= 0.5 * sd_actual]),
                ("top-decile shock SD", pl[pl.shock_sd >= pl.shock_sd.quantile(.9)]),
                ("top-5% shock SD", pl[pl.shock_sd >= pl.shock_sd.quantile(.95)]),
            ]
            # non-overlapping blocks, all 21 offsets
            blk = []
            for off in range(WIN + 1):
                b = pl[(pl.start - off) % (WIN + 1) == 0]
                blk.append({"offset": off, "n": len(b),
                            "rej5_hac5": (b.t_hac5.abs() > 1.96).mean(),
                            "rej5_hc3": (b.t_hc3.abs() > 1.96).mean(),
                            "rej5_cl": (b.t_cl.abs() > 1.96).mean(),
                            "p_two_sided_t_hac5": (b.t_hac5.abs() >= abs(actual["t5"])).mean(),
                            "p_two_sided_t_cl": (b.t_cl.abs() >= abs(actual["tcl"])).mean(),
                            "sd_beta": b.beta.std()})
            blk = pd.DataFrame(blk)
            blk.to_csv(OUT / f"04_placebo_blocks_{label[0]}.csv", index=False)
            log(f"     [non-overlapping blocks, 21 offsets] n per offset {blk.n.min()}-{blk.n.max()}: "
                f"rejection HAC(5) {blk.rej5_hac5.min():.1%}-{blk.rej5_hac5.max():.1%} (mean {blk.rej5_hac5.mean():.1%}), "
                f"HC3 {blk.rej5_hc3.min():.1%}-{blk.rej5_hc3.max():.1%} (mean {blk.rej5_hc3.mean():.1%}), "
                f"classical mean {blk.rej5_cl.mean():.1%}; two-sided p on |t| HAC(5) "
                f"{blk.p_two_sided_t_hac5.min():.3f}-{blk.p_two_sided_t_hac5.max():.3f} (mean {blk.p_two_sided_t_hac5.mean():.3f}), "
                f"on classical |t| mean {blk.p_two_sided_t_cl.mean():.3f}")
        if target == "tariff" and not label.startswith("B"):
            p4 = placebo_fourth_window(m)
            p4.to_csv(OUT / f"04_placebo_fourthwindow_{label[0]}.csv", index=False)
            variants.append(("fourth-window design (real tariff window kept)", p4))
            variants.append(("fourth-window design, top-decile shock SD",
                             p4[p4.shock_sd >= p4.shock_sd.quantile(.9)]))
        for note, sub in variants:
            r = summarise(sub, actual, note)
            r.update({"spec": label, "target": target, "variant": note,
                      "actual_beta": ab, "actual_t_hac5": actual["t5"], "actual_t_hc3": actual["t3"],
                      "actual_t_cl": actual["tcl"], "actual_shock_sd": sd_actual, "shock_sd_percentile": pct})
            pl_summ.append(r)
        # what is in the tail: the largest |beta| placebos and their identification strength
        tail = pl.reindex(pl.beta.abs().sort_values(ascending=False).index).head(10)
        log(f"     ten largest |beta| placebo windows (start date, beta, within-window shock SD, t HAC5): " + "; ".join(
            f"{r.start_date:%Y-%m-%d} {r.beta:+.2f} sd{r.shock_sd:.2f} t{r.t_hac5:+.1f}" for r in tail.itertuples()))
        tail_t = pl.reindex(pl.t_hac5.abs().sort_values(ascending=False).index).head(8)
        log(f"     eight largest |t HAC5| placebo windows: " + "; ".join(
            f"{r.start_date:%Y-%m-%d} beta{r.beta:+.3f} t{r.t_hac5:+.1f} (HC3 {r.t_hc3:+.1f})" for r in tail_t.itertuples()))
pd.DataFrame(pl_summ).to_csv(OUT / "04_placebo_summary.csv", index=False)

# --------------------------------------------------------------------------- 5
log()
log("=" * 78)
log("5. Relationship versus significance: summary of the tariff interaction")
log("=" * 78)
summ = []
for label, m in [("A: VIX", "vix"), ("C: VXY", "vxy")]:
    term = f"{m}_tariff"
    r = base[label]
    c = cov[(cov.spec == label) & (cov.term == term)].set_index("estimator")
    loo = pd.DataFrame([x for x in infl_rows if x["spec"] == label and isinstance(x["dropped"], pd.Timestamp)])
    pl = placebo_store[(label, "tariff")]
    summ.append({
        "spec": label, "beta": r.params[term], "normal_slope": r.params[m],
        "within_slope": r.params[m] + r.params[term],
        "t_hac5": c.loc["HAC(5)", "t"], "t_hc3": c.loc["HC3", "t"], "t_hc0": c.loc["HC0", "t"],
        "t_classical": c.loc["classical", "t"],
        "loo_beta_min": loo.beta.min(), "loo_beta_max": loo.beta.max(),
        "loo_within_slope_min": loo.within_slope.min(), "loo_within_slope_max": loo.within_slope.max(),
        "loo_days_t_hac5_below_1.96": int((loo.t_HAC5.abs() < 1.96).sum()),
        "loo_days_t_hac5_below_2.58": int((loo.t_HAC5.abs() < 2.58).sum()),
        "placebo_p_two_sided_t_hac5": (pl.t_hac5.abs() >= abs(c.loc["HAC(5)", "t"])).mean(),
        "placebo_p_one_sided_t_hac5": (pl.t_hac5 <= c.loc["HAC(5)", "t"]).mean(),
        "placebo_p_two_sided_t_hc3": (pl.t_hc3.abs() >= abs(c.loc["HC3", "t"])).mean(),
        "placebo_p_two_sided_t_cl": (pl.t_cl.abs() >= abs(c.loc["classical", "t"])).mean(),
        "placebo_p_one_sided_t_cl": (pl.t_cl <= c.loc["classical", "t"]).mean(),
        "placebo_rej5_hac5": (pl.t_hac5.abs() > 1.96).mean(),
        "placebo_rej5_hc3": (pl.t_hc3.abs() > 1.96).mean(),
        "placebo_rej5_cl": (pl.t_cl.abs() > 1.96).mean(),
        "placebo_sd_beta": pl.beta.std(),
    })
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "05_summary_tariff_interaction.csv", index=False)
for _, s in summ.iterrows():
    log(f"\n  {s.spec}: beta_tariff {s.beta:+.4f} (normal slope {s.normal_slope:+.4f}, within-window slope {s.within_slope:+.4f})")
    log(f"     stars: t HAC(5) {s.t_hac5:+.2f} | HC0 {s.t_hc0:+.2f} | HC3 {s.t_hc3:+.2f} | classical {s.t_classical:+.2f}  (coefficient identical)")
    log(f"     LOO beta range {s.loo_beta_min:+.4f}..{s.loo_beta_max:+.4f}; within-window slope range "
        f"{s.loo_within_slope_min:+.4f}..{s.loo_within_slope_max:+.4f}; days pushing |t HAC5| below 1.96: "
        f"{s['loo_days_t_hac5_below_1.96']}/21, below 2.58: {s['loo_days_t_hac5_below_2.58']}/21")
    log(f"     placebo randomisation p two-sided/one-sided: |t| HAC(5) {s.placebo_p_two_sided_t_hac5:.3f}/{s.placebo_p_one_sided_t_hac5:.3f}, "
        f"|t| HC3 {s.placebo_p_two_sided_t_hc3:.3f}, classical |t| {s.placebo_p_two_sided_t_cl:.3f}/{s.placebo_p_one_sided_t_cl:.3f}; "
        f"nominal-5% rejection HAC(5) {s.placebo_rej5_hac5:.1%}, HC3 {s.placebo_rej5_hc3:.1%}, classical {s.placebo_rej5_cl:.1%}")

(OUT / "regression_sensitivity_log.txt").write_text("\n".join(log_lines) + "\n")
log(f"\nWrote outputs to {OUT.name if OUT.name == 'out' else 'Output/tables/regression_sensitivity'}")
