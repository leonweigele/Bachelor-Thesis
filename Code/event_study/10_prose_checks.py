"""
10_prose_checks.py — the hand-typed numbers of Chapters 5 to 7, read from the thesis and
recomputed from the data.
=========================================================================================
WHY. Leon's rule (2026-09-16): every number in the thesis that is computed from the
data must come from a script in Code/. The event-study tables, the cross-event tests,
the regression and its sensitivity, and the twelve-day war benchmark have their scripts
(03, 04, 05, 07, 08, 09). The numbers below were computed in session scratch scripts and
typed into the prose. This script (v2, 2026-09-20) does two things:

  1. it recomputes them from the frozen inputs with the conventions of 03/05, and
  2. it READS THE LIVE LaTeX SOURCE, finds each sentence or table row that quotes them,
     extracts the printed values and compares them with the recomputed ones.

A passage that is missing, or that matches more than once, is a failure in its own
right, so a deleted or duplicated sentence can never pass silently. Commented-out
LaTeX (whole-line and trailing `%` comments, `\\%` is not a comment) is ignored, so an
old number kept in a comment is neither read nor counted. Nothing is hard-coded about
the thesis values: the templates below carry the thesis wording with placeholders
where the numbers stand, and the printed values come from the file at run time.

WHAT IT CHECKS (passage, file)
  Table 6.4, rows 1 to 4              content/tab_se_sensitivity.tex
     the headline contrast Liberation Day - Hormuz, CAR(0,+20): difference, standard
     error, t and the rejection level under four variance treatments (row 1 as in 05;
     row 2 times sqrt(1+L/T); row 3 with the sd of the abnormal returns of days 0..+20;
     row 4 both)
  Section 5.1.1 holiday footnote      content/05_data_methodology.tex
     t before and after removing the zero-return days from both estimation windows
     and re-estimating means and sds
  Section 6.3 Ukraine caveat          content/06_results.tex
     VIX on 9 and 23 February 2022 and on day +1, that 23 February is day -1, the
     dollar's CAR(-20,-1) and its t, CAR(-5,-1), the abnormal return of day 0
  Section 6.5 IEEPA disclosure        content/06_results.tex
     the ruling of 20 February 2026 as day -6, the end of the estimation window, the
     abnormal returns summed over days -6..-1 and their t, CAR(0,+1)
  Section 7.1 asymmetry sentence      content/07_discussion.tex
     Liberation Day CAR(0,+1) and the same six-day sum
  Section 7.5 strikes sentence        content/07_discussion.tex
     the 25 May 2026 strikes' estimation window: days on or after 2 March 2026 of 120
  Table 6.1 note, ceasefire sentence  content/tab_es_dollar_horizon.tex
     the 7 April 2026 ceasefire's estimation window: end date and days on or after
     2 March 2026

DIAGNOSTICS (recomputed and reported, NOT printed in the thesis, never compared): zero
counts in the estimation and event windows, the benchmark means and sds, the event-window
sds behind Table 6.4 rows 3 and 4, the raw return sum of the IEEPA days, trading days
left after 25 May and after 17 June 2026.

CONVENTIONS  Identical to 03/05: day 0 = first trading day on or after the event date
(searchsorted), estimation window [-140,-21] (120 days), abnormal return r - mu-hat,
sample standard deviation (ddof = 1), L = h+1, t = CAR / (sd sqrt(L)), cross-event
s.e. = sqrt(L (sA^2 + sB^2)), T = 120. A printed value counts as matching when the
recomputed value, rounded to the decimals the thesis prints, equals it.

INPUTS   Data/processed/returns_daily.csv, daily_panel.csv, events.csv,
         Data/processed/event_study/cross_event_diff.csv, and the four .tex files above
OUTPUT   Output/tables/prose_checks/prose_checks.csv         (one row per compared value)
         Output/tables/prose_checks/prose_checks_diagnostics.csv
         Output/tables/prose_checks/prose_checks.txt         (the report printed below)
EXIT     0 only when every passage is found exactly once and every printed value
         matches; 1 on any mismatch, missing or ambiguous passage, or unreadable file.

USAGE    python3 Code/event_study/10_prose_checks.py
TESTS    python -m unittest discover -s Code/tests -p 'test_prose_checks.py'
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from es_common import ES_OUT, EST_WIN_MAIN, EVT_WIN_MAIN, PROC, load_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "Main/LaTeX Thesis/content"
OUTDIR = ROOT / "Output/tables/prose_checks"
SERIES = "r_DTWEXBGS"
T_EST = EST_WIN_MAIN[1] - EST_WIN_MAIN[0] + 1          # 120
RULING = "2026-02-20"          # Learning Resources, Inc. v. Trump, decided 20 Feb 2026
HORMUZ_ONSET = "2026-03-02"    # Hormuz day 0 (28 Feb 2026 is a Saturday); asserted below
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June",
                                       "July", "August", "September", "October", "November",
                                       "December"], start=1)}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20}

# ---- the passages: thesis wording with placeholders ------------------------------------
# Each template is the LaTeX exactly as written in the file (whitespace collapsed), with
# {num:key}, {int:key}, {word:key} or {date:key} where a value stands. The key names the
# recomputed quantity it is compared with. Templates are matched against the ACTIVE text
# only and must match exactly once.
PASSAGES = [
    dict(id="T64r1", file="tab_se_sensitivity.tex", passage="Table 6.4 row 1 (estimation-window sd)",
         template=r"As reported: estimation-window $\hat\sigma$, $L\hat\sigma^2$ & ${num:t64_diff}$ & ${num:t64_se1}$ & ${num:t64_t1}$ & ${int:t64_lvl1}\%$ \\"),
    dict(id="T64r2", file="tab_se_sensitivity.tex", passage="Table 6.4 row 2 (plus mu-hat estimation error)",
         template=r"$+$ estimation error in $\hat\mu$: $L\hat\sigma^2(1+L/T)$ & ${num:t64_diff}$ & ${num:t64_se2}$ & ${num:t64_t2}$ & ${int:t64_lvl2}\%$ \\"),
    dict(id="T64r3", file="tab_se_sensitivity.tex", passage="Table 6.4 row 3 (event-window sd)",
         template=r"Event-window $\hat\sigma$ instead & ${num:t64_diff}$ & ${num:t64_se3}$ & ${num:t64_t3}$ & ${int:t64_lvl3}\%$ \\"),
    dict(id="T64r4", file="tab_se_sensitivity.tex", passage="Table 6.4 row 4 (both corrections)",
         template=r"Both corrections & ${num:t64_diff}$ & ${num:t64_se4}$ & ${num:t64_t4}$ & ${int:t64_lvl4}\%$ \\"),
    dict(id="S511", file="05_data_methodology.tex", passage="Section 5.1.1 holiday footnote",
         template=r"changes the broad-dollar Liberation Day--Hormuz $\CAR(0,+20)$ comparison from $t={num:t_original}$ to ${num:t_excl_zero_returns}$, leaving significance at the 1~percent level unchanged."),
    dict(id="S63", file="06_results.tex", passage="Section 6.3 Ukraine anticipation caveat",
         template=r"the VIX rose from ${num:vix_9feb}$ on 9~February 2022 to ${num:vix_daym1}$ on {date:ukr_daym1}, the last trading day before the invasion, then eased to ${num:vix_dayp1}$ on day~$+1$. The dollar shows no such run-up, with a cumulative abnormal return of ${num:ukr_car_m20_m1}\%$ ($t={num:ukr_t_m20_m1}$) over the twenty trading days before the invasion and ${num:ukr_car_m5_m1}\%$ over the last five, and its appreciation begins on day~0 with an abnormal return of ${num:ukr_ar0}\%$ on the invasion day itself."),
    dict(id="S65", file="06_results.tex", passage="Section 6.5 IEEPA disclosure",
         template=r"falls on day~${int:ruling_rel_day}$, after the estimation window closes on {date:hormuz_est_end} and before the first headline window opens on day~$-1$, so it enters neither the benchmark nor any tabulated CAR. The dollar's abnormal returns sum to ${num:hormuz_car_m6_m1}\%$ over the six trading days from the ruling to day~$-1$ ($t={num:hormuz_t_m6_m1}$), against ${num:hormuz_car_0_1}\%$ over the two-day announcement window."),
    dict(id="S71", file="07_discussion.tex", passage="Section 7.1 asymmetry sentence",
         template=r"Imposing the tariffs took the broad dollar index down ${num:ld_car_0_1_abs}\%$ over the two-day announcement window, while the Supreme Court ruling of 20~February 2026 that removed their legal basis moved it by only ${num:hormuz_car_m6_m1}\%$ over six trading days"),
    dict(id="S75", file="07_discussion.tex", passage="Section 7.5 strikes sentence",
         template=r"The estimation window of the first contains {int:strikes_days_on_or_after_onset} of its {int:strikes_est_days} trading days on or after the 2~March onset of the Hormuz crisis"),
    dict(id="T61", file="tab_es_dollar_horizon.tex", passage="Table 6.1 note, ceasefire sentence",
         template=r"The ceasefire's estimation window ends on {date:ceasefire_est_end} and contains the first {word:ceasefire_days_on_or_after_onset} trading days of the crisis"),
]

_PH = re.compile(r"\{(num|int|word|date):([a-z0-9_]+)\}")
_KIND = {"num": r"[+-]?\d+(?:\.\d+)?", "int": r"[+-]?\d+",
         "word": r"[a-z]+|\d+", "date": r"\d{1,2}~[A-Z][a-z]+(?: \d{4})?"}


def compile_template(template):
    """Escape the literal parts, turn placeholders into named groups, collapse spaces."""
    parts, kinds, last = [], {}, 0
    for m in _PH.finditer(template):
        parts.append(re.escape(template[last:m.start()]))
        kind, key = m.groups()
        kinds[key] = kind
        parts.append(f"(?P<{key}>{_KIND[kind]})")
        last = m.end()
    parts.append(re.escape(template[last:]))
    pattern = "".join(parts).replace("\\ ", " ")
    pattern = re.sub(r"(?: |\\\n|\\\t)+", r"\\s+", pattern)   # any whitespace run
    return re.compile(pattern), kinds


def active_text(path):
    """The LaTeX with comments removed: whole-line and trailing comments go, an escaped
    `\\%` stays. Whitespace runs collapse to one space so templates can be single-line."""
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        cut = None
        for i, ch in enumerate(line):
            if ch == "%":
                back = 0
                j = i - 1
                while j >= 0 and line[j] == "\\":
                    back += 1
                    j -= 1
                if back % 2 == 0:
                    cut = i
                    break
        out.append(line if cut is None else line[:cut])
    return re.sub(r"\s+", " ", "\n".join(out))


def parse_printed(kind, text):
    if kind == "num":
        return float(text), len(text.split(".")[1]) if "." in text else 0
    if kind == "int":
        return int(text), 0
    if kind == "word":
        return (int(text) if text.isdigit() else WORDS[text]), 0
    d, mon, *yr = text.replace("~", " ").split(" ")
    return (int(d), MONTHS[mon], int(yr[0]) if yr else None), None


def printed_equal(kind, printed, decimals, computed):
    if kind == "date":
        d, mon, yr = printed
        return computed.day == d and computed.month == mon and (yr is None or computed.year == yr)
    if kind == "num":
        return float(f"{computed:.{decimals}f}") == printed
    return int(computed) == printed


def fmt(kind, value):
    if kind == "date":
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value)


# ---- the recomputation ------------------------------------------------------------------
def positions(idx, date):
    pos = idx.searchsorted(pd.Timestamp(date))
    est = idx[max(0, pos + EST_WIN_MAIN[0]): pos + EST_WIN_MAIN[1] + 1]
    evt = idx[pos + EVT_WIN_MAIN[0]: pos + EVT_WIN_MAIN[1] + 1]
    return pos, est, evt


def bench(r, est_idx, drop_zeros=False):
    est = r.loc[est_idx].dropna()
    n_zero = int((est == 0).sum())
    if drop_zeros:
        est = est[est != 0]
    return {"mu": float(est.mean()), "sd": float(est.std()), "n": int(len(est)),
            "n_zero": n_zero, "start": est_idx[0].date(), "end": est_idx[-1].date()}


def car(r, pos, mu, a, b):
    """CAR(a,b) = sum of (r_t - mu) over relative days a..b, both inclusive."""
    return float((r.iloc[pos + a: pos + b + 1] - mu).sum())


def level(t):
    a = abs(t)
    return 1 if a > 2.58 else 5 if a > 1.96 else 10 if a > 1.65 else 0


def recompute():
    """Every quantity a template refers to (keys) plus the diagnostics."""
    rets = pd.read_csv(PROC / "returns_daily.csv", parse_dates=["date"], index_col="date")
    panel = pd.read_csv(PROC / "daily_panel.csv", parse_dates=["date"], index_col="date")
    _, dates = load_events()
    for ev in ("liberation_day", "hormuz", "ukraine", "us_strikes", "hormuz_ceasefire"):
        assert ev in dates, (f"{ev} is missing from events.csv: a checked passage quotes its "
                             f"windows (see the docstring); update the passage list if the "
                             f"event was removed on purpose")
    r, idx = rets[SERIES], rets.index
    v, diag = {}, []

    # Table 6.4 and the holiday footnote (Liberation Day - Hormuz, CAR(0,+20))
    L = 21
    pos_l, est_l, _ = positions(idx, dates["liberation_day"])
    pos_h, est_h, _ = positions(idx, dates["hormuz"])
    assert idx[pos_h].date().isoformat() == HORMUZ_ONSET, idx[pos_h]
    bl, bh = bench(r, est_l), bench(r, est_h)
    assert bl["n"] == T_EST and bh["n"] == T_EST, (bl["n"], bh["n"])
    cl, ch = car(r, pos_l, bl["mu"], 0, 20), car(r, pos_h, bh["mu"], 0, 20)
    diff = cl - ch
    sd_evt_l = float((r.iloc[pos_l: pos_l + L] - bl["mu"]).std())
    sd_evt_h = float((r.iloc[pos_h: pos_h + L] - bh["mu"]).std())
    corr = np.sqrt(1 + L / T_EST)
    se = {1: np.sqrt(L * (bl["sd"] ** 2 + bh["sd"] ** 2))}
    se[2] = se[1] * corr
    se[3] = np.sqrt(L * (sd_evt_l ** 2 + sd_evt_h ** 2))
    se[4] = se[3] * corr
    v["t64_diff"] = diff * 100
    for k in se:
        v[f"t64_se{k}"] = se[k] * 100
        v[f"t64_t{k}"] = diff / se[k]
        v[f"t64_lvl{k}"] = level(diff / se[k])
    ced = pd.read_csv(ES_OUT / "cross_event_diff.csv")
    ref = ced[(ced.series == SERIES) & (ced.pair == "liberation_day-hormuz")
              & (ced.window == "(0,20)")].iloc[0]
    assert round(diff * 100, 3) == ref["diff_%"] and round(diff / se[1], 2) == ref["t"], \
        ("Table 6.4 row 1 must equal the pinned cross_event_diff.csv cell", diff * 100, ref["diff_%"])
    v["t_original"] = diff / se[1]
    bl0, bh0 = bench(r, est_l, drop_zeros=True), bench(r, est_h, drop_zeros=True)
    cl0, ch0 = car(r, pos_l, bl0["mu"], 0, 20), car(r, pos_h, bh0["mu"], 0, 20)
    v["t_excl_zero_returns"] = (cl0 - ch0) / np.sqrt(L * (bl0["sd"] ** 2 + bh0["sd"] ** 2))
    diag += [("Table 6.4", "event-window sd of abnormal returns, days 0..+20, Liberation Day, % per day", sd_evt_l * 100),
             ("Table 6.4", "event-window sd of abnormal returns, days 0..+20, Hormuz, % per day", sd_evt_h * 100),
             ("Table 6.4", "estimation-window sd, Liberation Day, % per day", bl["sd"] * 100),
             ("Table 6.4", "estimation-window sd, Hormuz, % per day", bh["sd"] * 100),
             ("Sec. 5.1.1", "zero returns in the Liberation Day estimation window (of 120)", bl0["n_zero"]),
             ("Sec. 5.1.1", "zero returns in the Hormuz estimation window (of 120)", bh0["n_zero"]),
             ("Sec. 5.1.1", "zero returns in the two (0,+20) event windows", int((r.iloc[pos_l: pos_l + L] == 0).sum() + (r.iloc[pos_h: pos_h + L] == 0).sum()))]

    # Section 6.3, Ukraine
    pos_u, est_u, _ = positions(idx, dates["ukraine"])
    bu = bench(r, est_u)
    assert bu["n"] == T_EST
    vix = panel["VIXCLS"]
    v["vix_9feb"] = float(vix.loc["2022-02-09"])
    v["ukr_daym1"] = idx[pos_u - 1].date()
    v["vix_daym1"] = float(vix.loc[idx[pos_u - 1]])
    v["vix_dayp1"] = float(vix.loc[idx[pos_u + 1]])
    c20 = car(r, pos_u, bu["mu"], -20, -1)
    v["ukr_car_m20_m1"] = c20 * 100
    v["ukr_t_m20_m1"] = c20 / (bu["sd"] * np.sqrt(20))
    v["ukr_car_m5_m1"] = car(r, pos_u, bu["mu"], -5, -1) * 100
    v["ukr_ar0"] = (float(r.iloc[pos_u]) - bu["mu"]) * 100
    diag += [("Sec. 6.3", "Ukraine day 0", idx[pos_u].date().isoformat()),
             ("Sec. 6.3", "Ukraine benchmark window", f"{bu['start']} to {bu['end']}"),
             ("Sec. 6.3", "Ukraine benchmark mean, % per day", bu["mu"] * 100),
             ("Sec. 6.3", "Ukraine benchmark sd, % per day", bu["sd"] * 100)]

    # Section 6.5 and 7.1, IEEPA ruling inside the Hormuz window
    rel = int(idx.searchsorted(pd.Timestamp(RULING)) - pos_h)
    c6 = car(r, pos_h, bh["mu"], -6, -1)
    v["ruling_rel_day"] = rel
    v["hormuz_est_end"] = bh["end"]
    v["hormuz_car_m6_m1"] = c6 * 100
    v["hormuz_t_m6_m1"] = c6 / (bh["sd"] * np.sqrt(6))
    v["hormuz_car_0_1"] = car(r, pos_h, bh["mu"], 0, 1) * 100
    v["ld_car_0_1_abs"] = abs(car(r, pos_l, bl["mu"], 0, 1) * 100)
    diag += [("Sec. 6.5", f"ruling {RULING} = trading day of the Hormuz event", rel),
             ("Sec. 6.5", "raw return sum of days -6..-1, %", float(r.iloc[pos_h - 6: pos_h].sum()) * 100),
             ("Sec. 7.1", "Liberation Day CAR(0,+1), signed, %", car(r, pos_l, bl["mu"], 0, 1) * 100)]

    # Section 7.5 and the Table 6.1 note
    onset = pd.Timestamp(HORMUZ_ONSET)
    pos_s, est_s, _ = positions(idx, dates["us_strikes"])
    v["strikes_est_days"] = len(est_s)
    v["strikes_days_on_or_after_onset"] = int((est_s >= onset).sum())
    pos_c, est_c, _ = positions(idx, dates["hormuz_ceasefire"])
    v["ceasefire_est_end"] = est_c[-1].date()
    v["ceasefire_days_on_or_after_onset"] = int((est_c >= onset).sum())
    diag += [("Sec. 7.5", "strikes estimation window", f"{est_s[0].date()} to {est_s[-1].date()}"),
             ("Sec. 7.5", "trading days after 25 May 2026", len(idx) - pos_s - 1),
             ("Sec. 7.5", "trading days after 17 June 2026", len(idx) - idx.searchsorted(pd.Timestamp("2026-06-17")) - 1),
             ("Sec. 7.5", "sample end", idx[-1].date().isoformat())]
    return v, diag


# ---- read the thesis and compare ---------------------------------------------------------
def run_checks(values):
    rows, problems, texts = [], [], {}
    for p in PASSAGES:
        path = CONTENT / p["file"]
        if p["file"] not in texts:
            try:
                texts[p["file"]] = active_text(path)
            except OSError as e:
                texts[p["file"]] = None
                problems.append(f"{'UNREADABLE':<12s} {p['passage']}: {path} ({e})")
        text = texts[p["file"]]
        if text is None:
            continue
        regex, kinds = compile_template(p["template"])
        hits = list(regex.finditer(text))
        if len(hits) != 1:
            state = "MISSING" if not hits else f"AMBIGUOUS ({len(hits)} matches)"
            problems.append(f"{state:<12s} {p['passage']} in {p['file']}")
            continue
        for key, text_value in hits[0].groupdict().items():
            kind = kinds[key]
            printed, dec = parse_printed(kind, text_value)
            computed = values[key]
            ok = printed_equal(kind, printed, dec, computed)
            rows.append({"passage": p["passage"], "file": p["file"], "quantity": key,
                         "thesis": text_value, "computed": fmt(kind, computed), "match": ok})
            if not ok:
                problems.append(f"{'MISMATCH':<12s} {p['passage']}: {key} printed {text_value}, "
                                f"recomputed {fmt(kind, computed)}")
    return rows, problems


def main():
    values, diag = recompute()
    rows, problems = run_checks(values)
    res = pd.DataFrame(rows, columns=["passage", "file", "quantity", "thesis", "computed", "match"])
    dg = pd.DataFrame(diag, columns=["section", "quantity", "computed"])
    dg["computed"] = [f"{x:.6f}" if isinstance(x, (float, np.floating)) else x for x in dg["computed"]]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTDIR / "prose_checks.csv", index=False)
    dg.to_csv(OUTDIR / "prose_checks_diagnostics.csv", index=False)

    found = sorted({r["passage"] for r in rows})
    lines = [f"Prose checks: {len(rows)} thesis values in {len(found)} passages read from the active "
             f"LaTeX and compared with the recomputed results", ""]
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 60):
        lines += [res.to_string(index=False) if len(res) else "(no values compared)", "",
                  "Diagnostics (recomputed, not printed in the thesis, not compared):",
                  dg.to_string(index=False), ""]
    if problems:
        lines += ["PROBLEMS:"] + [f"  {p}" for p in problems] + ["",
                  f"RESULT: FAIL. {len(problems)} problem(s): a printed value differs from the recomputed "
                  f"result, or a passage is missing or ambiguous. Fix the thesis text, or the template "
                  f"if the wording changed on purpose."]
    else:
        lines += [f"RESULT: All selected thesis values match the recomputed results. {len(rows)} values in "
                  f"{len(found)} passages: " + "; ".join(found) + "."]
    (OUTDIR / "prose_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {OUTDIR.relative_to(ROOT)}/prose_checks.csv, prose_checks_diagnostics.csv, prose_checks.txt")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
