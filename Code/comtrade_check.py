"""
comtrade_check.py — coverage-aware classification check for the oil baskets.
===========================================================================
Proposed new module under Code/. Pure functions plus two entry points:

  python3 Code/comtrade_check.py --offline     # saved extracts only, NO network, no downloads
  (comtrade_pull.py calls run_check(main_df, make_api_fetch(pull)) after its pulls)

Definitions
  cell           (reporterISO, flowCode, year); expected = 12 countries x {M, X} x 2019-2024
  valid value    finite number; NaN, text, +/-inf are INVALID and never count as zero
  REPORTED       exactly one reporter-side row for the cell with a valid value
                 (a reported 0 is a confirmed zero)
  UNREPORTED     no row, or the only row has no valid value
  DUPLICATE      more than one reporter-side row for the cell (partner = World should
                 give exactly one) -> the extract is malformed for that cell; it is
                 rejected, not summed, and the check cannot pass
  MIRROR         partner-side data for an unreported cell:
                   imports of C = partners' exports TO C   (flow X, partnerCode = C)
                   exports of C = partners' imports FROM C (flow M, partnerCode = C)
                 complete -> every partner row valid: observed
                 partial  -> some partner rows invalid: LOWER BOUND, flagged
                 none     -> no valid partner value: unresolved
  Türkiye        both flows from mirror data, each of the six years fetched and
                 flagged individually; totals are taken over the observed years and
                 the coverage state is carried into the verdict wording

Status (never a bare PASS with incomplete coverage)
  FAIL                   a sign is wrong on the observed data (incl. Türkiye observed
                         exports >= observed imports)
  INCOMPLETE             duplicate records, a basket country without any observed data,
                         or Türkiye imports unobserved -> classification unverified
  PASS WITH ASSUMPTIONS  every sign verified on observed data, but some cells or years
                         are unresolved or only partially observed (each listed with the
                         size the unobserved part would need to reach to flip the sign)
  PASS                   full coverage: every cell and every Türkiye year fully observed
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = (2019, 2024)
YEARS = list(range(WINDOW[0], WINDOW[1] + 1))
ISO = ["BRA", "CAN", "COL", "JPN", "KOR", "KWT", "MEX", "S19", "NOR", "SAU", "IND", "THA"]
M49 = {"BRA": "76", "CAN": "124", "COL": "170", "JPN": "392", "KOR": "410", "KWT": "414",
       "MEX": "484", "S19": "490", "NOR": "579", "SAU": "682", "IND": "699", "THA": "764",
       "TUR": "792"}
EXPORTERS = {"NOR", "CAN", "MEX", "COL", "BRA", "SAU", "KWT"}
IMPORTERS = {"JPN", "KOR", "IND", "THA", "S19", "TUR"}

ROOT = Path(__file__).resolve().parent.parent
API_CSV = ROOT / "Data/manual/comtrade_crude_2709_api.csv"
MIRROR_CSV = ROOT / "Data/manual/comtrade_crude_2709_mirror_tur.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _valid(series):
    """Numeric, finite values only; everything else becomes NaN."""
    v = pd.to_numeric(series, errors="coerce").astype(float)
    return v.where(np.isfinite(v))


def observe(df):
    """Read a partner-side frame. Returns (value, complete):
       value    = sum of the valid entries, None if there is none
       complete = True only if every row carries a valid value."""
    if df is None or len(df) == 0 or "primaryValue" not in df:
        return None, False
    v = _valid(df["primaryValue"])
    if v.notna().sum() == 0:
        return None, False
    return float(v.sum()), bool(v.notna().all())


def reporter_cells(main_df):
    """Classify every reporter-side cell.
       Returns reported {cell: value}, duplicates [cells], and the set of cells that
       have a row but no valid value (they are treated as unreported)."""
    d = main_df.copy()
    d["refYear"] = pd.to_numeric(d["refYear"], errors="coerce")
    d = d[d.refYear.between(*WINDOW)]
    d["val"] = _valid(d["primaryValue"])
    reported, duplicates, invalid = {}, [], set()
    for (iso, flow, year), rows in d.groupby(["reporterISO", "flowCode", "refYear"]):
        cell = (iso, flow, int(year))
        if len(rows) > 1:
            duplicates.append(cell)
        elif rows["val"].notna().iloc[0]:
            reported[cell] = float(rows["val"].iloc[0])
        else:
            invalid.add(cell)
    return reported, duplicates, invalid


def mirror_resolve(missing, fetch):
    """fetch(flow, partner_m49, year) -> DataFrame | None (partner-side rows).
       Returns complete {cell: value}, partial {cell: lower_bound}, unresolved [cells]."""
    complete, partial, unresolved = {}, {}, []
    for iso, flow, year in sorted(missing):
        mirror_flow = "X" if flow == "M" else "M"
        val, full = observe(fetch(mirror_flow, M49[iso], year))
        if val is None:
            unresolved.append((iso, flow, year))
        elif full:
            complete[(iso, flow, year)] = val
        else:
            partial[(iso, flow, year)] = val
    return complete, partial, unresolved


def mirror_years(fetch, flow, partner):
    """One partner-side query per year. Returns total over observed years and the
       per-year states: {year: 'complete' | 'partial' | 'missing'}."""
    total, states = 0.0, {}
    for y in YEARS:
        val, full = observe(fetch(flow, partner, y))
        if val is None:
            states[y] = "missing"
        else:
            total += val
            states[y] = "complete" if full else "partial"
    observed_any = any(s != "missing" for s in states.values())
    return (total if observed_any else None), states


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def run_check(main_df, fetch, out=print):
    lines = []

    def say(s):
        lines.append(s); out(s)

    reported, duplicates, invalid = reporter_cells(main_df)
    expected = {(i, f, y) for i in ISO for f in ("M", "X") for y in YEARS}
    missing = (expected - set(reported)) - set(duplicates)
    complete, partial, unresolved = mirror_resolve(missing, fetch)
    observed = dict(reported); observed.update(complete)      # fully observed values
    bounds = dict(partial)                                     # lower bounds, flagged

    say(f"Coverage: {len(reported)} reported, {len(complete)} mirror-complete, "
        f"{len(partial)} mirror-partial, {len(unresolved)} unresolved, "
        f"{len(duplicates)} duplicate of {len(expected)} cells"
        f"{'; ' + str(len(invalid)) + ' reporter rows without a valid value' if invalid else ''}.")

    fail, incomplete, assumptions = False, [], []
    if duplicates:
        incomplete.append("DUPLICATES")
        say(f"INCOMPLETE: duplicate reporter records for {sorted(duplicates)} — extract malformed, cells rejected")

    for iso in ISO:
        obs = {(f, y): v for (i, f, y), v in observed.items() if i == iso}
        low = {(f, y): v for (i, f, y), v in bounds.items() if i == iso}
        gaps = [(f, y) for (i, f, y) in unresolved if i == iso]
        if not obs and not low:
            incomplete.append(iso)
            say(f"INCOMPLETE {iso}: no observed data {WINDOW[0]}-{WINDOW[1]} — unverified")
            continue
        X = sum(v for (f, y), v in obs.items() if f == "X") + sum(v for (f, y), v in low.items() if f == "X")
        M = sum(v for (f, y), v in obs.items() if f == "M") + sum(v for (f, y), v in low.items() if f == "M")
        net = X - M
        role = "exporter" if iso in EXPORTERS else "importer"
        if not ((net > 0) if role == "exporter" else (net < 0)):
            fail = True
            say(f"FAIL {iso}: classified {role} but net on observed data = {net/1e9:+.1f} bn")
        risk_flow = "M" if role == "exporter" else "X"       # the direction that works against the sign
        risky_gaps = [y for f, y in gaps if f == risk_flow]
        risky_part = [y for (f, y) in low if f == risk_flow]
        safe_open = sorted([y for f, y in gaps if f != risk_flow] + [y for (f, y) in low if f != risk_flow])
        notes = []
        if risky_gaps:
            notes.append(f"unreported {risk_flow} {risky_gaps} treated as zero")
        if risky_part:
            notes.append(f"{risk_flow} {risky_part} only partially observed (lower bound used)")
        if risky_gaps or risky_part:
            notes.append(f"sign flips only if the unobserved part exceeds {abs(net)/1e9:.1f} bn in total")
        if safe_open:
            notes.append(f"{'X' if risk_flow == 'M' else 'M'} {safe_open} unobserved or partial (cannot flip the sign)")
        if notes:
            assumptions.append(iso)
            say(f"ASSUMPTION {iso}: " + "; ".join(notes))

    # Türkiye: imports and exports from partner-side data, year by year
    imp, imp_states = mirror_years(fetch, "X", M49["TUR"])
    exp, exp_states = mirror_years(fetch, "M", M49["TUR"])
    imp_full = all(s == "complete" for s in imp_states.values())
    exp_full = all(s == "complete" for s in exp_states.values())
    imp_gaps = [y for y, s in imp_states.items() if s != "complete"]
    exp_gaps = [y for y, s in exp_states.items() if s != "complete"]
    if imp is None or imp <= 0:
        incomplete.append("TUR")
        say("INCOMPLETE TUR: mirror imports unobserved or not positive — unverified")
    elif exp is None:
        assumptions.append("TUR")
        say(f"ASSUMPTION TUR: net importer on available data (mirror imports {imp/1e9:.1f} bn"
            f"{', years ' + str(imp_gaps) + ' missing or partial' if imp_gaps else ''}); "
            f"exports not observed, assumed below that — classification conditional on incomplete coverage")
    elif exp >= imp:
        fail = True
        say(f"FAIL TUR: observed mirror exports {exp/1e9:.1f} bn >= observed mirror imports {imp/1e9:.1f} bn")
    elif imp_full and exp_full:
        say(f"TUR: mirror imports {imp/1e9:.1f} bn > mirror exports {exp/1e9:.1f} bn, all six years observed — importer confirmed")
    else:
        assumptions.append("TUR")
        say(f"ASSUMPTION TUR: net importer on available data (mirror imports {imp/1e9:.1f} bn, exports {exp/1e9:.1f} bn; "
            f"import years missing or partial {imp_gaps}, export years {exp_gaps}) — "
            f"classification conditional on incomplete coverage")

    if fail:
        status = "FAIL — see FAIL lines; classification not confirmed"
    elif incomplete:
        status = f"INCOMPLETE — unverified for {sorted(set(incomplete))}"
    elif assumptions:
        status = (f"PASS WITH ASSUMPTIONS — every sign verified on observed data; "
                  f"assumptions for {sorted(set(assumptions))}")
    else:
        status = "PASS — full coverage, every sign verified"
    say("CLASSIFICATION CHECK: " + status)
    return status, lines


# ---------------------------------------------------------------------------
# fetch wrappers
# ---------------------------------------------------------------------------
def make_api_fetch(pull, sleep=1.5, log=print):
    """Wrap comtrade_pull.pull(). An API error and an empty answer both return None
    (-> unresolved / missing); the reason is logged so an outage is not mistaken for absence."""
    import time

    def fetch(flow, partner, year):
        try:
            df = pull(period=str(year), reporterCode=None, flowCode=flow, partnerCode=partner)
        except Exception as e:                       # network / rate limit / API error
            log(f"  mirror query failed ({flow}, partner {partner}, {year}): {e}")
            df = None
        time.sleep(sleep)
        return df if df is not None and len(df) else None
    return fetch


def make_offline_fetch(mirror_tur_df):
    """No network: only the saved Türkiye import mirror (partners' exports to Türkiye) exists."""
    m = mirror_tur_df.copy()
    m["period"] = pd.to_numeric(m["period"], errors="coerce")

    def fetch(flow, partner, year):
        if partner == M49["TUR"] and flow == "X":
            r = m[(m.partnerCode.astype(str) == M49["TUR"]) & (m.flowCode == "X") & (m.period == year)]
            return r if len(r) else None
        return None
    return fetch


if __name__ == "__main__":
    if "--offline" not in sys.argv:
        sys.exit("Use --offline here (reads the saved extracts, no network). "
                 "For a live run, comtrade_pull.py calls run_check() after its pulls.")
    main_df = pd.read_csv(API_CSV)                    # saved extracts, read before anything else
    mirror = pd.read_csv(MIRROR_CSV)
    status, _ = run_check(main_df, make_offline_fetch(mirror))
    sys.exit(0 if status.startswith("PASS —") else 2)
