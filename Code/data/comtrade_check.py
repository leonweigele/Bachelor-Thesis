"""Check oil-basket signs using saved reporter totals and partner observations.

An empty response is unobserved, never a confirmed zero. Numeric partner rows
do not establish full partner coverage. Any mirror use keeps the verdict
qualified. Frozen reporter cells take precedence over supplementary mirrors.

Offline CLI exit codes: 0 = reporter coverage and matching signs,
2 = matching observed signs with coverage assumptions, 1 = fail or incomplete.
The live collector uses run_check(main_df, make_api_fetch(pull)).
"""
import argparse
import hashlib
import json
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
ROOT = Path(__file__).resolve().parents[2]
API_CSV = ROOT / "Data/manual/comtrade_crude_2709_api.csv"
MIRROR_CSV = ROOT / "Data/manual/comtrade_crude_2709_mirror_tur.csv"
SUPPLEMENT = ROOT / "Data/manual/comtrade_supplement_2026-09-15"


def _valid(series):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.ge(0))


def validate_rows(frame, flow=None, partner=None, year=None):
    """Reject mixed commodities, breakdowns, directions or query years."""
    fixed = {"cmdCode": 2709, "partner2Code": 0, "motCode": 0}
    if partner is not None:
        fixed["partnerCode"] = int(partner)
    if year is not None:
        fixed["refYear"] = int(year)
    required = set(fixed) | {"reporterCode", "flowCode", "customsCode", "primaryValue", "refYear"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    for column, value in fixed.items():
        if not pd.to_numeric(frame[column], errors="coerce").eq(value).all():
            raise ValueError(f"Unexpected {column}, expected {value}")
    if not frame.customsCode.eq("C00").all():
        raise ValueError("Unexpected customs breakdown")
    if flow is not None and not frame.flowCode.eq(flow).all():
        raise ValueError(f"Unexpected mirror direction, expected {flow}")
    codes = pd.to_numeric(frame.reporterCode, errors="coerce")
    if not (codes.gt(0) & codes.mod(1).eq(0)).all():
        raise ValueError("Invalid reporter code")


def reporter_cells(main_df):
    data = main_df.copy()
    data["refYear"] = pd.to_numeric(data["refYear"], errors="coerce")
    data = data[data.refYear.between(*WINDOW)]
    validate_rows(data, partner=0)
    if not data.flowCode.isin(["M", "X"]).all():
        raise ValueError("Unexpected reporter trade direction")
    data = data[data.reporterISO.isin(M49)]
    expected_codes = data.reporterISO.map(M49).astype(int)
    if not pd.to_numeric(data.reporterCode).eq(expected_codes).all():
        raise ValueError("Reporter ISO and M49 codes disagree")
    data["value"] = _valid(data.primaryValue)
    reported, duplicates, invalid = {}, [], set()
    for (iso, flow, year), rows in data.groupby(["reporterISO", "flowCode", "refYear"]):
        cell = (iso, flow, int(year))
        if len(rows) != 1:
            duplicates.append(cell)
        elif pd.notna(rows.value.iloc[0]):
            reported[cell] = float(rows.value.iloc[0])
        else:
            invalid.add(cell)
    return reported, duplicates, invalid


def observe(frame, flow, partner, year):
    """Return an observed sum, its row-validity state and partner reporter codes."""
    if frame is None or frame.empty:
        reason = "query unavailable" if frame is None else frame.attrs.get("query_status", "no returned records")
        return np.nan, "unresolved", "", reason
    validate_rows(frame, flow=flow, partner=partner, year=year)
    codes = pd.to_numeric(frame.reporterCode)
    if codes.duplicated().any():
        raise ValueError("Duplicate partner reporter rows")
    values = _valid(frame.primaryValue)
    partners = ",".join(str(int(c)) for c in sorted(codes))
    if not values.notna().any():
        return np.nan, "unresolved", partners, "no valid nonnegative values"
    state = "mirror_observed" if values.notna().all() else "mirror_partial"
    return float(values.sum()), state, partners, "partner coverage not established"


def assess(main_df, fetch):
    """Build one row per country, own trade direction and classification year."""
    reported, duplicates, invalid = reporter_cells(main_df)
    rows, errors = [], []
    for iso in ISO + ["TUR"]:
        for flow in ("M", "X"):
            for year in YEARS:
                cell = (iso, flow, year)
                value, partners, detail = np.nan, "", ""
                if cell in duplicates:
                    state, detail = "rejected", "duplicate reporter records"
                    errors.append(f"{cell}: {detail}")
                elif cell in reported:
                    value, state = reported[cell], "reporter"
                else:
                    reverse = "X" if flow == "M" else "M"
                    try:
                        value, state, partners, detail = observe(fetch(reverse, M49[iso], year), reverse, M49[iso], year)
                    except ValueError as exc:
                        state, detail = "rejected", str(exc)
                        errors.append(f"{cell}: {detail}")
                    if cell in invalid:
                        detail = "invalid reporter value. " + detail
                rows.append({"iso": iso, "flow": flow, "year": year, "observed_usd": value,
                             "source": state, "partner_reporter_codes": partners, "detail": detail})
    evidence = pd.DataFrame(rows)
    summary = []
    for iso in ISO + ["TUR"]:
        data = evidence[evidence.iso.eq(iso)]
        # Summing observed amounts does not assign zeros to the missing cells.
        exports = float(data.loc[data.flow.eq("X"), "observed_usd"].sum())
        imports = float(data.loc[data.flow.eq("M"), "observed_usd"].sum())
        net = exports - imports
        role = "exporter" if iso in EXPORTERS else "importer"
        observed_count = int(data.observed_usd.notna().sum())
        matches = net > 0 if role == "exporter" else net < 0
        summary.append({"iso": iso, "role": role, "observed_exports_usd": exports,
                        "observed_imports_usd": imports, "observed_net_exports_usd": net,
                        "observed_cells": observed_count,
                        "unresolved_cells": int(data.observed_usd.isna().sum()),
                        "mirror_cells": int(data.source.str.startswith("mirror_").sum()),
                        "sign_matches": bool(matches)})
    summary = pd.DataFrame(summary)
    if errors or summary.observed_cells.eq(0).any():
        status = "INCOMPLETE: malformed or unobserved country data"
    elif not summary.sign_matches.all():
        status = "FAIL: an observed trade balance does not support its assigned basket"
    elif not evidence.source.eq("reporter").all():
        status = "PASS WITH ASSUMPTIONS: all 13 observed signs match, coverage remains qualified"
    else:
        status = "PASS: all 13 signs match on available reporter totals for every cell"
    return status, evidence, summary, errors


def format_report(result):
    status, evidence, summary, errors = result
    base = evidence[evidence.iso.ne("TUR")]
    missing = base[base.source.ne("reporter")]
    supplemented = missing.source.str.startswith("mirror_").sum()
    lines = [f"2019-2024 HS 2709, current USD. Frozen reporter cells take priority.",
             f"12-country reporter extract: {len(base) - len(missing)}/144 valid cells. "
             f"Partner observations supply {supplemented}/{len(missing)} other cells."]
    for row in summary.itertuples():
        gaps = evidence[evidence.iso.eq(row.iso) & evidence.observed_usd.isna()]
        gap_text = ", ".join(f"{r.flow} {r.year}" for r in gaps.itertuples()) or "none"
        lines.append(f"{row.iso} {row.role}: observed X {row.observed_exports_usd / 1e9:.6f} bn, "
                     f"M {row.observed_imports_usd / 1e9:.6f} bn, "
                     f"X-M {row.observed_net_exports_usd / 1e9:+.6f} bn. "
                     f"Mirror cells {row.mirror_cells}, unobserved cells {gap_text}. "
                     f"Observed sign {'matches' if row.sign_matches else 'does not match'}.")
    for row in evidence[evidence.source.isin(["unresolved", "rejected", "mirror_partial"])].itertuples():
        lines.append(f"GAP {row.iso} {row.flow} {row.year}: {row.source}, {row.detail}.")
    tur_imports = evidence[evidence.iso.eq("TUR") & evidence.flow.eq("M") & evidence.source.str.startswith("mirror_")]
    for code, name in [(368, "Iraq"), (643, "Russia")]:
        absent = [r.year for r in tur_imports.itertuples() if str(code) not in r.partner_reporter_codes.split(",")]
        if absent:
            lines.append(f"TUR import mirror: no returned {name} records in {absent}. This does not establish zero trade.")
    if not evidence.source.eq("reporter").all():
        lines.extend([
            "Mirror imports = partners' exports to the target. Mirror exports = partners' imports from the target.",
            "Valid numeric rows do not establish complete partner coverage. Missing cells remain unobserved.",
            "Totals sum available values only. Observed net balances are not bounds on true net trade.",
            "Mirror and reporter values may differ in valuation, timing and attribution. Basket support is conditional on coverage.",
        ])
    lines.extend("INCOMPLETE: " + error for error in errors)
    lines.append("CLASSIFICATION CHECK: " + status)
    return lines


def run_check(main_df, fetch, out=print):
    """Stable interface used by comtrade_pull.py."""
    result = assess(main_df, fetch)
    lines = format_report(result)
    for line in lines:
        out(line)
    return result[0], lines


def make_api_fetch(pull, sleep=1.5, log=print):
    import time

    def fetch(flow, partner, year):
        try:
            frame = pull(period=str(year), reporterCode=None, flowCode=flow, partnerCode=partner,
                         partner2Code="0", customsCode="C00", motCode="0")
        except Exception as exc:
            log(f"Mirror query failed ({flow}, partner {partner}, {year}): {exc}")
            frame = None
        time.sleep(sleep)
        return frame
    return fetch


def load_supplement(folder):
    """Load original API responses, verifying hashes and query perspective."""
    folder = Path(folder)
    manifest = json.loads((folder / "request_manifest.json").read_text())
    queries = {}
    for meta in manifest:
        label = meta["label"]
        if Path(label).name != label:
            raise ValueError("Invalid raw response label")
        raw = (folder / "raw" / f"{label}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError(f"Raw response hash differs: {label}")
        payload = json.loads(raw)
        data = payload.get("data")
        if (meta["status"] != "ok" or meta["http_status"] != 200 or payload.get("error")
                or not isinstance(data, list) or len(data) != payload.get("count")
                or len(data) != meta["row_count"] or len(data) >= int(meta["parameters"]["maxRecords"])):
            raise ValueError(f"Failed, inconsistent or capped response: {label}")
        if meta["query_kind"] != "mirror":
            continue
        params = meta["parameters"]
        flow, partner, year = params["flowCode"], params["partnerCode"], int(params["period"])
        if (partner != M49[meta["target_iso"]] or year != meta["year"]
                or flow != ("X" if meta["target_flow"] == "M" else "M")):
            raise ValueError(f"Target and API perspectives disagree: {label}")
        key = (flow, partner, year)
        if key in queries:
            raise ValueError(f"Duplicate query: {key}")
        frame = pd.DataFrame(data)
        if not frame.empty:
            validate_rows(frame, flow=flow, partner=partner, year=year)
        frame.attrs["query_status"] = "successful API query returned no records"
        queries[key] = frame
    return queries


def make_offline_fetch(mirror_tur_df=None, supplement=None):
    """Saved supplement first, legacy Türkiye imports only if not queried there."""
    queries = {} if supplement is None else load_supplement(supplement)

    def fetch(flow, partner, year):
        key = (flow, str(partner), year)
        if key in queries:
            return queries[key].copy()
        if mirror_tur_df is not None and str(partner) == M49["TUR"] and flow == "X":
            data = mirror_tur_df
            rows = data[pd.to_numeric(data.partnerCode).eq(int(partner)) & data.flowCode.eq(flow)
                        & pd.to_numeric(data.refYear).eq(year)].copy()
            return rows if not rows.empty else None
        return None
    return fetch


def exit_code(status):
    if status.startswith("PASS WITH ASSUMPTIONS:"):
        return 2
    return 0 if status.startswith("PASS:") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--supplement-dir", type=Path, default=SUPPLEMENT)
    parser.add_argument("--report-dir", type=Path, help="Save evidence, country totals and the check log")
    args = parser.parse_args()
    try:
        fetch = make_offline_fetch(pd.read_csv(MIRROR_CSV), args.supplement_dir)
        result = assess(pd.read_csv(API_CSV), fetch)
        lines = format_report(result)
    except (OSError, ValueError, KeyError) as exc:
        print(f"INCOMPLETE: {exc}")
        return 1
    print("\n".join(lines))
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        result[1].to_csv(args.report_dir / "classification_evidence.csv", index=False)
        result[2].to_csv(args.report_dir / "classification_summary.csv", index=False)
        (args.report_dir / "classification_check.txt").write_text("\n".join(lines) + "\n")
    return exit_code(result[0])


if __name__ == "__main__":
    sys.exit(main())
