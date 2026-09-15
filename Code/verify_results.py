"""
verify_results.py — compare current event-study outputs against a pinned baseline.
==================================================================================
Proposed complete replacement for Code/verify_results.py.

Usage:
  python3 Code/verify_results.py            # compare current vs baseline; exit 1 on any difference
  python3 Code/verify_results.py --pin      # bless the CURRENT outputs as baseline (deliberate step)

A difference is ANY of: a column present in one file but not the other; a
duplicated identifying key in either file; an added or removed row; a numeric
change larger than TOL; a value that became NaN or stopped being NaN; a changed
text field (e.g. the significance stars). Matching files mean the OUTPUT FILES
are unchanged since the pin — nothing more. Whether the thesis text quotes them
correctly is a separate check.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ES = ROOT / "Data/processed/event_study"
BASE = ES / "baseline"
FILES = ["car_summary.csv", "cross_event_diff.csv"]
KEYS = {
    "car_summary.csv": ["event", "series", "method", "window"],
    "cross_event_diff.csv": ["series", "pair", "window"],
}
TOL = 1e-10          # absolute tolerance on numeric values


def pin():
    BASE.mkdir(exist_ok=True)
    for f in FILES:
        shutil.copy(ES / f, BASE / f)
    print(f"Baseline pinned in {BASE} ({', '.join(FILES)}).")


def compare_frames(name, old, cur):
    """Compare one file. Returns (ok, report_lines)."""
    key = KEYS[name]
    rep = []
    # 1. same columns in both files
    if set(old.columns) != set(cur.columns):
        rep.append(f"{name}: column sets differ — only in baseline {sorted(set(old.columns) - set(cur.columns))}, "
                   f"only in current {sorted(set(cur.columns) - set(old.columns))}")
        return False, rep
    missing_key = [k for k in key if k not in cur.columns]
    if missing_key:
        rep.append(f"{name}: key column(s) {missing_key} missing")
        return False, rep
    # 2. identifying keys unique in both files
    dup_old, dup_cur = int(old.duplicated(key).sum()), int(cur.duplicated(key).sum())
    if dup_old or dup_cur:
        rep.append(f"{name}: duplicated keys — baseline {dup_old}, current {dup_cur}")
        return False, rep
    # 3. row-by-row comparison of every non-key column
    cols = [c for c in cur.columns if c not in key]
    m = old.merge(cur, on=key, suffixes=("_base", "_now"), how="outer", indicator=True)
    added = int((m["_merge"] == "right_only").sum())
    gone = int((m["_merge"] == "left_only").sum())
    both = m[m["_merge"] == "both"]
    changed = pd.Series(False, index=both.index)
    for c in cols:
        a, b = both[f"{c}_base"], both[f"{c}_now"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            changed |= ((a - b).abs() > TOL).fillna(False) | (a.isna() != b.isna())
        else:
            changed |= (a.fillna("").astype(str) != b.fillna("").astype(str))
    n = int(changed.sum())
    rep.append(f"{name}: {len(both)} shared rows | {n} changed | {added} new | {gone} removed")
    if n:
        show = key + [f"{c}_{s}" for c in cols for s in ("base", "now")]
        rep.append(both.loc[changed, show].head(10).to_string(index=False))
    if added:
        rep.append("  new rows (first 5):\n" + m.loc[m["_merge"] == "right_only", key].head(5).to_string(index=False))
    if gone:
        rep.append("  removed rows (first 5):\n" + m.loc[m["_merge"] == "left_only", key].head(5).to_string(index=False))
    return (n == 0 and added == 0 and gone == 0), rep


def compare():
    if not BASE.exists():
        sys.exit("No baseline yet — run with --pin first.")
    ok_all = True
    for f in FILES:
        ok, rep = compare_frames(f, pd.read_csv(BASE / f), pd.read_csv(ES / f))
        print("\n".join(rep))
        ok_all &= ok
    print("\nRESULT:", "IDENTICAL to baseline — output files unchanged since the pin."
          if ok_all else "DIFFERS from baseline — inspect the lines above; re-pin only after a deliberate change.")
    return ok_all


if __name__ == "__main__":
    if "--pin" in sys.argv:
        pin()
    else:
        sys.exit(0 if compare() else 1)
