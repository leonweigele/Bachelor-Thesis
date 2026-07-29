"""
verify_results.py — compare current event-study outputs against a pinned baseline.
==================================================================================
Fixes the "275 cells changed" false alarm: comparing against git HEAD mixes in
months of unrelated changes. This compares against an explicit snapshot instead.

Usage:
  python3 Code/verify_results.py            # compare current vs baseline
  python3 Code/verify_results.py --pin      # bless the CURRENT outputs as baseline

Pin a new baseline only after you have deliberately changed data or method and
checked the headline numbers once by hand.
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
TOL = 1e-10          # absolute tolerance on CAR/t values


def pin():
    BASE.mkdir(exist_ok=True)
    for f in FILES:
        shutil.copy(ES / f, BASE / f)
    print(f"Baseline pinned in {BASE} ({', '.join(FILES)}).")


def compare():
    if not BASE.exists():
        sys.exit("No baseline yet — run with --pin first.")
    ok = True
    for f in FILES:
        cur, old = pd.read_csv(ES / f), pd.read_csv(BASE / f)
        key = KEYS[f]
        vals = [c for c in cur.columns
                if c not in key and pd.api.types.is_numeric_dtype(cur[c])]
        m = old.merge(cur, on=key, suffixes=("_base", "_now"), how="outer",
                      indicator=True)
        added = (m["_merge"] == "right_only").sum()
        gone = (m["_merge"] == "left_only").sum()
        both = m[m["_merge"] == "both"]
        changed = pd.Series(False, index=both.index)
        for v in vals:
            a, b = both[f"{v}_base"], both[f"{v}_now"]
            changed |= ((a - b).abs() > TOL) & ~(a.isna() & b.isna())
        n = int(changed.sum())
        print(f"{f}: {len(both)} shared rows | {n} changed | "
              f"{added} new | {gone} removed")
        if n:
            ok = False
            cols = key + [c for v in vals for c in (f"{v}_base", f"{v}_now")]
            print(both.loc[changed, cols].head(10).to_string(index=False))
    print("\nRESULT:", "IDENTICAL to baseline — tables and prose still valid."
          if ok else "DIFFERS from baseline — re-check Tables 6.1/6.2 and "
                     "every number quoted in the prose before trusting them.")


if __name__ == "__main__":
    pin() if "--pin" in sys.argv else compare()
