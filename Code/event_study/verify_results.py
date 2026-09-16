"""
verify_results.py — compare current event-study outputs against a pinned baseline.
==================================================================================

Usage:
  python3 Code/event_study/verify_results.py                        # compare current vs baseline; exit 1 on any difference
  python3 Code/event_study/verify_results.py --pin                  # bless ALL current outputs as baseline (deliberate step)
  python3 Code/event_study/verify_results.py --pin FILE [FILE ...]  # bless only the named file(s), e.g. one pinned for the first time

A difference is ANY of: a column present in one file but not the other; a
duplicated or empty identifying key in either file; an added or removed row; a
numeric change larger than TOL; a value that became NaN or stopped being NaN; a
changed text field (e.g. the significance stars). A current or baseline file
that is missing or unreadable counts as a difference as well. Matching files
mean the OUTPUT FILES are unchanged since the pin — nothing more. Whether the
thesis text quotes them correctly is a separate check.

Pinning never overwrites a baseline silently. An existing baseline file is
first copied to <file>.bak_<YYYY-MM-DD>_prepin next to it.
"""
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ES = ROOT / "Data/processed/event_study"
BASE = ES / "baseline"
FILES = ["car_summary.csv", "cross_event_diff.csv", "car_persistence_w50.csv"]
KEYS = {
    "car_summary.csv": ["event", "series", "method", "window"],
    "cross_event_diff.csv": ["series", "pair", "window"],
    "car_persistence_w50.csv": ["series", "event"],
}
# car_persistence_w50.csv (written by 04_event_study_w50.py) is the wide table
# behind Table 6.1's (0,50) column: one row per (series, event), columns
# "(0,5)", "(0,20)", "(0,50)", cells like "-1.0**" (CAR in percent, two
# decimals, plus significance stars). The cells are text, so the file is read
# as text and compared cell by cell. TOL plays no role for it. Reading it as
# text also stops pandas from turning a column that happens to carry no stars
# into floats, which would make the comparison depend on number formatting.
READ_OPTS = {"car_persistence_w50.csv": {"dtype": str, "keep_default_na": False}}
TOL = 1e-10          # absolute tolerance on numeric values


def backup_path(path):
    """<file>.bak_<today>_prepin next to the file, never overwriting an earlier backup."""
    stem = f"{path.name}.bak_{date.today():%Y-%m-%d}_prepin"
    cand, n = path.with_name(stem), 1
    while cand.exists():
        n += 1
        cand = path.with_name(f"{stem}_{n}")
    return cand


def pin(names=None):
    """Copy the named current files (default: all of FILES) into BASE, backing up first."""
    names = list(names) if names else list(FILES)
    unknown = [n for n in names if n not in FILES]
    if unknown:
        sys.exit(f"Unknown file(s) {unknown}. Choose from {FILES}.")
    missing = [n for n in names if not (ES / n).is_file()]
    if missing:
        sys.exit(f"Cannot pin. Current file(s) missing in {ES}: {missing}.")
    BASE.mkdir(exist_ok=True)
    for f in names:
        dst = BASE / f
        if dst.is_file():
            bak = backup_path(dst)
            shutil.copy(dst, bak)
            print(f"{f}: previous baseline kept as {bak.name}")
        shutil.copy(ES / f, dst)
    print(f"Baseline pinned in {BASE} ({', '.join(names)}).")


def read(path, name):
    """Read one CSV. Returns (frame, None) or (None, error line)."""
    if not path.is_file():
        return None, f"{name}: MISSING, expected at {path}"
    try:
        return pd.read_csv(path, **READ_OPTS.get(name, {})), None
    except Exception as e:          # empty file, malformed CSV, ...
        return None, f"{name}: UNREADABLE, {path} ({type(e).__name__}: {e})"


def blank_keys(df, key):
    """Number of rows whose identifying key has an empty or missing part."""
    bad = pd.Series(False, index=df.index)
    for k in key:
        bad |= df[k].isna() | (df[k].astype(str).str.strip() == "")
    return int(bad.sum())


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
    # 2. identifying keys complete and unique in both files
    blank_old, blank_cur = blank_keys(old, key), blank_keys(cur, key)
    if blank_old or blank_cur:
        rep.append(f"{name}: rows with an empty key — baseline {blank_old}, current {blank_cur}")
        return False, rep
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
        old, err_old = read(BASE / f, f)
        cur, err_cur = read(ES / f, f)
        if err_old or err_cur:
            for e in (err_old, err_cur):
                if e:
                    print(e)
            if err_old and not err_cur:
                print(f"  -> no baseline for {f} yet. Pin this file alone with: "
                      f"python3 Code/event_study/verify_results.py --pin {f}")
            ok_all = False
            continue
        ok, rep = compare_frames(f, old, cur)
        print("\n".join(rep))
        ok_all &= ok
    print("\nRESULT:", f"IDENTICAL to baseline. All {len(FILES)} output files are unchanged since the pin."
          if ok_all else "DIFFERS from baseline — inspect the lines above; re-pin only after a deliberate change.")
    return ok_all


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--pin":
        pin(args[1:])
    elif args:
        sys.exit(f"Unknown argument(s) {args}. Usage: verify_results.py [--pin [FILE ...]]")
    else:
        sys.exit(0 if compare() else 1)
