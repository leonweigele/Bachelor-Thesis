"""
consolidate_lseg.py — turn the individual LSEG "Price History" exports into the
clean panel files the pipeline expects.
=========================================================================
You exported one file per RIC into Data/manual/LSEG/ (EUR.xlsx, EUR1M.xlsx,
XAU=ZKBZ.xlsx, STOXX50E.xlsx, ...). LSEG's Price History layout has a metadata
header + a price-distribution block on top, then the daily series (newest first)
with columns like 'Exchange Date | Bid | Ask | ...' or 'Exchange Date | Close'.
This script parses that, takes the daily MID (Bid+Ask)/2 for FX/forwards/gold and
CLOSE for indices, and writes:

  Data/manual/lseg_fx_spot.csv          32 spot, RAW quotes (majors USD per FC)
  Data/manual/lseg_fx_fwd1m_points.csv  clean <CCY>1M forward POINTS
  Data/manual/lseg_gold.csv             column 'Gold'
  Data/manual/lseg_stoxx.csv            column 'EuroStoxx50_EUR'
  Data/manual/lseg_ttf.csv              column 'TTF'

Re-run this whenever you drop fresh exports into Data/manual/LSEG/.
RUN ORDER:  consolidate_lseg.py -> get_data.py -> build_fx_factors.py -> 02_build_returns.py
"""

import datetime
import glob
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data/manual/LSEG"
OUT = ROOT / "Data/manual"

# Forward variants whose scaling/definition is non-standard -> skipped.
# (COP has no 1M file at all; CLP/PEN only as =FMD products.)
SKIP_FWD_SUFFIX = ("=FMD", "OR=FMD")


def first_dt(rows):
    for i, r in enumerate(rows):
        if isinstance(r[0], (datetime.datetime, datetime.date)):
            return i
    return None


def parse(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    d = first_dt(rows)
    if d is None:
        return None
    hdr = [(str(c).strip() if c is not None else "") for c in rows[d - 1]]
    if "Close" in hdr:
        cols = [hdr.index("Close")]
    elif "Bid" in hdr and "Ask" in hdr:
        cols = [hdr.index("Bid"), hdr.index("Ask")]
    else:
        cols = [1]
    rec = {}
    for r in rows[d:]:
        if isinstance(r[0], (datetime.datetime, datetime.date)):
            vals = [r[c] for c in cols
                    if c < len(r) and isinstance(r[c], (int, float))]
            if vals:
                rec[pd.Timestamp(r[0]).normalize()] = sum(vals) / len(vals)
    return pd.Series(rec).sort_index() if rec else None


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found — put your LSEG exports there.")
    spot, pts = {}, {}
    gold = stoxx = ttf = None
    for f in sorted(glob.glob(str(SRC / "*.xlsx"))):
        b = os.path.basename(f)[:-5]
        s = parse(f)
        if s is None:
            print(f"  ! could not parse {b}")
            continue
        if b.startswith("XAU"):
            gold = s
        elif b in ("STOXX50E", ".STOXX50E"):
            stoxx = s
        elif b == "TRNLTTFMc1":
            ttf = s
        elif b.endswith("1M") and "=" not in b:        # clean <CCY>1M points
            pts[b[:-2]] = s
        elif "=" not in b and len(b) == 3:              # 3-letter spot RIC
            spot[b] = s
        else:
            print(f"  - skipped {b} (non-standard variant)")

    def wide(d):
        df = pd.DataFrame(d).sort_index()
        df.index.name = "date"
        return df

    OUT.mkdir(parents=True, exist_ok=True)
    wide(spot).to_csv(OUT / "lseg_fx_spot.csv")
    wide(pts).to_csv(OUT / "lseg_fx_fwd1m_points.csv")
    if gold is not None:
        pd.DataFrame({"Gold": gold}).rename_axis("date").to_csv(OUT / "lseg_gold.csv")
    if stoxx is not None:
        pd.DataFrame({"EuroStoxx50_EUR": stoxx}).rename_axis("date").to_csv(OUT / "lseg_stoxx.csv")
    if ttf is not None:
        pd.DataFrame({"TTF": ttf}).rename_axis("date").to_csv(OUT / "lseg_ttf.csv")

    print(f"Consolidated: spot {len(spot)} ccy, forwards(points) {len(pts)} ccy, "
          f"gold={'y' if gold is not None else 'n'}, "
          f"stoxx={'y' if stoxx is not None else 'n'}, ttf={'y' if ttf is not None else 'n'}")
    print("Wrote -> Data/manual/lseg_fx_spot.csv, lseg_fx_fwd1m_points.csv, "
          "lseg_gold.csv, lseg_stoxx.csv, lseg_ttf.csv")
    miss = sorted({"AUD","BRL","CAD","CHF","CLP","CNY","COP","CZK","DKK","EUR",
                   "GBP","HKD","HUF","IDR","ILS","INR","JPY","KRW","KWD","MXN",
                   "MYR","NOK","NZD","PEN","PLN","SAR","SEK","SGD","THB","TRY",
                   "TWD","ZAR"} - set(spot))
    if miss:
        print(f"  note: spot missing for {miss}")
    nofwd = sorted(set(spot) - set(pts) - {"DKK","HKD","SAR","KWD","CNY"})
    if nofwd:
        print(f"  note: no clean 1M forward for {nofwd} (dropped from carry sort)")


if __name__ == "__main__":
    main()
