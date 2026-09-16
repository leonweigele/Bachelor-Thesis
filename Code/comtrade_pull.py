"""Refresh the original Comtrade extracts and check oil-basket signs.

This overwrites the saved inputs. For reproduction without new API calls, use
Code/comtrade_check.py --offline instead.

Pull annual HS 2709 reporter totals for 12 countries, 2019-2025, partner World,
plus partners' exports to Türkiye. The coverage-aware checker then queries
missing reporter cells and both mirror directions for Türkiye over 2019-2024.
The original reporter query follows the manual portal download of 14 August
2026. The separately saved September supplement is documented in the README.

Numeric mirror responses do not establish complete partner coverage. The check
keeps missing observations explicit and returns exit 2 for qualified support,
exit 1 for a failed or incomplete check, and exit 0 for matching signs with
reporter totals in every classification cell.

USAGE
  python3 Code/comtrade_pull.py                     # keyless public preview API
  COMTRADE_API_KEY=xxx python3 Code/comtrade_pull.py # optional subscription key

OUTPUTS
  Data/manual/comtrade_crude_2709_api.csv         12 reporter countries
  Data/manual/comtrade_crude_2709_mirror_tur.csv  partners' exports to Türkiye
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import comtradeapicall as ct

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Data/manual"
KEY = os.environ.get("COMTRADE_API_KEY")

CMD = "2709"                       # HS heading: petroleum oils, crude
YEARS = "2019,2020,2021,2022,2023,2024,2025"
WINDOW = (2019, 2024)              # classification window, subject to coverage gaps

# UN M49 reporter codes, identical to the portal download
REPORTERS = {
    "76": "Brazil", "124": "Canada", "170": "Colombia", "392": "Japan",
    "410": "Rep. of Korea", "414": "Kuwait", "484": "Mexico",
    "490": "Other Asia, nes (= Taiwan)", "579": "Norway",
    "682": "Saudi Arabia", "699": "India", "764": "Thailand",
}
TUR = "792"                        # Türkiye (partner side only, mirror)

EXPORTERS = {"NOR", "CAN", "MEX", "COL", "BRA", "SAU", "KWT"}
IMPORTERS = {"JPN", "KOR", "IND", "THA", "S19"}   # S19 = Other Asia, nes


def pull(**kw):
    """One API call; keyless preview by default, subscription key if set."""
    base = dict(typeCode="C", freqCode="A", clCode="HS", cmdCode=CMD,
                partner2Code="0", customsCode="C00", motCode="0",
                format_output="JSON", aggregateBy=None,
                breakdownMode="classic", countOnly=None, includeDesc=True)
    base.update(kw)
    if KEY:
        return ct.getFinalData(KEY, maxRecords=250000, **base)
    return ct.previewFinalData(maxRecords=500, **base)


def require(df, label):
    if df is None or len(df) == 0:
        sys.exit(f"ERROR: no data returned for {label} — API down or "
                 "rate-limited. Retry, or set COMTRADE_API_KEY.")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- main pull: 12 reporters x both flows x 2019-2025, partner World ----
    # The keyless preview endpoint accepts only ONE period per call, so loop
    # over the years when no key is set; with a key one combined call works.
    print("Pulling main query (12 reporters, HS 2709, 2019-2025) ...")
    if KEY:
        main_df = pull(period=YEARS, reporterCode=",".join(REPORTERS),
                       flowCode="M,X", partnerCode="0")
    else:
        frames = []
        for y in YEARS.split(","):
            df = pull(period=y, reporterCode=",".join(REPORTERS),
                      flowCode="M,X", partnerCode="0")
            if df is not None and len(df):
                frames.append(df)
                print(f"  {y}: {len(df)} rows")
            else:
                print(f"  {y}: no rows returned")
            time.sleep(1.5)
        main_df = pd.concat(frames, ignore_index=True) if frames else None
    main_df = require(main_df, "main query")
    main_df.to_csv(OUT / "comtrade_crude_2709_api.csv", index=False)
    print(f"  {len(main_df)} rows -> Data/manual/comtrade_crude_2709_api.csv")

    # ---- mirror pull: world exports TO Türkiye, year by year ----------------
    print("Pulling Türkiye mirror (world exports to Türkiye) ...")
    frames = []
    for y in YEARS.split(","):
        df = pull(period=y, reporterCode=None, flowCode="X", partnerCode=TUR)
        if df is not None and len(df):
            frames.append(df)
        time.sleep(1.5)            # be polite to the keyless endpoint
    mirror = require(pd.concat(frames, ignore_index=True) if frames else None,
                     "Türkiye mirror")
    mirror.to_csv(OUT / "comtrade_crude_2709_mirror_tur.csv", index=False)
    print(f"  {len(mirror)} rows -> Data/manual/comtrade_crude_2709_mirror_tur.csv")

    # ---- coverage-aware classification check (comtrade_check.py, 2026-09-15) --
    from comtrade_check import run_check, make_api_fetch, exit_code
    status, report = run_check(main_df, make_api_fetch(pull))
    return exit_code(status)


if __name__ == "__main__":
    sys.exit(main())
