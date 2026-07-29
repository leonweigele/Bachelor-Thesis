# LSEG data pull — exact steps (clean re-download to 2026-06-30)

Written 2026-07-15. Verified against the actual code: `Code/get_data.py`,
`Code/build_fx_factors.py`, `Code/02_build_returns.py`, `Code/consolidate_lseg.py`.

Universal settings for **every** request:
- Frequency: **Daily**
- Range: **2019-01-01 → 2026-06-30** (the `FREEZE` date in `get_data.py`)
- Price basis: **MID** for FX / forwards / gold (Exchange Rate Middle, "ER"),
  **CLOSE** for the equity/gas indices
- Every exported file needs a **`date`** column (a column literally named
  `date` or `day` — the loader looks for that word).

There are two ways to produce the files. Pick per block:
- **Route A (batch, fewest clicks):** pull a whole list into one wide sheet,
  save straight to the target CSV. Best for the 32-currency FX blocks.
- **Route B (one file per RIC → auto-clean):** export each RIC via right-click
  "Price History" into `Data/manual/LSEG/`, then run `consolidate_lseg.py`,
  which parses LSEG's messy header and writes all the clean files for you.
  Use this if the batch export is fiddly — it does the header/column naming
  automatically.

---

## BLOCK 1 — FX spot (32 currencies) → `Data/manual/lseg_fx_spot.csv`

RIC list (paste one per row into a blank Excel column):

    EUR=  GBP=  AUD=  NZD=
    JPY=  CHF=  CAD=  SEK=  NOK=  DKK=
    MXN=  BRL=  ZAR=  TRY=  INR=  KRW=  TWD=  THB=  SGD=  MYR=
    IDR=  COP=  CLP=  PEN=  PLN=  HUF=  CZK=  ILS=  CNY=  HKD=
    SAR=  KWD=

Route A steps (Datastream / Workspace Excel add-in, Time Series request):
1. Open Excel with the LSEG add-in signed in (your remote access).
2. On a blank sheet, paste the 32 RICs down column A, one per cell.
3. Add-in → **Time Series** request → click the **"Select Series from Sheet"**
   icon next to the Series/Lists box → drag over column A → **Select**.
4. Datatype: **ER (Exchange Rate Middle / MID)**.
5. Set From = `2019-01-01`, To = `2026-06-30`, Frequency = **Daily**.
6. Run. You get one grid: a date column + one column per currency.
7. **Rename the header row to bare ISO codes** — `EUR=` → `EUR`, `JPY=` → `JPY`,
   etc. (strip the `=`). The pipeline matches columns by bare ISO code; it will
   silently drop any column it doesn't recognise. Name the date column `date`.
8. **Leave the majors EUR/GBP/AUD/NZD as quoted (USD per 1 FC).** `get_data.py`
   inverts exactly those four itself (line ~383). Everything else stays as
   foreign-per-USD. Don't apply any currency conversion in the add-in.
9. File → Save As → **CSV** → `Data/manual/lseg_fx_spot.csv`.

If a RIC doesn't resolve, look it up in the Workspace app search bar, then add
the corrected code to the column and re-run.

---

## BLOCK 2 — 1-month FX forwards (same 32) → `Data/manual/lseg_fx_fwd1m_outright.csv`

This feeds `build_fx_factors.py` (Dollar + Carry factors). It is **not** merged
into the panel — `get_data.py` skips any file with "fwd"/"forward" in the name,
so the filename must contain `fwd`.

Preferred — **outright** rates (one number per day):
- Search each currency's **"1 MONTH FORWARD"** outright (WMR or Barclays),
  datatype ER/MID. Same 2019→2026 daily settings.
- Batch the same way (paste the outright codes in a column, Select Series from
  Sheet). Rename headers to bare ISO, date column `date`.
- Save as `Data/manual/lseg_fx_fwd1m_outright.csv`.

Fallback — **forward points** via `<CCY>1M=` RICs (EUR1M=, JPY1M=, …):
- Save as `Data/manual/lseg_fx_fwd1m_points.csv` instead, and set
  `FWD_IS_POINTS = True` in `build_fx_factors.py` (it converts points →
  outright using the spot file and the PIP scale).
- Pegged/managed DKK, HKD, SAR, KWD, CNY: pull them anyway; the factor build
  drops them from the carry sort automatically.

---

## BLOCK 3 — series not on FRED (native currency, do NOT convert)

Three single series. Easiest via Route B (Price History → one file each), or
just export each and name the column exactly as shown:

| RIC          | What                     | File                        | Column           | Basis |
|--------------|--------------------------|-----------------------------|------------------|-------|
| `XAU=`       | Gold spot, USD/oz        | `lseg_gold.csv`             | `Gold`           | MID   |
| `.STOXX50E`  | Euro Stoxx 50 price idx  | `lseg_stoxx.csv`            | `EuroStoxx50_EUR`| CLOSE |
| `TRNLTTFMc1` | TTF gas front month      | `lseg_ttf.csv`              | `TTF`            | CLOSE |

(Gold column is forgiving — `Gold`, `XAU`, `XAU=` are all auto-mapped to `Gold`.
Stoxx and TTF must match the column names above.) Keep Stoxx in EUR and TTF in
EUR on purpose — native quote keeps the asset return free of the EUR/USD leg.

---

## BLOCK 4 — optional cross-check backups

Only if cheap while you're in there; used to sanity-check FRED, not required.

| RIC     | Cross-checks       | Suggested file / column        |
|---------|--------------------|--------------------------------|
| `.VIX`  | FRED VIXCLS        | `lseg_vix.csv` / `VIX`         |
| `LCOc1` | FRED Brent spot    | `lseg_brent.csv` / `Brent`     |
| `.SPX`  | FRED SP500         | `lseg_spx.csv` / `SPX`         |

---

## After the exports land

Run order (from the thesis root):

    python3 "Code/consolidate_lseg.py"     # ONLY if you used Route B (per-RIC files in Data/manual/LSEG/)
    python3 "Code/get_data.py"             # merges lseg_*.csv, pulls FRED, builds daily_panel.csv
    python3 "Code/build_fx_factors.py"     # Dollar + Carry factors from spot + forwards
    python3 "Code/02_build_returns.py"     # log returns, portfolios, events
    python3 "Code/03_event_study.py"       # CARs + tables + figures
    python3 "Code/04_event_study_w50.py"   # long-horizon (persistence) CARs

Sanity checks:
- `get_data.py` should print `FX source: LSEG (lseg_fx_spot.csv)`.
- If a currency is missing, its header wasn't a recognised bare ISO code — fix
  the header and re-run from `get_data.py`.
- Gold: `03` prints `gold: ... -> r_Gold`; if it says "not in panel yet", the
  gold file/column name is off.
- Don't forget the **window fix** in `03_event_study.py` first
  (`EST_WIN=(-140,-21)`) — see `TODO.md`.
