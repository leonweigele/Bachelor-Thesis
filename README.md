# Bachelor Thesis — The Dollar's Safe-Haven Status Across Two Shocks

University of Mannheim, 2026

## What this thesis does

It measures how currency markets — and in particular the **U.S. dollar's
safe-haven role** — react to two kinds of shock, and asks whether that
reaction depends on the *type* of shock: the 2025 "Liberation Day" tariff
announcement (a U.S. trade-policy shock) against the 2026 Strait of Hormuz
crisis (a military and oil-supply shock), with the 2022 Russian invasion of
Ukraine as an external benchmark and the 2025 twelve-day Israel–Iran war as a
second comparison episode.

The core method is an **event study**: around each dated event it computes
abnormal returns and cumulative abnormal returns (CARs) against a
constant-mean benchmark (estimation window [−140, −21], event window
[−20, +20] trading days; [−170, −51] and [−50, +50] for the long-horizon
check), with basket-level t-tests and a formal difference-in-CARs test
*across* events.

On top of the event study it:

- sorts the 23 floating currencies with a clean one-month forward quote into
  safe/risky terciles by their average forward discount (the classification
  behind the event-study baskets; the same script also writes monthly
  DOL/CARRY factors, which feed no reported result);
- tracks the oil channel with oil-exporter and oil-importer baskets
  classified from UN Comtrade net crude-oil trade 2019–2024;
- runs a **safe-haven regression** — OLS with risk-measure × crisis-window
  interactions (VIX, VIX + GPR, VXY, GPR + TPU) and Newey–West standard
  errors — as supporting evidence for the event study;
- checks robustness with a market model (equal-weighted dollar basket as the
  market return), a second, literature-standard basket scheme and the
  ±50-day window.

**Data:** FRED (broad dollar indices, 10-year yield, VIX, Brent and WTI,
S&P 500; H.10 bilateral rates as a cross-check), LSEG Workspace (32-currency
spot panel, one-month forward points for 28 currencies, gold, Euro Stoxx 50,
TTF gas, J.P. Morgan VXY), the Caldara–Iacoviello GPR index, the Caldara et
al. TPU index and UN Comtrade (HS 2709). Sample 1 January 2019 to 30 June
2026; inputs frozen with the pull of 23 July 2026.

## Repository layout

| Path | Contents |
|---|---|
| `Code/` | The analysis pipeline (factors → returns → event study → regressions → figures), the download scripts and the two checks |
| `Data/` | `raw/` (FRED, GPR, TPU), `manual/` (LSEG CSVs, Comtrade extracts, legacy LSEG exports) and `processed/` (panel, returns, event-study outputs, pinned baseline) |
| `Main/LaTeX Thesis/` | The thesis document (LaTeX source, chapters, tables) |
| `Output/` | Generated figures and result tables |
| `Literature/` | Reference papers, grouped by topic (`Literature/Literature-md` holds a Markdown mirror) |

## Reproducing the thesis outputs from the saved inputs

These steps touch nothing in `Data/manual/`, `Data/raw/` or
`Data/processed/daily_panel.csv`. The canonical inputs are the LSEG pull of
23 July 2026 (`Data/manual/lseg_*.csv`, series to 30 June 2026),
`Data/raw/gpr_daily.csv` and `Data/raw/tpu_daily.csv` (TPU), both downloaded
on 23 July 2026, and the Comtrade extracts in `Data/manual/`.

```bash
python Code/build_fx_factors.py          # tercile classification from the saved lseg_fx_spot / lseg_fx_fwd1m_points CSVs (carry_classification.csv)
python Code/02_build_returns.py          # returns, baskets, events.csv from the saved daily_panel.csv
python Code/03_event_study.py            # CARs, estimation window [-140,-21]
python Code/04_event_study_w50.py        # +/-50-day horizon, estimation window [-170,-51]
python Code/05_cross_event_tests.py      # difference-in-CARs across events
python Code/07_safehaven_regression.py   # Table 6.5
python Code/make_mm_table.py             # Appendix Table 2 (market model)
python Code/06_make_thesis_figures.py --install ; python Code/fig41_overview.py --install
python Code/verify_results.py            # compare car_summary.csv and cross_event_diff.csv with the pinned baseline; exit 1 on any difference
python Code/comtrade_check.py --offline  # coverage-aware check of the oil-basket classification on the saved Comtrade extracts
```

The baseline in `Data/processed/event_study/baseline/` was pinned on
15 September 2026 (`python Code/verify_results.py --pin`) after the
`OIL_SPREAD` portfolio had been added and the retired 8 April `ceasefire`
event dropped; the previous pin of 23 July 2026 is kept next to it as
`*.bak_2026-07-23_prepin`. A matching run means the output files are
unchanged since the pin, nothing more; whether the thesis text quotes them
correctly is a separate check.

## Refreshing the raw data — not needed for reproduction; overwrites the saved inputs

```bash
python Code/lseg_pull.py                 # LSEG Workspace API (needs the app key in Code/lseg-data.config.json); rewrites Data/manual/lseg_*.csv
python Code/get_data.py                  # FRED (API key, or run off the university network), GPR, TPU; rebuilds Data/processed/daily_panel.csv
python Code/comtrade_pull.py             # UN Comtrade HS 2709 extracts; runs the classification check afterwards
```

After a refresh, rerun the reproduction steps above; the verifier will then
report differences against the pinned baseline, which is expected.

Notes:

- `Data/manual/LSEG_legacy_ends_2026-06-16/` holds the manual LSEG Excel
  exports of June 2026 (series end 16 June 2026, gold as `XAU=ZKBZ`, one
  Brazilian NDF file). `Code/consolidate_lseg.py` belongs to that legacy
  route only; run over the shipped CSVs it would overwrite the 23 July inputs,
  so it refuses to write unless called with `--force`.
- `Data/raw/data_gpr_daily_recent.xls` (2 July 2026) is a superseded manual
  download; the pipeline reads `Data/raw/gpr_daily.csv` (23 July 2026).
- `Code/plot_currency_sensitivities.py` drew a figure that is no longer in the
  thesis and is not part of the run.
- `Code/fred_api_key.txt` and `Code/lseg-data.config.json` are local
  credentials, ignored by git and not part of anything handed in.

Scripts run on whatever data is present and warn about gaps rather than
crashing, so steps can be re-run individually.
