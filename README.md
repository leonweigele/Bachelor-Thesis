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
2026. Market and risk-series inputs use the pull of 23 July 2026.
The Comtrade classification uses the original extract and the official API
supplement collected on 15 September 2026, as described below.

## Repository layout

| Path | Contents |
|---|---|
| `Code/` | The analysis pipeline, split by method since 16 September 2026: `data/` (downloads, FX factors, returns, Comtrade extracts and check), `event_study/` (03 to 05, market-model table, verifier), `regression/` (07, 08), `figures/` (06, Figure 4.1), `common/` (shared helpers `es_common.py`, `thesis_style.py`), `legacy/` (scripts no longer part of the run), `tests/` |
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
python Code/data/build_fx_factors.py          # tercile classification from the saved lseg_fx_spot / lseg_fx_fwd1m_points CSVs (carry_classification.csv)
python Code/data/02_build_returns.py          # returns, baskets, events.csv from the saved daily_panel.csv
python Code/event_study/03_event_study.py            # CARs, estimation window [-140,-21]
python Code/event_study/04_event_study_w50.py        # +/-50-day horizon, estimation window [-170,-51]
python Code/event_study/05_cross_event_tests.py      # difference-in-CARs across events
python Code/regression/07_safehaven_regression.py   # Table 6.5
python Code/regression/08_regression_sensitivity.py # sensitivity of Table 6.5 (standard errors, single days, placebo windows) quoted in Sections 6.6 and 7.5; writes Output/tables/regression_sensitivity/
python Code/event_study/make_mm_table.py             # Appendix Table 2 (market model)
python Code/figures/06_make_thesis_figures.py --install ; python Code/figures/fig41_overview.py --install
python Code/event_study/verify_results.py            # compare car_summary.csv, cross_event_diff.csv and car_persistence_w50.csv with the pinned baseline; exit 1 on any difference
python Code/data/comtrade_check.py --offline  # coverage-aware check of the oil-basket classification on the saved Comtrade extracts
```

The baseline in `Data/processed/event_study/baseline/` was pinned on
15 September 2026 (`python Code/event_study/verify_results.py --pin`) after the
`OIL_SPREAD` portfolio had been added and the retired 8 April `ceasefire`
event dropped; the previous pin of 23 July 2026 is kept next to it as
`*.bak_2026-07-23_prepin`. A matching run means the output files are
unchanged since the pin, nothing more; whether the thesis text quotes them
correctly is a separate check.

On 16 September 2026 `car_persistence_w50.csv`, the +/-50-day table behind the
(0,50) column of Table 6.1, was added to the verifier (v3, record
`.handoff/VERIFIER-W50-PROPOSAL-2026-09-16.md`). Its first pin was the file of
11 August 2026, reproduced byte for byte from the saved inputs beforehand, with
`python Code/event_study/verify_results.py --pin car_persistence_w50.csv`. Since v3 a pin
can name single files, the other baselines stay untouched, and any baseline
that is overwritten is first copied to `<file>.bak_<date>_prepin`. The
verifier's own tests run in temporary folders with
`python -m unittest discover -s Code/tests -p 'test_verify_results.py'`.

On 16 September 2026 the redundant `hormuz_closure` event was removed from
`02_build_returns.py`. Its date (Saturday 28 February 2026 for `hormuz`, Monday
2 March 2026 for the closure) mapped to the same trading day 0, so its rows
duplicated the `hormuz` rows exactly. After re-running 02, 03 and 04,
`car_summary.csv` (505 rows fewer) and `car_persistence_w50.csv` (15 rows fewer)
were re-pinned with `python Code/event_study/verify_results.py --pin car_summary.csv
car_persistence_w50.csv`. The previous pins are kept as `*.bak_2026-09-16_prepin`
and `cross_event_diff.csv` was unaffected. Record: `.handoff/HORMUZ-CLOSURE-REMOVAL-2026-09-16.md`.

On 16 September 2026 `08_regression_sensitivity.py` was added. It rebuilds the four
specifications of Table 6.5 from the saved inputs, ties them to
`Output/tables/safehaven_regression.csv` and to the printed table, and then reports the
tariff interaction under classical, White, HC3 and Newey-West standard errors with 0 to
21 lags, leave-one-out over the 21 tariff-window days, and placebo windows (the tariff
window moved to every other 21-day stretch that does not overlap the modelled windows).
It reads only the saved inputs and writes only `Output/tables/regression_sensitivity/`
(CSV per block, `provenance.json` with input hashes, full log). The numbers quoted in
Sections 6.6 and 7.5 come from `02_covariance_estimators.csv`, `03_leave_one_out.csv`
and `04_placebo_summary.csv` there.

## Comtrade evidence and remaining coverage gaps

`Code/data/comtrade_check.py --offline` now reads the original reporter totals and
the saved official API responses in `Data/manual/comtrade_supplement_2026-09-15/`.
It verifies response hashes and query directions. Existing reporter cells take
priority, so supplementary partner rows are never added to an already reported
country-flow-year value.

Partner reports supply 31 of the original 33 missing observations across six
countries. Kuwait's imports in 2019 and 2023 remain unobserved. Türkiye uses
partners' exports to Türkiye for imports and partners' imports from Türkiye for
exports. The observed totals are USD 28.58 billion and USD 9.56 billion,
respectively, giving an observed import surplus of USD 19.02 billion.

All 13 observed balance signs support the existing baskets. The verdict remains
**PASS WITH ASSUMPTIONS**, with exit code **2**. Numeric partner records do not
prove complete coverage. Türkiye's import mirror contains no Iraqi reports for
2019-2024 or Russian reports for 2022-2024. Empty responses are not confirmed
zeros, and observed net balances are not bounds on true net trade. Mirror and
reporter values may differ in valuation, timing and attribution.

The supplement retains raw JSON, request metadata, a coverage table and a review
notebook. Its fresh reporter totals are retained for comparison only. Japan's
2023 import revision is not substituted into the original values. See the
[collection record](Data/manual/comtrade_supplement_2026-09-15/README.md).
The basket lists in `02_build_returns.py` and the pinned event-study results
remain unchanged.

Optional local check outputs and regression tests:

```bash
python Code/data/comtrade_check.py --offline --report-dir Output/comtrade_check
python -m unittest discover -s Code/tests -p 'test_comtrade_check.py'
```

Exit code 0 means matching signs with reporter totals for every cell. Exit code
2 is the expected qualified result for the saved evidence. Exit code 1 means
an observed sign contradicts its basket, malformed data or insufficient evidence.
The offline check makes no API calls and does not change frozen inputs.

## Refreshing the raw data — not needed for reproduction; overwrites the saved inputs

```bash
python Code/data/lseg_pull.py                 # LSEG Workspace API (needs the app key in Code/data/lseg-data.config.json); rewrites Data/manual/lseg_*.csv
python Code/data/get_data.py                  # FRED (API key, or run off the university network), GPR, TPU; rebuilds Data/processed/daily_panel.csv
python Code/data/comtrade_pull.py             # UN Comtrade HS 2709 extracts; runs the classification check afterwards
```

After a refresh, rerun the reproduction steps above; the verifier will then
report differences against the pinned baseline, which is expected.

Notes:

- `Data/manual/LSEG_legacy_ends_2026-06-16/` holds the manual LSEG Excel
  exports of June 2026 (series end 16 June 2026, gold as `XAU=ZKBZ`, one
  Brazilian NDF file). `Code/legacy/consolidate_lseg.py` belongs to that legacy
  route only; run over the shipped CSVs it would overwrite the 23 July inputs,
  so it refuses to write unless called with `--force`.
- `Data/raw/data_gpr_daily_recent.xls` (2 July 2026) is a superseded manual
  download; the pipeline reads `Data/raw/gpr_daily.csv` (23 July 2026).
- `Code/legacy/plot_currency_sensitivities.py` drew a figure that is no longer in the
  thesis and is not part of the run.
- `Code/data/fred_api_key.txt` and `Code/data/lseg-data.config.json` are local
  credentials, ignored by git and not part of anything handed in.

Scripts run on whatever data is present and warn about gaps rather than
crashing, so steps can be re-run individually.
