# Bachelor Thesis — Geopolitical Risk and Currency Markets

University of Mannheim, 2026

## What this thesis does

It measures how currency markets — and in particular the **US dollar's
safe-haven role** — react to geopolitical risk shocks, and asks whether
that reaction depends on the *type* of shock.

The core method is an **event study**: around each dated event it computes
abnormal exchange-rate returns and cumulative abnormal returns (CARs) versus
a constant-mean benchmark (estimation window −130…−11, event window ±20
trading days), with per-event significance tests. The events span two kinds
of shock — **trade-policy** shocks (the 2025 "Liberation Day" tariffs and
the tariff pause) and **military / oil-supply** shocks (the 2025 Iran
12-day war and the 2026 Strait of Hormuz war, closure, and ceasefires), plus
the 2022 Russia–Ukraine invasion as a non-US reference event.

On top of the event study it:

- builds the standard **FX factors** (Dollar `DOL` and Carry `HML`) and
  sorts currencies into safe/funding vs. risky/carry buckets;
- runs a **safe-haven regression** — an OLS test with a VIX interaction and
  Newey-West (HAC) standard errors — of whether the dollar's sensitivity to
  risk changed during the tariff shock relative to the Hormuz/Ukraine shocks
  (a transparent alternative to a DCC-GARCH);
- tests the **"mirror image"** hypothesis with formal difference-in-CARs
  tests *across* events (trade-policy vs. military/oil-supply);
- checks robustness with a long-horizon (±50 trading day) event window to
  see whether the initial reactions persist or revert.

**Data:** FRED (USD indices, yields, TIPS, VIX, oil, S&P 500, ~21 daily
bilateral FX rates) and LSEG/Refinitiv price-history exports; no Yahoo
Finance.

## Repository layout

| Path | Contents |
|---|---|
| `Code/` | The analysis pipeline (data download → factors → event study → regressions → figures) |
| `Data/` | `raw/`, `processed/`, and `manual/` (incl. LSEG exports) datasets |
| `Main/LaTeX Thesis/` | The thesis document (LaTeX source, chapters, tables) |
| `Output/` | Generated figures and result tables |
| `Literature/` | Reference papers, grouped by topic |

## Running the pipeline

```bash
python Code/get_data.py                    # download FRED + LSEG data
python Code/consolidate_lseg.py            # clean LSEG exports into panels
python Code/build_fx_factors.py            # rebuild DOL & HML factors, safe/risky classes
python Code/02_build_returns.py            # log returns, portfolios, event windows, Figure 1
python Code/03_event_study.py              # abnormal returns & CARs per event
python Code/04_event_study_w50.py          # long-horizon (+/-50d) robustness
python Code/05_cross_event_tests.py        # difference-in-CARs across events
python Code/07_safehaven_regression.py     # safe-haven VIX-interaction regression
python Code/plot_currency_sensitivities.py # currency-classification figure
```

Scripts are incremental — they run on whatever data has been downloaded so
far and warn about gaps rather than crashing, so steps can be re-run as new
data lands.
