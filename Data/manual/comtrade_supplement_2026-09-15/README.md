# Missing crude-oil trade data collected through the UN Comtrade API

Collected 15 September 2026, 18:10–18:15 CEST. Annual HS 2709, 2019–2024, values in current US dollars.

**Result:** Partner reports were retrieved for **31 of the 33 missing country–flow–year cells** in the original six-country extract. Türkiye’s previously missing export direction was retrieved for **all six years**. The observed trade balances support all seven existing basket assignments. Kuwait’s 2019 and 2023 import directions still return no records. Missing records have not been converted into confirmed zeros.

## Connection to your original collection

Your original `Code/comtrade_pull.py` documents the manual Comtrade portal query of 14 August 2026, then reproduces it using the official `comtradeapicall` Python package. Its keyless route calls the public UN Comtrade API endpoint below. It downloads reporter totals with partner World and, separately, partners’ exports to Türkiye.

This supplement uses **the same official API endpoint**, without a subscription key:

`https://comtradeapi.un.org/public/v1/preview/C/A/HS`

The data in this folder were returned by that API. No values were copied from webpages, search results or third-party datasets. The new collector uses Python `requests` to retain the original JSON responses, request URLs, parameters and timestamps. This is the same endpoint used internally by the original package.

The study’s classification window is 2019–2024, so the supplement excludes the original script’s additional 2025 observations. Secondary partner, customs procedure and transport mode are explicitly set to their totals, avoiding duplicate breakdowns. Each API response has fewer than 500 records. The largest has 19.

## What the Türkiye queries mean

| Target quantity | API reporter | API partner | API flow |
|---|---|---|---|
| Türkiye’s imports | All available reporters | Türkiye, code 792 | Exports, `X` |
| Türkiye’s exports | All available reporters | Türkiye, code 792 | Imports, `M` |

The second query is the direction missing from your original saved Türkiye file. The flow label always describes the reporting country’s transaction.

## Results

The following totals retain your frozen reporter values and add partner reports only where the original country–flow–year record is absent. Türkiye uses partner reports for both directions. This prevents double counting Japan’s known export observations and Mexico’s known 2019 imports.

| Country | Observed exports, USD bn | Observed imports, USD bn | Observed net exports, USD bn | Existing role supported by observed amounts |
|---|---:|---:|---:|---|
| India | 0.253897 | 728.369656 | −728.115759 | Importer |
| Japan | 0.068599 | 433.304343 | −433.235745 | Importer |
| Kuwait | 270.333462 | 0.000098 | +270.333364 | Exporter |
| Mexico | 142.557535 | 0.020804 | +142.536731 | Exporter |
| Taiwan / Other Asia, nes | 0.000018 | 133.895172 | −133.895154 | Importer |
| Saudi Arabia | 1,042.331367 | 0.901508 | +1,041.429859 | Exporter |
| Türkiye | 9.560085 | 28.583584 | −19.023499 | Importer |

**Türkiye’s exports should no longer be described as negligible.** The API reports about USD 9.56 billion through partner-country import records. Its observed import surplus remains about USD 19.02 billion.

## What remains uncertain

- **Kuwait:** no partner export records were returned for crude sent to Kuwait in 2019 or 2023. The direct reporter query also contains no corresponding import record. The collected imports for its other four years sum to about USD 97,793. An empty successful query remains unobserved, not a measured zero.
- **Partner coverage:** a nonempty response does not prove that every relevant trading partner reported. In the Türkiye import mirror, Iraq has no rows in any of the six years and Russia has no rows in 2022–2024. The USD 28.58 billion is therefore a sum of available partner reports, not a verified total of all Turkish imports.
- **Mirror comparability:** partner reports and a country’s own statistics can differ in valuation, timing, attribution and treatment of re-exports. These balances are evidence for the basket classification rather than a fully reconciled national trade account.
- **Data vintage:** the fresh reporter query has the same 111 country–flow–year records as the frozen extract. One value has been revised: Japan’s 2023 imports increased by USD 43,717,605.28, approximately 0.054%. That revision is saved separately and is not substituted into the summary above. It does not change Japan’s classification.

The collector and analysis deliberately make no claim of complete mirror coverage. There is no unconditional “PASS” based only on the existence of numeric rows.

## Files to use

- `mirror_partner_records_2019_2024.csv`: all **328 partner records**, with the original API fields plus target-country direction labels.
- `turkiye_both_directions_2019_2024.csv`: Türkiye’s imports and exports, measured from the partner side.
- `reporter_world_records_2019_2024.csv`: **111 fresh reporter totals**, for comparison with the original extract.
- `coverage_by_country_flow_year.csv`: records, amounts and remaining gaps for each of the 48 mirror queries.
- `classification_evidence_by_year.csv`: the country-year evidence used in the classification summary, with source labels and missing values preserved.
- `classification_summary_2019_2024.csv`: the seven-country summary above, at full precision.
- `observed_net_trade_by_year.csv`: annual observed trade balances.
- `reporter_vintage_comparison.csv`: comparison of fresh and frozen reporter values.
- `raw/` and `request_manifest.json`: original API JSON, parameters, URLs, timestamps, HTTP status and hashes.
- `quality_checks.json`: checks of query success, duplicates, negative or invalid values, row caps, coverage and classification signs.
- `collection_review.ipynb`: an inspectable notebook with the main checks and results.
- `collect_comtrade.py`: the API collector. Completed successful responses are cached, so rerunning it resumes unfinished work rather than replacing saved responses.
- `analyze_collection.py`: creates the coverage and classification outputs using these downloads and the frozen thesis extract.

## Reusing the API collector

From this folder, using your thesis environment:

```sh
"/Users/leon/Bachelor Thesis/.venv/bin/python" collect_comtrade.py
"/Users/leon/Bachelor Thesis/.venv/bin/python" analyze_collection.py
```

The collector now waits at least four seconds between new queries and backs off after rate-limit responses. The completed collection contains **54 successful queries**: 48 mirror queries and six reporter queries. A replay from the saved responses was also checked. No API key is required for these small queries.

To start a genuinely fresh vintage, copy the collector into a new empty folder and run it there. Keep the dated raw responses so later revisions can be distinguished from coding changes.

## Integrated on 16 September 2026

The working checker now reads the raw responses in this folder and verifies their
hashes. Original reporter cells take priority over supplementary mirrors. Run
`python Code/comtrade_check.py --offline` from the repository root. The result is
PASS WITH ASSUMPTIONS (exit 2), with all 13 observed signs matching their baskets.
The appendix and Section 5.1.3 explain the evidence and remaining gaps.

The original reporter and Türkiye import-mirror files are unchanged. The fresh
reporter comparison above is retained separately. Basket membership and the
pinned empirical results are unchanged. Do not interpret an empty response as
zero or a numeric response as complete partner coverage.

The analysis script resolves the repository relative to this installed folder.
The notebook can be rerun from this folder. The original request metadata and
raw response bytes are retained without alteration.
