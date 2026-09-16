"""Inspect downloaded API records without changing the thesis's frozen inputs."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
THESIS = OUT.parents[2]
NAME = {'IND': 'India', 'JPN': 'Japan', 'KWT': 'Kuwait', 'MEX': 'Mexico',
        'S19': 'Taiwan (Other Asia, nes)', 'SAU': 'Saudi Arabia', 'TUR': 'Türkiye'}
ROLE = {'IND': 'importer', 'JPN': 'importer', 'KWT': 'exporter', 'MEX': 'exporter',
        'S19': 'importer', 'SAU': 'exporter', 'TUR': 'importer'}
baseline_path = THESIS / 'Data/manual/comtrade_crude_2709_api.csv'
baseline_mirror_path = THESIS / 'Data/manual/comtrade_crude_2709_mirror_tur.csv'


def main():
    manifest = json.loads((OUT / 'request_manifest.json').read_text())
    mirror = pd.read_csv(OUT / 'mirror_partner_records_2019_2024.csv')
    current = pd.read_csv(OUT / 'reporter_world_records_2019_2024.csv')
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline.refYear.between(2019, 2024)].copy()
    assert not mirror.duplicated(['query_label', 'reporterCode']).any()
    assert not current.duplicated(['reporterCode', 'flowCode', 'refYear']).any()
    assert not baseline.duplicated(['reporterCode', 'flowCode', 'refYear']).any()
    for frame in [mirror, current, baseline]:
        assert frame.refYear.between(2019, 2024).all()
        assert frame.cmdCode.eq(2709).all()
        assert np.isfinite(pd.to_numeric(frame.primaryValue, errors='coerce')).all()
        assert frame.primaryValue.ge(0).all()
    assert all(m.get('row_count', 0) < 500 for m in manifest)

    # Each row describes the target country's own trade direction. The API's
    # reporter perspective is retained separately to prevent swapping imports/exports.
    coverage = []
    for m in manifest:
        if m['query_kind'] != 'mirror':
            continue
        iso, flow, year = m['target_iso'], m['target_flow'], m['year']
        rows = mirror[mirror.query_label.eq(m['label'])]
        old = baseline[baseline.reporterISO.eq(iso) & baseline.flowCode.eq(flow)
                       & baseline.refYear.eq(year)]
        new = current[current.reporterISO.eq(iso) & current.flowCode.eq(flow)
                      & current.refYear.eq(year)]
        coverage.append({
            'country': NAME[iso], 'target_iso': iso, 'target_flow': flow,
            'target_flow_meaning': 'imports' if flow == 'M' else 'exports',
            'year': year, 'query_label': m['label'],
            'api_partner_flow': m['parameters']['flowCode'],
            'api_status': m['status'], 'returned_partner_rows': len(rows),
            'partner_reporters': ','.join(sorted(rows.reporterISO.unique())),
            'mirror_observed_usd': rows.primaryValue.sum() if len(rows) else np.nan,
            'frozen_reporter_usd': old.primaryValue.iloc[0] if len(old) else np.nan,
            'current_reporter_usd': new.primaryValue.iloc[0] if len(new) else np.nan,
            'was_missing_from_frozen_reporter': not len(old),
            'mirror_status': ('query_failed' if m['status'] != 'ok' else
                              'observed_partner_reports_coverage_not_established' if len(rows)
                              else 'no_records_returned_not_a_confirmed_zero'),
        })
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(OUT / 'coverage_by_country_flow_year.csv', index=False)

    # Keep frozen reported cells. Add mirrors only to missing cells, avoiding
    # double counting known Japan exports and Mexico's known 2019 imports.
    cells = []
    for iso in NAME:
        for flow in ['M', 'X']:
            for year in range(2019, 2025):
                old = baseline[baseline.reporterISO.eq(iso) & baseline.flowCode.eq(flow)
                               & baseline.refYear.eq(year)]
                mir = coverage[coverage.target_iso.eq(iso) & coverage.target_flow.eq(flow)
                               & coverage.year.eq(year)]
                value, source = np.nan, 'unresolved'
                if len(old):
                    value, source = old.primaryValue.iloc[0], 'frozen_reporter'
                elif len(mir) and pd.notna(mir.mirror_observed_usd.iloc[0]):
                    value, source = mir.mirror_observed_usd.iloc[0], 'new_partner_reports'
                cells.append({'target_iso': iso, 'target_flow': flow, 'year': year,
                              'observed_usd': value, 'source': source,
                              'complete_partner_coverage_established': False if source != 'frozen_reporter' else None})
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / 'classification_evidence_by_year.csv', index=False)

    summary = []
    for iso, name in NAME.items():
        c = cells[cells.target_iso.eq(iso)]
        x = c[c.target_flow.eq('X')].observed_usd.sum(min_count=1)
        im = c[c.target_flow.eq('M')].observed_usd.sum(min_count=1)
        net = x - im
        role = 'exporter' if net > 0 else 'importer' if net < 0 else 'undetermined'
        gaps = c[c.observed_usd.isna()]
        summary.append({
            'country': name, 'target_iso': iso, 'existing_basket': ROLE[iso],
            'observed_exports_usd': x, 'observed_imports_usd': im,
            'observed_net_exports_usd': net, 'observed_role': role,
            'observed_role_matches_existing_basket': role == ROLE[iso],
            'unresolved_country_flow_years': len(gaps),
            'unresolved_cells': ';'.join(f'{r.target_flow}-{r.year}' for r in gaps.itertuples()),
            'interpretation': 'Observed amounts support this role; mirror coverage remains incomplete or unestablished',
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'classification_summary_2019_2024.csv', index=False)
    mirror[mirror.target_iso.eq('TUR')].to_csv(OUT / 'turkiye_both_directions_2019_2024.csv', index=False)
    country_year = cells.groupby(['target_iso', 'year', 'target_flow']).observed_usd.sum(min_count=1).unstack()
    country_year['observed_net_exports_usd'] = country_year['X'] - country_year['M']
    country_year.to_csv(OUT / 'observed_net_trade_by_year.csv')

    # Compare fresh reporter values with the original saved extract.
    key = ['reporterISO', 'flowCode', 'refYear']
    comp = baseline[key + ['primaryValue']].merge(current[key + ['primaryValue']],
            on=key, how='outer', suffixes=('_frozen', '_current'), indicator=True)
    comp['difference_usd'] = comp.primaryValue_current - comp.primaryValue_frozen
    comp['changed'] = comp._merge.ne('both') | ~np.isclose(
        comp.primaryValue_current, comp.primaryValue_frozen, rtol=0, atol=0.01)
    comp.to_csv(OUT / 'reporter_vintage_comparison.csv', index=False)

    missing_six = coverage[coverage.target_iso.ne('TUR') & coverage.was_missing_from_frozen_reporter]
    turkey_exports = coverage[coverage.target_iso.eq('TUR') & coverage.target_flow.eq('X')]
    checks = {
        'source': 'UN Comtrade official public API',
        'successful_queries': sum(m['status'] == 'ok' for m in manifest),
        'failed_queries': sum(m['status'] != 'ok' for m in manifest),
        'mirror_queries': sum(m['query_kind'] == 'mirror' for m in manifest),
        'reporter_queries': sum(m['query_kind'] == 'reporter' for m in manifest),
        'raw_mirror_rows': len(mirror), 'raw_reporter_rows': len(current),
        'empty_mirror_queries': int(coverage.returned_partner_rows.eq(0).sum()),
        'maximum_response_rows': max(m.get('row_count', 0) for m in manifest),
        'duplicate_mirror_reporter_keys': 0, 'duplicate_reporter_keys': 0,
        'invalid_or_negative_values': 0,
        'original_missing_six_country_cells': len(missing_six),
        'original_missing_cells_now_with_partner_records': int(missing_six.mirror_observed_usd.notna().sum()),
        'turkiye_export_years_with_partner_records': int(turkey_exports.mirror_observed_usd.notna().sum()),
        'all_observed_net_signs_match_existing_baskets': bool(summary.observed_role_matches_existing_basket.all()),
        'fresh_reporter_rows_changed_or_added_or_removed': int(comp.changed.sum()),
        'no_complete_mirror_coverage_claim': True,
        'baseline_file': str(baseline_path),
        'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'baseline_mirror_sha256': hashlib.sha256(baseline_mirror_path.read_bytes()).hexdigest(),
    }
    (OUT / 'quality_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))
    print(summary[['country', 'existing_basket', 'observed_exports_usd',
                   'observed_imports_usd', 'observed_net_exports_usd',
                   'unresolved_cells']].to_string(index=False))


if __name__ == '__main__':
    main()
