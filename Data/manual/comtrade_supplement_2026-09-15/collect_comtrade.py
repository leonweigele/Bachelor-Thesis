"""Collect public HS 2709 records separately from the frozen thesis inputs.

No returned rows is recorded as missing, never as a confirmed zero.
Valid returned rows are not described as complete partner coverage.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
RAW = OUT / 'raw'
RAW.mkdir(exist_ok=True)
ENDPOINT = 'https://comtradeapi.un.org/public/v1/preview/C/A/HS'
YEARS = range(2019, 2025)
COUNTRIES = {'IND': 699, 'JPN': 392, 'KWT': 414, 'MEX': 484,
             'S19': 490, 'SAU': 682, 'TUR': 792}
ALL_REPORTERS = [76, 124, 170, 392, 410, 414, 484, 490, 579, 682, 699, 764, 792]
MISSING_DIRECTION = {'IND': 'X', 'JPN': 'X', 'KWT': 'M', 'MEX': 'M',
                     'S19': 'X', 'SAU': 'M', 'TUR': 'M,X'}
session = requests.Session()
session.headers['User-Agent'] = 'ThesisTradeVerification/1.0 (public research)'
manifest = []
DELAY_SECONDS = 4.0


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(label, parameters, context):
    params = {'cmdCode': '2709', 'partner2Code': '0', 'customsCode': 'C00',
              'motCode': '0', 'maxRecords': 500, 'format': 'JSON',
              'includeDesc': 'true', **parameters}
    raw_path, meta_path = RAW / f'{label}.json', RAW / f'{label}.request.json'
    meta = {'label': label, 'endpoint': ENDPOINT, 'parameters': params, **context}
    if raw_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('status') == 'ok':
            payload = json.loads(raw_path.read_text())
            manifest.append(meta)
            print(label, 'cached', len(payload['data']), 'rows', flush=True)
            return payload['data']
    payload = None
    for attempt in range(1, 4):
        meta.update(retrieved_at=now(), attempt=attempt)
        try:
            response = session.get(ENDPOINT, params=params, timeout=(10, 30))
            meta.update(url=response.url, http_status=response.status_code)
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                meta['retry_after'] = retry_after
            raw_path.write_bytes(response.content)
            meta['sha256'] = hashlib.sha256(response.content).hexdigest()
            response.raise_for_status()
            payload = response.json()
            assert isinstance(payload.get('data'), list), 'No data array in response'
            assert not payload.get('error'), f"API error: {payload.get('error')}"
            assert len(payload['data']) < 500, 'Row cap reached; requires split query'
            assert payload.get('count') == len(payload['data']), 'Count differs from rows'
            for row in payload['data']:
                assert str(row['cmdCode']) == '2709', 'Wrong commodity'
                assert str(row['refYear']) == str(params['period']), 'Wrong year'
                assert str(row['partnerCode']) == str(params['partnerCode']), 'Wrong partner'
                assert row['flowCode'] in params['flowCode'].split(','), 'Wrong direction'
                assert str(row['partner2Code']) == '0', 'Unexpected secondary partner'
                assert str(row['customsCode']) == 'C00', 'Unexpected customs breakdown'
                assert str(row['motCode']) == '0', 'Unexpected transport breakdown'
            meta.update(status='ok', row_count=len(payload['data']),
                        reported_count=payload.get('count'), api_error=payload.get('error'))
            break
        except (requests.RequestException, ValueError, AssertionError, KeyError) as exc:
            meta.update(status='error', error=str(exc))
            print(label, 'attempt', attempt, type(exc).__name__, str(exc)[:160], flush=True)
            payload = None
            if attempt < 3:
                requested_delay = meta.get('retry_after', '')
                wait = float(requested_delay) if str(requested_delay).isdigit() else 0
                time.sleep(max(10 * attempt, wait))
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    manifest.append(meta)
    (OUT / 'request_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    time.sleep(DELAY_SECONDS)
    if payload is None:
        return []
    print(label, meta['row_count'], 'rows', flush=True)
    return payload['data']


def main():
    reporter_rows, mirror_rows = [], []
    # Both directions for Türkiye, explicitly keeping the target perspective.
    for iso in ['TUR', 'IND', 'JPN', 'KWT', 'MEX', 'S19', 'SAU']:
        for target_flow in MISSING_DIRECTION[iso].split(','):
            partner_flow = 'X' if target_flow == 'M' else 'M'
            for year in YEARS:
                label = f'mirror_{iso}_target_{target_flow}_{year}'
                rows = fetch(label, {'period': str(year), 'flowCode': partner_flow,
                                    'partnerCode': str(COUNTRIES[iso])},
                             {'query_kind': 'mirror', 'target_iso': iso,
                              'target_flow': target_flow, 'year': year})
                mirror_rows.extend({**row, 'query_label': label,
                                    'target_iso': iso, 'target_flow': target_flow}
                                   for row in rows)
    # Current reporter totals, for comparison with the saved 2019–2024 extract.
    for year in YEARS:
        label = f'reporter_world_{year}'
        rows = fetch(label, {'period': str(year), 'flowCode': 'M,X',
                            'partnerCode': '0',
                            'reporterCode': ','.join(map(str, ALL_REPORTERS))},
                     {'query_kind': 'reporter', 'year': year})
        reporter_rows.extend({**row, 'query_label': label} for row in rows)
    pd.DataFrame(mirror_rows).to_csv(OUT / 'mirror_partner_records_2019_2024.csv', index=False)
    pd.DataFrame(reporter_rows).to_csv(OUT / 'reporter_world_records_2019_2024.csv', index=False)
    (OUT / 'request_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('DONE:', len(manifest), 'queries;', len(mirror_rows), 'mirror rows;',
          len(reporter_rows), 'reporter rows;',
          sum(m['status'] != 'ok' for m in manifest), 'failed queries', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delay', type=float, default=4.0,
                        help='Seconds between new API queries (minimum 4; default 4).')
    args = parser.parse_args()
    DELAY_SECONDS = max(4.0, args.delay)
    main()
