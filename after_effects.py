"""Write a self-contained, repeatable After Effects handoff (no video output)."""
import json
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from artifacts import artifact_lock
from library import atomic_text, video_path


SCRIPT_NAME = 'VideoWords.jsx'
TEMPLATE = Path(__file__).with_name('after_effects.jsx')


def exchange_path(data):
    return Path(data) / 'After Effects' / SCRIPT_NAME


def exchange_id(data):
    path = Path(data) / 'after-effects-id'
    with artifact_lock(path):
        if not path.exists():
            atomic_text(path, uuid4().hex)
        return path.read_text(encoding='utf-8').strip()


def unlinked_records(items, records):
    blocked = []
    for key in dict.fromkeys(item['record_key'] for item in items):
        try:
            video_path(records[key]['original_path'])
        except (KeyError, OSError, ValueError):
            blocked.append(key)
    return blocked


def write_exchange(data, items, records):
    missing = unlinked_records(items, records)
    if missing:
        raise ValueError('Привяжите оригиналы: ' + ', '.join(records[k]['name'] for k in missing))
    payload = {'id': exchange_id(data), 'clips': []}
    sources = {}
    for item in items:
        key = item['record_key']
        record = records[key]
        if key not in sources:
            sources[key] = quote(str(video_path(record['original_path'])), safe='/')
        label = ' '.join((record.get('person') or record['name']).split())
        payload['clips'].append({'id': item['id'], 'path': sources[key],
                                 'fileName': record['name'], 'label': label,
                                 'start': item['start'], 'end': item['end']})
    path = exchange_path(data)
    script = TEMPLATE.read_text(encoding='utf-8').replace('/*__VIDEOWORDS_DATA__*/ null',
                                                          json.dumps(payload, ensure_ascii=True, allow_nan=False),
                                                          1)
    atomic_text(path, '\ufeff' + script)
    return path
