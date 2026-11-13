"""Write a self-contained, repeatable After Effects handoff (no video output)."""
import json
from bisect import bisect_left
from pathlib import Path
from textwrap import wrap
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


def timed_text(entries, text_key):
    result = []
    for entry in entries:
        try:
            start, end = float(entry['start']), float(entry['end'])
            text = ' '.join(entry[text_key].split())
        except (KeyError, AttributeError, TypeError, ValueError):
            continue
        if text and 0 <= start < end < float('inf'):
            result.append(((start + end) / 2, start, end, text))
    return sorted(result)


def subtitle_cues(item, record):
    """Build source-time captions from the words audible inside one trim."""
    if record.get('pending'):
        return []
    clip_start, clip_end = item['start'], item['end']
    words = timed_text(record.get('words', []), 'word')
    midpoints = [word[0] for word in words]
    cues = []
    for _, start, end, text in timed_text(record.get('segments', []), 'text'):
        if start >= clip_end or end <= clip_start:
            continue
        left = bisect_left(midpoints, start)
        right = bisect_left(midpoints, end)
        phrase = words[left:right]
        if not phrase:
            # Keep the saved segment timing when there are no word timestamps.
            cues.append(caption([(start, end, text)], clip_start, clip_end))
            continue
        group = []
        for midpoint, word_start, word_end, token in phrase:
            if not clip_start <= midpoint < clip_end:
                continue
            if group:
                line = ' '.join([word[2] for word in group] + [token])
                if (len(wrap(line, width=36)) > 2 or word_end - group[0][0] > 5
                        or word_start - group[-1][1] > .7):
                    cues.append(caption(group, clip_start, clip_end))
                    group = []
            group.append((word_start, word_end, token))
        if group:
            cues.append(caption(group, clip_start, clip_end))
    return sorted(cues, key=lambda cue: (cue['start'], cue['end']))


def caption(words, clip_start, clip_end):
    return {'start': max(clip_start, words[0][0]), 'end': min(clip_end, words[-1][1]),
            'text': '\r'.join(wrap(' '.join(word[2] for word in words), width=36))}


def write_exchange(data, items, records, *, include_subtitles=False):
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
        clip = {'id': item['id'], 'path': sources[key], 'fileName': record['name'],
                'label': label, 'start': item['start'], 'end': item['end']}
        if include_subtitles:
            clip['captions'] = subtitle_cues(item, record)
        payload['clips'].append(clip)
    path = exchange_path(data)
    script = TEMPLATE.read_text(encoding='utf-8').replace('/*__VIDEOWORDS_DATA__*/ null',
                                                          json.dumps(payload, ensure_ascii=True, allow_nan=False),
                                                          1)
    atomic_text(path, '\ufeff' + script)
    return path
