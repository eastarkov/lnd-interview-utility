import re
import subprocess
import imageio_ffmpeg


def ffmpeg(args):
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])


def normalize(text):
    return re.findall(r'[^\W_]+', text.casefold().replace('ё', 'е'), re.UNICODE)


def find_matches(words, query):
    tokens, owners = [], []
    for i, word in enumerate(words):
        parts = normalize(word['word'])
        tokens.extend(parts)
        owners.extend([i] * len(parts))
    needle = normalize(query)
    if not needle:
        return []
    found = []
    for i in range(len(tokens) - len(needle) + 1):
        if tokens[i:i + len(needle)] == needle:
            first, last = owners[i], owners[i + len(needle) - 1]
            found.append({'start': words[first]['start'], 'end': words[last]['end'],
                          'text': ''.join(w['word'] for w in words[max(0, first-5):last+6])})
    return found


def stamp(seconds, separator=','):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{secs:02}{separator}{ms:03}'


def srt(segments):
    return '\n\n'.join(f"{i}\n{stamp(s['start'])} --> {stamp(s['end'])}\n{s['text'].strip()}" for i, s in enumerate(segments, 1)) + '\n'


def extract_audio(source, target):
    ffmpeg(['-i', source, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', target])


def timecode(seconds):
    ms = max(0, round(float(seconds) * 1000))
    minutes, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f'{minutes:02}:{sec:02}' + (f'.{ms:03}'.rstrip('0') if ms else '')


def parse_time(value):
    parts = value.strip().replace(',', '.').split(':')
    if len(parts) not in (2, 3) or any(not re.fullmatch(r'\d+', x) for x in parts[:-1]) or not re.fullmatch(r'\d+(?:\.\d+)?', parts[-1]):
        raise ValueError('Введите время в формате ММ:СС или ММ:СС.ммм.')
    if any(float(x) >= 60 for x in parts[1:]):
        raise ValueError('Секунды и минуты внутри ЧЧ:ММ:СС должны быть меньше 60.')
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(parts)))


def alternatives(query):
    # Explicit alternatives only. Never add translations or synonyms silently.
    return sorted({part.strip() for part in re.split(r'[,;\n]+', query) if normalize(part)})


def person_query(query, people):
    for person in sorted(set(people), key=len, reverse=True):
        if not person.strip():
            continue
        # Only consume a complete prefix; "Дамир" does not match "Дамирбек".
        match = re.match(r'^\s*' + re.escape(person.strip()) + r'(?:\s+|[,;:]\s*|$)', query, re.IGNORECASE)
        if match:
            return person, query[match.end():].strip()
    return None, query.strip()


def lexical_matches(record, query):
    result = {}
    from words import word_matches
    for phrase in alternatives(query):
        for match in word_matches(record['words'], phrase):
            relevant = [s for s in record['segments'] if s['end'] > match['start'] and s['start'] < match['end']]
            start = relevant[0]['start'] if relevant else match['start']
            end = relevant[-1]['end'] if relevant else match['end']
            result[(start, end)] = {'start': start, 'end': end, 'text': ' '.join(s['text'].strip() for s in relevant) or match['text'], 'kind': 'Совпадение слов'}
    return sorted(result.values(), key=lambda m: m['start'])
