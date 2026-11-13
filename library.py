"""Durable library. Only transcript metadata is replaced on re-recognition."""
import json
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from artifacts import artifact_lock

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.m4v', '.avi', '.mxf', '.mts', '.m2ts'}


def video_path(value):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f'Не поддерживается как видео: {path.name}')
    return path


def content_key(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:24]


def choose_videos(multiple=True):
    # Standard Additions opens the local Mac file picker; no Finder automation.
    flag = 'true' if multiple else 'false'
    script = """
var local = Application.currentApplication();
local.includeStandardAdditions = true;
try {
    var paths = local.chooseFile({withPrompt: "LND Interview Utility", multipleSelectionsAllowed: FLAG});
    if (!Array.isArray(paths)) paths = [paths];
    JSON.stringify(paths.map(function (path) { return path.toString(); }));
} catch (error) {
    if (error.errorNumber !== -128) throw error;
    "[]";
}
""".replace('FLAG', flag)
    result = subprocess.run(['/usr/bin/osascript', '-l', 'JavaScript', '-e', script],
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        if '(-128)' in result.stderr:
            return []
        raise RuntimeError('Не удалось открыть окно выбора файлов. Можно указать путь вручную.')
    try:
        return json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError('Не удалось прочитать выбранные пути. Можно указать их вручную.') from error


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def save_record(base, key, record):
    atomic_json(Path(base)/key/'record.json', {k: v for k, v in record.items()
                if k in {'name','content_key','duration','media_duration','segments','words','person','summary','recognition','pending','original_path'}})


def update_record(base, key, changes, expected_segments=None):
    path = Path(base) / key / 'record.json'
    with artifact_lock(path):
        record = json.loads(path.read_text(encoding='utf-8'))
        if expected_segments is not None and record.get('segments') != expected_segments:
            return
        record.update(changes)
        save_record(base, key, record)


def resolve_source(folder, record):
    original = record.get('original_path')
    cached = next((p for p in sorted(Path(folder).glob('source.*')) if p.is_file()), None)
    # If an original moved, retain its path for AE and use the legacy copy for preview if available.
    if original:
        return str(original if Path(original).is_file() or cached is None else cached)
    if cached is None:
        raise FileNotFoundError('Файл записи не найден.')
    return str(cached)


def load_records(base):
    records, errors = {}, []
    for path in sorted(Path(base).glob('*/record.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            record.update(source=resolve_source(path.parent, record), audio=str(path.parent/'audio.wav'))
            records[path.parent.name] = record
        except (ValueError, OSError) as error:
            errors.append(f'{path.parent.name}: {error}')
    return records, errors


def delete_transcript(base, key, record):
    for field in ('summary', 'vectors', 'passages', 'search_vectors', 'search_passages'):
        record.pop(field, None)
    record.update(words=[], segments=[], pending=True)
    for name in ('vectors.npz', 'search_vectors.npz'):
        (Path(base)/key/name).unlink(missing_ok=True)
    update_record(base, key, {'words': [], 'segments': [], 'pending': True, 'summary': None})
