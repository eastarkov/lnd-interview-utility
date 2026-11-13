"""Local background jobs. Workers write files and publish state without touching Streamlit."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel

from artifacts import artifact_lock
from core import extract_audio, ffmpeg
from library import update_record
from semantic import SemanticIndex, passages, summary_indices
from summarize import Summarizer


@lru_cache(maxsize=1)
def whisper_model(cache, size):
    return WhisperModel(size, device='cpu', compute_type='int8', download_root=str(cache))


@lru_cache(maxsize=1)
def summary_index(data):
    return SemanticIndex(data / 'models')


@lru_cache(maxsize=1)
def summary_model(data):
    return Summarizer(data / 'summary-model')


class Processing:
    def __init__(self, data):
        self.data = Path(data)
        self.library = self.data / 'library'
        self._inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lnd-inference')
        self._frames = ThreadPoolExecutor(max_workers=2, thread_name_prefix='lnd-frame')
        self._states = {}
        self._lock = Lock()

    def snapshot(self):
        with self._lock:
            return self._states.copy()

    def clear(self, kind, key):
        with self._lock:
            job = (kind, key)
            if self._states.get(job, ('',))[0] not in ('queued', 'running'):
                self._states.pop(job, None)

    def _submit(self, kind, key, work, *args):
        job = (kind, key)
        with self._lock:
            if self._states.get(job, ('',))[0] in ('queued', 'running'):
                return
            self._states[job] = ('queued', 0, '')
        executor = self._frames if kind == 'frame' else self._inference
        executor.submit(self._run, job, work, args)

    def _run(self, job, work, args):
        def progress(value):
            with self._lock:
                self._states[job] = ('running', value, '')
        progress(0)
        try:
            work(*args, progress)
            result = ('done', 100, '')
        except Exception as error:
            with self._lock:
                last_progress = self._states[job][1]
            result = ('error', last_progress, str(error).strip() or type(error).__name__)
        with self._lock:
            self._states[job] = result

    def transcribe(self, key, record, size, language):
        self._submit('transcript', key, self._transcribe, key, record.copy(), size, language)

    def _transcribe(self, key, record, size, language, progress):
        audio = Path(record['audio'])
        with artifact_lock(audio):
            if not audio.is_file():
                temp = audio.with_name('extracting.wav')
                try:
                    extract_audio(record['source'], temp)
                    temp.replace(audio)
                finally:
                    temp.unlink(missing_ok=True)
        model = whisper_model(self.data / 'whisper', size)
        segments, info = model.transcribe(
            str(audio), language={'Авто': None, 'Русский': 'ru', 'English': 'en'}[language],
            word_timestamps=True, vad_filter=True)
        result, words = [], []
        for segment in segments:
            result.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
            words.extend({'start': w.start, 'end': w.end, 'word': w.word} for w in (segment.words or []))
            progress(min(99, int(100 * segment.end / max(info.duration, 1))))
        update_record(self.library, key, {
            'segments': result, 'words': words, 'duration': info.duration,
            'pending': False, 'summary': None, 'recognition': {'model': size, 'language': language},
        })
        self.clear('summary', key)

    def summarize(self, key, segments):
        self._submit('summary', key, self._summarize, key, segments)

    def _summarize(self, key, segments, _progress):
        parts = passages(segments)
        vectors = summary_index(self.data).encode([part['text'] for part in parts])
        try:
            summary = summary_model(self.data).summarize(parts, vectors)
        except Exception:
            summary = 'Ключевые цитаты:\n\n' + '\n\n'.join(parts[i]['text'] for i in summary_indices(vectors))
        update_record(self.library, key, {'summary': summary}, expected_segments=segments)

    def thumbnail(self, key, source):
        self._submit('frame', key, self._thumbnail, key, source)

    def _thumbnail(self, key, source, _progress):
        target = self.library / key / 'thumbnail.jpg'
        with artifact_lock(target):
            if target.is_file():
                return
            temp = target.with_name('thumbnail-working.jpg')
            try:
                args = ['-i', source, '-frames:v', 1, '-vf',
                        'scale=256:144:force_original_aspect_ratio=decrease', '-q:v', 4, temp]
                ffmpeg(['-ss', 1, *args])
                if not temp.is_file():
                    ffmpeg(args)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
