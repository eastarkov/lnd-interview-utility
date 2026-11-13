import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from core import extract_audio, timecode, parse_time, person_query, lexical_matches
from semantic import SemanticIndex, passages, summary_indices
from summarize import Summarizer
from library import (load_records, save_record, delete_transcript, atomic_json,
                     choose_videos, content_key, resolve_source, video_path)
from words import frequent_words
from player import editor, timeline
from montage import clip_id, validate, reordered, edited
from after_effects import exchange_path, unlinked_records, write_exchange

st.set_page_config(page_title='LND Interview Utility', page_icon=':material/movie:', layout='wide')
BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('VIDEO_WORDS_DATA_DIR', str(BASE / '.data'))).resolve()
DATA.mkdir(exist_ok=True)
settings_path = DATA / 'settings.json'
try:
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
except (OSError, ValueError):
    settings = {}
header, options = st.columns([5, 1.3])
with header:
    st.title('LND Interview Utility')
    st.caption('Видеотека → поиск по словам → монтаж для After Effects')
with options:
    with st.popover('Настройки'):
        appearance = st.selectbox('Тема приложения', ['Светлая', 'Тёмная'], index=1 if settings.get('theme')=='dark' else 0, key='appearance-v5')
        size = st.selectbox('Модель для новых записей', ['small', 'base', 'medium', 'large-v3'], index=0)
        language = st.selectbox('Язык речи', ['Авто', 'Русский', 'English'])
        pre = st.number_input('Запас до фразы, сек.', 0.0, 30.0, 0.4, 0.1)
        post = st.number_input('Запас после фразы, сек.', 0.0, 30.0, 0.6, 0.1)
        st.caption('Модели скачиваются при первом использовании. Видео и текст обрабатываются локально.')
# Stable preference, never inferred from a page whose CSS we change ourselves.
theme = 'dark' if appearance=='Тёмная' else 'light'
if settings.get('theme') != theme:
    settings['theme'] = theme
    atomic_json(settings_path, settings)
st.html(f'<span data-vw-theme="{theme}"></span><style>' + (BASE / 'styles.css').read_text() + '</style>')
LIBRARY = DATA / 'library'
LIBRARY.mkdir(exist_ok=True)
if 'records' not in st.session_state:
    st.session_state.records, st.session_state.library_errors = load_records(LIBRARY)
records = st.session_state.records
for error in st.session_state.get('library_errors', []):
    st.warning('Не удалось открыть запись: ' + error)

people_path = DATA / 'people.json'
try:
    saved_people = json.loads(people_path.read_text()) if people_path.exists() else {}
except (OSError, ValueError):
    saved_people = {}

@st.cache_resource(max_entries=1)
def model(size):
    return WhisperModel(size, device='cpu', compute_type='int8', download_root=str(DATA / 'whisper'))

@st.cache_resource
def semantic_model():
    return SemanticIndex(DATA / 'models')

@st.cache_resource
def summary_model():
    return Summarizer(DATA / "summary-model")

@st.cache_data(show_spinner=False)
def make_summary(texts):
    items = [{"text": text} for text in texts]
    vectors = embeddings(texts)
    return summary_model().summarize(items, vectors)

@st.cache_data(show_spinner=False)
def embeddings(texts):
    return semantic_model().encode(list(texts))

def record_vectors(key, name, items):
    texts = tuple(p['text'] for p in items)
    digest = hashlib.sha256(repr(texts).encode()).hexdigest()
    path = LIBRARY / key / (name + '.npz')
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as cache:
                if str(cache['digest']) == digest:
                    return cache['vectors']
        except (OSError, ValueError, KeyError):
            pass
    result = embeddings(texts)
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, digest=digest, vectors=result)
    temp.replace(path)
    return result


def recognize(key, record, recognition_size=None):
    selected_size = recognition_size or size
    try:
        with st.status(f"Распознавание: {record['name']}", expanded=True) as status:
            if not Path(record['audio']).exists():
                extract_audio(record['source'], record['audio'])
            segments, info = model(selected_size).transcribe(record['audio'], language={'Авто': None, 'Русский': 'ru', 'English': 'en'}[language], word_timestamps=True, vad_filter=True)
            result, words = [], []
            progress = st.progress(0)
            for segment in segments:
                result.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
                words.extend({'start': w.start, 'end': w.end, 'word': w.word} for w in (segment.words or []))
                progress.progress(min(0.99, segment.end / max(info.duration, 1)))
            delete_transcript(LIBRARY, key, record)
            record.update(segments=result, words=words, duration=info.duration, pending=False, recognition={'model':selected_size,'language':language})
            save_record(LIBRARY, key, record)
            progress.progress(1.0)
            status.update(label='Расшифровка сохранена', state='complete')
            return True
    except Exception as error:
        st.error(f"{record['name']}: {error}. Можно повторить распознавание кнопкой ниже.")
        return False

def attach_original(key, record, path):
    record.update(original_path=str(path), source=str(path))
    save_record(LIBRARY, key, record)


def add_videos(paths):
    for value in paths:
        try:
            source = video_path(value)
            with st.spinner(f'Проверяю файл: {source.name}'):
                digest = content_key(source)
            existing = next((k for k, r in records.items() if r.get('content_key', k) == digest), None)
            if existing:
                attach_original(existing, records[existing], source)
                st.info(f'{source.name}: оригинал привязан к существующей записи.')
                continue
            folder = LIBRARY / digest
            folder.mkdir(exist_ok=True)
            record = {'name':source.name, 'content_key':digest, 'original_path':str(source), 'source':str(source),
                      'audio':str(folder/'audio.wav'), 'segments':[], 'words':[], 'duration':0, 'pending':True}
            records[digest] = record
            save_record(LIBRARY, digest, record)
            recognize(digest, record)
        except (OSError, ValueError) as error:
            st.error(f'Не удалось добавить файл: {error}')


with st.expander('Добавить видео', expanded=not bool(records)):
    st.caption('Выберите оригиналы на этом Mac. Видеофайлы остаются на своих местах.')
    if st.button('Выбрать видео на Mac', type='primary'):
        try:
            add_videos(choose_videos())
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            st.error(str(error))
    with st.popover('Указать пути вручную'):
        paths = st.text_area('Пути к файлам, по одному на строку', placeholder='/Users/…/Видео.mov')
        if st.button('Добавить по путям', disabled=not paths.strip()):
            add_videos(line.strip() for line in paths.splitlines() if line.strip())

edits_path = DATA / 'clip-edits.json'
if 'clip-edits' not in st.session_state:
    try:
        st.session_state['clip-edits'] = json.loads(edits_path.read_text()) if edits_path.exists() else {}
    except (OSError, ValueError):
        st.session_state['clip-edits'] = {}
project_path = DATA / 'montage.json'
if 'montage-v5' not in st.session_state:
    try:
        st.session_state['montage-v5'] = json.loads(project_path.read_text()) if project_path.exists() else []
    except (OSError, ValueError):
        st.session_state['montage-v5'] = []
project = {item['id']: item for item in validate(st.session_state['montage-v5'], records)}
pending_action = st.session_state.pop('pending-montage-action', None)
if pending_action:
    previous = project.get(pending_action.get('id'))
    if previous:
        try:
            updated = edited(list(project.values()), pending_action, records)
            identity = previous['id']
            edit = st.session_state.setdefault('clip-edits', {}).setdefault(identity, {})
            if pending_action['type']=='remove':
                edit.update(start=timecode(previous['start']), end=timecode(previous['end']), selected=False)
                st.session_state['select-'+identity] = False
            else:
                item = next(i for i in updated if i['id']==identity)
                edit.update(start=timecode(item['start']), end=timecode(item['end']), selected=True)
                st.session_state['start-'+identity] = edit['start']
                st.session_state['end-'+identity] = edit['end']
            project = {item['id']:item for item in updated}
            st.session_state['montage-v5'] = updated
            atomic_json(project_path, updated)
        except (ValueError, KeyError, TypeError) as error:
            st.warning(f'Не удалось изменить фрагмент: {error}')


if not records:
    st.info('Добавьте видео, чтобы получить содержание и найти нужные фрагменты.')
    st.stop()

# Name fields stay visible; the detailed transcript is secondary.
st.subheader('Видеотека')
st.caption('Каждая карточка — отдельная запись: человек, содержание и расшифровка.')
for key, record in records.items():
    with st.container(border=True, key='library-card-'+key):
        st.subheader(record['name'], anchor=False)
        record.setdefault('content_key', key)
        record['source'] = resolve_source(LIBRARY / key, record)
        linked = bool(record.get('original_path') and Path(record['original_path']).is_file())
        if not linked:
            st.caption('Для After Effects привяжите оригинал на диске.')
        with st.expander('Управление записью'):
            original = record.get('original_path')
            if original:
                st.caption(('Оригинал: ' if linked else 'Оригинал перемещён или недоступен: ') + original)
            else:
                st.caption('В библиотеке есть копия. Укажите исходный файл, чтобы After Effects ссылался на него.')
            pick, manual = st.columns(2)
            selected_path = None
            if pick.button('Привязать оригинал', key='link-original-'+key):
                try:
                    selected = choose_videos(multiple=False)
                    selected_path = selected[0] if selected else None
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    st.error(str(error))
            with manual.popover('Путь к оригиналу'):
                value = st.text_input('Путь к файлу', key='original-path-'+key, placeholder='/Users/…/Видео.mov')
                if st.button('Привязать по пути', key='save-original-'+key, disabled=not value.strip()):
                    selected_path = value.strip()
            if selected_path:
                try:
                    path = video_path(selected_path)
                    with st.spinner('Проверяю, что это тот же исходник…'):
                        if content_key(path) != record['content_key']:
                            raise ValueError('Этот файл отличается от записи в библиотеке. Выберите тот же оригинал.')
                    attach_original(key, record, path)
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(str(error))
            if record.get('pending'):
                st.caption('Расшифровка удалена или ещё не создана. Исходное видео сохранено.')
                if st.button('Сгенерировать расшифровку', key='generate-'+key):
                    if recognize(key, record):
                        st.rerun()
            else:
                delete, regenerate = st.columns(2)
                if delete.button('Удалить расшифровку', key='delete-'+key):
                    delete_transcript(LIBRARY, key, record)
                    st.rerun()
                if regenerate.button('Перегенерировать', key='regenerate-'+key):
                    if recognize(key, record):
                        st.rerun()
        if record.get('pending'):
            continue
        left, right = st.columns([2, 3])
        left.caption(f"ДЛИТЕЛЬНОСТЬ · {timecode(record['duration'])}")
        person = right.text_input('Кто говорит?', value=saved_people.get(record['content_key'], record.get('person','')), placeholder='Например: Дамир', key=f'person-{key}')
        record['person'] = person.strip()
        if saved_people.get(record['content_key'], '') != person.strip():
            saved_people[record['content_key']] = person.strip()
            atomic_json(people_path, saved_people)
            save_record(LIBRARY, key, record)
        record['passages'] = passages(record['segments'])
        if record['passages']:
            try:
                with st.spinner('Выделяю основные темы… При первом запуске скачивается модель.'):
                    vectors = record_vectors(key, 'vectors', record['passages'])
                record['vectors'] = vectors
                with st.spinner('Составляю краткое содержание… Первая загрузка языковой модели — около 3 ГБ.'):
                    if not record.get('summary'):
                        record['summary'] = make_summary(tuple(p['text'] for p in record['passages']))
                        save_record(LIBRARY, key, record)
                    st.markdown('**О чём это видео**')
                    st.write(record['summary'])
                st.caption('Автоматическое краткое содержание; сверяйте важные детали с записью. На длинных видео используются ключевые фрагменты.')
            except Exception as error:
                st.warning(f'Не удалось сформировать содержание. Поиск по словам продолжает работать. {error}')
                if 'vectors' in record:
                    for index in summary_indices(record['vectors']):
                        passage = record['passages'][index]
                        st.write(f"{timecode(passage['start'])} — {passage['text']}")
                    st.caption('Резервный вариант: ключевые цитаты; связное содержание пока недоступно.')
                else:
                    st.caption('Краткое содержание пока недоступно; полная расшифровка ниже.')
        else:
            st.caption('Речь не найдена.')
        with st.expander('Полная расшифровка'):
            current_model = record.get('recognition', {}).get('model', 'small')
            more_accurate = 'large-v3' if current_model in ('medium','large-v3') else 'medium'
            st.caption(f'Модель записи: {current_model}')
            if st.button('Расшифровать точнее', key='improve-'+key, disabled=current_model=='large-v3', help=f'Повторное распознавание моделью {more_accurate}. Дольше, но обычно точнее. Сохранённый текст заменится после успешной обработки.'):
                if recognize(key, record, recognition_size=more_accurate):
                    st.rerun()
            for segment in record['segments']:
                st.write(f"{timecode(segment['start'])} — {segment['text']}")

st.divider()
st.subheader('Найти фрагменты')
def choose_word(word):
    st.session_state['search-query'] = word

@st.cache_data(show_spinner=False)
def popular(serialized):
    return frequent_words([{'words':json.loads(words)} for words in serialized])

active = [r for r in records.values() if not r.get('pending')]
if active:
    st.caption('Частые слова во всех готовых записях · формы слов объединены, предлоги и служебные слова исключены')
    common = popular(tuple(json.dumps(r['words'], ensure_ascii=False) for r in active))
    for offset in range(0, len(common), 6):
        columns = st.columns(6)
        for column, (word, count) in zip(columns, common[offset:offset+6]):
            column.button(f'{word} · {count}', key='popular-'+word, on_click=choose_word, args=(word,))
query = st.text_input('Человек, тема или варианты фразы', placeholder='дамир редизайн · или: редизайн, redesign', key='search-query')
people = sorted({r.get('person', '') for r in records.values()} - {''})
filters = st.multiselect('Искать у людей', people, placeholder='Все люди') if people else []
person, text_query = person_query(query, people)
if person:
    st.caption(f'Поиск только у: {person}')
st.caption('Запятая разделяет варианты «или». Поиск учитывает формы и родственные слова. Синонимы и переводы автоматически не добавляются.')
search_active = bool(query.strip() or filters)

matches = []
for key, record in records.items():
    if record.get('pending') or not search_active:
        continue
    if filters and record.get('person') not in filters:
        continue
    if person and record.get('person') != person:
        continue
    hits = lexical_matches(record, text_query) if text_query else [dict(p, kind='Фрагмент записи') for p in record['passages']]
    for hit in sorted(hits, key=lambda h: h['start']):
        identity = clip_id(key, hit['start'], hit['end'])
        hit['record_key'] = key
        matches.append((identity, record, hit))

if not matches and search_active:
    st.info('Совпадений не найдено. Попробуйте другой корень слова или добавьте варианты через запятую.')
if matches:
    st.caption(f'Найдено: {len(matches)}. Отмеченные фрагменты добавляются в общий монтаж, в том числе из разных поисков.')
    page = st.number_input('Страница результатов', min_value=1, max_value=max(1, math.ceil(len(matches)/5)), value=1, key='results-page-'+hashlib.sha256(repr((query, filters, len(matches))).encode()).hexdigest()[:12])
else:
    page = 1
visible = {m[0] for m in matches[(page-1)*5:page*5]}
invalid = False
for identity, record, hit in matches:
    default_start = timecode(max(0, hit['start']-pre))
    default_end = timecode(min(record['duration'], hit['end']+post))
    start_key, end_key, select_key = f'start-{identity}', f'end-{identity}', f'select-{identity}'
    # Separate permanent state preserves edits and selection when paging hides widgets.
    saved = st.session_state.setdefault('clip-edits', {})
    existing = project.get(identity)
    legacy = hashlib.sha256(f"{hit['record_key']}-{query}-{filters}-{pre}-{post}-{hit['start']}-{hit['end']}".encode()).hexdigest()[:20]
    if identity not in saved and legacy in saved:
        saved[identity] = saved[legacy].copy()
    edit = saved.setdefault(identity, {'start': timecode(existing['start']) if existing else default_start, 'end': timecode(existing['end']) if existing else default_end, 'selected': True})
    if identity in visible:
        for widget_key, field in [(start_key, 'start'), (end_key, 'end'), (select_key, 'selected')]:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = edit[field]
        with st.container(border=True, key='result-card-'+identity):
            st.caption(record['name'])
            edit['selected'] = st.checkbox(f"{record.get('person') or record['name']} · {timecode(hit['start'])} · {hit['kind']}", key=select_key)
            st.write(hit['text'])
            try:
                start, end = parse_time(edit['start']), parse_time(edit['end'])
                source = record['source'] if Path(record['source']).is_file() else None
                value = editor(source, start, end, record['duration'], key='editor-'+identity, theme=theme)
                if value and value.get('sequence') != edit.get('sequence'):
                    new_start, new_end = float(value['start']), float(value['end'])
                    if not 0 <= new_start < new_end <= record['duration'] + .001:
                        raise ValueError('Проверьте границы фрагмента.')
                    edit.update(start=timecode(new_start), end=timecode(new_end), sequence=value['sequence'])
            except Exception as error:
                st.error(f'Редактор: {error}')

    if edit['selected']:
        try:
            start, end = parse_time(edit['start']), parse_time(edit['end'])
            if not 0 <= start < end <= record['duration'] + .001:
                raise ValueError('Некорректные границы')
            project[identity] = {'id':identity, 'record_key':hit['record_key'], 'start':start,
                                 'end':min(end, record['duration']), 'text':hit['text']}
        except ValueError:
            invalid = True
    else:
        project.pop(identity, None)

st.divider()
items = validate(list(project.values()), records)
st.session_state['montage-v5'] = items
atomic_json(project_path, items)
atomic_json(edits_path, st.session_state.get('clip-edits', {}))

def clear_montage():
    st.session_state['montage-v5'] = []
    for identity, edit in st.session_state.get('clip-edits', {}).items():
        edit['selected'] = False
        if 'select-'+identity in st.session_state:
            st.session_state['select-'+identity] = False
    atomic_json(project_path, [])

with st.container(key='montage-dock'):
    if invalid:
        st.error('Исправьте неверные границы выбранных фрагментов.')
    if items:
        try:
            preview_data = []
            for item in items:
                record = records[item['record_key']]
                source = record['source']
                preview_data.append(dict(item, source=source if Path(source).is_file() else None,
                                         label=record.get('person') or record['name'],
                                         duration=item['end'] - item['start'],
                                         source_duration=record['duration']))
            action = timeline(preview_data, theme=theme)
            if action and action.get('sequence') != st.session_state.get('timeline-sequence'):
                st.session_state['timeline-sequence'] = action['sequence']
                if action.get('type') in ('trim', 'remove'):
                    st.session_state['pending-montage-action'] = action
                    st.rerun()
                new_items = reordered(items, action['order'])
                if [i['id'] for i in new_items] != [i['id'] for i in items]:
                    st.session_state['montage-v5'] = new_items
                    atomic_json(project_path, new_items)
                    st.rerun()
        except Exception as error:
            st.error(f'Не удалось показать таймлайн: {error}')
    else:
        st.info('Отметьте фрагменты в результатах поиска — они появятся здесь.')

    path = exchange_path(DATA)
    has_exchange = path.is_file()
    if items or has_exchange:
        missing = unlinked_records(items, records)
        if missing:
            labels = [records[key]['name'] for key in missing]
            note = ', '.join(labels[:2]) + (f' и ещё {len(labels)-2}' if len(labels) > 2 else '')
            st.caption(f'Привяжите оригиналы через «Управление записью»: {note}.')
        elif not items:
            st.caption('Если обновить файл сейчас, при следующем запуске в AE импортированные фрагменты будут убраны.')
        actions = st.columns([2.5, 2, 1.25])
        button_label = 'Обновить файл для AE' if has_exchange else 'Экспорт в After Effects'
        if actions[0].button(button_label, type='primary', disabled=invalid or bool(missing)):
            try:
                path = write_exchange(DATA, items, records)
                st.session_state['ae-notice'] = 'Файл обновлён.' if has_exchange else 'Файл готов.'
                if not has_exchange:
                    subprocess.run(['/usr/bin/open', '-R', str(path)], check=False, timeout=5)
                st.rerun()
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                st.error(f'Не удалось подготовить файл для After Effects: {error}')
        with actions[1].popover('Как открыть в AE'):
            st.write('В After Effects: **File → Scripts → Run Script File…** → **VideoWords.jsx**. После первого запуска сохраните проект как .aep.')
            st.write('Чтобы обновить монтаж, перезапишите файл здесь и запустите тот же скрипт в том же проекте AE. Для этого можно использовать **File → Scripts → Recent Script Files**.')
            if has_exchange:
                st.code(str(path), language=None)
                if st.button('Показать в Finder'):
                    try:
                        subprocess.run(['/usr/bin/open', '-R', str(path)], check=True, timeout=5)
                    except (OSError, subprocess.SubprocessError):
                        st.info('Скопируйте путь выше и вставьте его в диалог выбора файла.')
        if items:
            actions[2].button('Очистить', on_click=clear_montage, help='Снять все отметки монтажа, сохранив видеотеку.')
        notice = st.session_state.pop('ae-notice', None)
        if notice:
            st.toast(notice + ' Запустите VideoWords.jsx в After Effects.')
