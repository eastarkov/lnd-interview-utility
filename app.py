import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import hashlib
from html import escape
import json
import math
import subprocess
from pathlib import Path
from uuid import uuid4
import streamlit as st
from core import timecode, parse_time, person_query, lexical_matches, video_duration
from semantic import passages
from processing import DEFAULT_MODEL, MODELS, JobState, Processing
from library import (load_records, save_record, delete_transcript, atomic_json,
                     choose_videos, content_key, update_record, video_path)
from words import frequent_words
from player import editor, timeline
from montage import clip_bounds, clip_id, source_duration, validate, reordered, edited
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
model_choices = list(MODELS)
saved_model = settings.get('transcription_model', DEFAULT_MODEL)
if saved_model not in model_choices:
    saved_model = DEFAULT_MODEL
header, options = st.columns([5, 1.3])
with header:
    st.title('LND Interview Utility')
    st.caption('Ручной монтаж и поиск по словам для After Effects')
with options:
    with st.popover('Настройки'):
        appearance = st.selectbox('Тема приложения', ['Светлая', 'Тёмная'], index=1 if settings.get('theme')=='dark' else 0, key='appearance-v5')
        size = st.selectbox('Модель распознавания', model_choices, index=model_choices.index(saved_model),
                            format_func=MODELS.__getitem__, key='transcription-model',
                            help='По умолчанию Whisper Large V3 Turbo. Для более быстрой обработки выберите Small.')
        language = st.selectbox('Язык речи', ['Авто', 'Русский', 'English'])
        pre = st.number_input('Запас до фразы, сек.', 0.0, 30.0, 0.4, 0.1)
        post = st.number_input('Запас после фразы, сек.', 0.0, 30.0, 0.6, 0.1)
        st.caption('Модели скачиваются при первом использовании. Видео и текст обрабатываются локально.')
# Stable preference, never inferred from a page whose CSS we change ourselves.
theme = 'dark' if appearance=='Тёмная' else 'light'
if settings.get('theme') != theme or settings.get('transcription_model') != size:
    settings.update(theme=theme, transcription_model=size)
    atomic_json(settings_path, settings)
st.html(f'<span data-vw-theme="{theme}"></span><style>' + (BASE / 'styles.css').read_text() + '</style>')
LIBRARY = DATA / 'library'
LIBRARY.mkdir(exist_ok=True)
@st.cache_resource
def background(data):
    return Processing(data)


processing = background(DATA)
# Read the worker state first so a just-finished write can trigger a refresh.
jobs_at_load = processing.snapshot()
records, library_errors = load_records(LIBRARY)
for error in library_errors:
    st.warning('Не удалось открыть запись: ' + error)

people_path = DATA / 'people.json'
try:
    saved_people = json.loads(people_path.read_text()) if people_path.exists() else {}
except (OSError, ValueError):
    saved_people = {}


def attach_original(key, record, path):
    record.update(original_path=str(path), source=str(path))
    update_record(LIBRARY, key, {'original_path': str(path)})
    processing.clear('frame', key)


def add_videos(paths, transcribe=True):
    added = []
    errors = []
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
            duration = video_duration(source)
            folder = LIBRARY / digest
            folder.mkdir(exist_ok=True)
            record = {'name':source.name, 'content_key':digest, 'original_path':str(source), 'source':str(source),
                      'audio':str(folder/'audio.wav'), 'segments':[], 'words':[], 'duration':0,
                      'media_duration': duration, 'pending':True}
            records[digest] = record
            save_record(LIBRARY, digest, record)
            added.append((digest, record))
        except (OSError, ValueError) as error:
            errors.append(f'Не удалось добавить файл: {error}')
    if transcribe:
        for key, record in added:
            processing.transcribe(key, record, size, language)
    if added:
        if not transcribe:
            st.session_state['last-manual-source'] = added[0][0]
            st.session_state.pop('manual-source', None)
        st.session_state['import-errors'] = errors
        st.rerun()
    for error in errors:
        st.error(error)


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
                edit.update(start=timecode(previous['start']), end=timecode(previous['end']))
            else:
                item = next(i for i in updated if i['id']==identity)
                edit.update(start=timecode(item['start']), end=timecode(item['end']))
            project = {item['id']:item for item in updated}
            st.session_state['montage-v5'] = updated
            atomic_json(project_path, updated)
        except (ValueError, KeyError, TypeError) as error:
            st.warning(f'Не удалось изменить фрагмент: {error}')


for key, record in records.items():
    record.setdefault('content_key', key)
    record['person'] = saved_people.get(record['content_key'], record.get('person', ''))
    record['passages'] = passages(record['segments']) if not record.get('pending') else []


def manage_record(key, record, busy):
    linked = bool(record.get('original_path') and Path(record['original_path']).is_file())
    with st.expander('Управление записью'):
        original = record.get('original_path')
        if original:
            st.caption(('Оригинал: ' if linked else 'Оригинал перемещён или недоступен: ') + original)
        else:
            st.caption('В библиотеке есть копия. Укажите исходный файл, чтобы After Effects ссылался на него.')
        selected_path = None
        if st.button('Привязать оригинал', key='link-original-'+key):
            try:
                selected = choose_videos(multiple=False)
                selected_path = selected[0] if selected else None
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                st.error(str(error))
        value = st.text_input('Путь к оригиналу', key='original-path-'+key, placeholder='/Users/…/Видео.mov')
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
            if st.button('Сгенерировать расшифровку', key='generate-'+key, disabled=busy):
                processing.transcribe(key, record, size, language)
                st.rerun()
        else:
            delete, regenerate = st.columns(2)
            if delete.button('Удалить расшифровку', key='delete-'+key, disabled=busy):
                delete_transcript(LIBRARY, key, record)
                processing.clear('transcript', key)
                processing.clear('summary', key)
                st.rerun()
            if regenerate.button('Перегенерировать', key='regenerate-'+key, disabled=busy):
                processing.transcribe(key, record, size, language)
                st.rerun()


def render_details(key, record, busy, summary_state, transcript_error, frame_error):
    with st.popover('Подробнее', help=f"Запись {record['name']}"):
        if transcript_error:
            st.error(transcript_error)
        if frame_error:
            st.caption('Кадр недоступен. Проверьте исходный файл.')
        manage_record(key, record, busy)
        if record.get('pending'):
            st.caption('Полная расшифровка появится после обработки.')
            return
        st.caption(f"Длительность · {timecode(record['duration'])}")
        st.markdown('**Краткое содержание**')
        if record.get('summary'):
            st.write(record['summary'])
        elif not record['passages']:
            st.caption('Речь не найдена.')
        elif summary_state == 'error':
            st.caption('Краткое содержание не удалось подготовить.')
            if st.button('Повторить', key='summary-'+key, disabled=busy):
                processing.summarize(key, record['segments'])
        else:
            st.caption('Содержание готовится.')
        with st.expander('Полная расшифровка'):
            current_model = record.get('recognition', {}).get('model', 'small')
            more_accurate = 'large-v3' if current_model == DEFAULT_MODEL else DEFAULT_MODEL
            st.caption(f'Модель записи: {MODELS.get(current_model, current_model)}')
            if st.button('Расшифровать точнее', key='improve-'+key,
                         disabled=busy or current_model == 'large-v3',
                         help=f'Повторить распознавание с {MODELS[more_accurate]}.'):
                processing.transcribe(key, record, more_accurate, language)
                st.rerun()
            for segment in record['segments']:
                st.write(f"{timecode(segment['start'])} — {segment['text']}")


def render_progress(key, record):
    jobs = processing.snapshot()
    job = jobs.get(('transcript', key), JobState())
    percent = job.percent
    if job.state == 'queued':
        label = 'В очереди'
    elif job.state == 'running':
        label = f'{job.stage} · {percent}%' if percent is not None else job.stage or 'Подготовка…'
    elif job.state == 'error':
        label = 'Ошибка'
    elif record.get('pending') and job.state != 'done':
        label = 'Без расшифровки'
    else:
        label = 'Готово'
        percent = 100
        if jobs.get(('summary', key), JobState()).state in ('queued', 'running'):
            label = 'Готово · пересказ…'
    name = escape(record['name'], quote=True)
    value = f' value="{percent}"' if percent is not None else ''
    st.html(f'<div class="vw-file-progress"><span class="vw-file-name">{name}</span>'
            f'<progress{value} max="100" aria-label="Расшифровка {name}"></progress>'
            f'<span role="status" aria-label="{name}: {label}">{label}</span></div>')


def render_library():
    jobs = processing.snapshot()
    if finished_jobs(jobs) != finished_jobs(jobs_at_load):
        st.rerun()
    for key, record in sorted(records.items(), key=lambda item: item[1]['name'].casefold()):
        transcript = jobs.get(('transcript', key), JobState())
        summary_state = jobs.get(('summary', key), JobState()).state
        frame_state = jobs.get(('frame', key), JobState()).state
        transcribing = transcript.state in ('queued', 'running')
        if not transcribing and not summary_state and record['passages'] and not record.get('summary'):
            processing.summarize(key, record['segments'])
            summary_state = 'queued'
        thumbnail = LIBRARY / key / 'thumbnail.jpg'
        if not thumbnail.is_file() and not frame_state and Path(record['source']).is_file():
            processing.thumbnail(key, record['source'])

        with st.container(border=True, key='record-row-'+key, gap='small'):
            preview, file, speaker, details = st.columns([1, 4.5, 2.2, 1.35], gap='small',
                                                        vertical_alignment='center')
            with preview:
                if thumbnail.is_file():
                    st.image(str(thumbnail), width=112)
                else:
                    st.html('<div class="vw-frame-placeholder" aria-label="Кадр недоступен"></div>')
            with file:
                render_progress(key, record)
            with speaker:
                person = st.text_input(f"Кто говорит? {record['name']}", value=record['person'],
                                       placeholder='Кто говорит?', label_visibility='collapsed',
                                       key=f'person-{key}').strip()
                if person != record['person']:
                    saved_people[record['content_key']] = person
                    atomic_json(people_path, saved_people)
                    update_record(LIBRARY, key, {'person': person})
                    st.rerun()
            with details:
                render_details(key, record, transcribing, summary_state, transcript.error, frame_state == 'error')


def finished_jobs(states):
    return {job: status.state for job, status in states.items() if status.state in ('done', 'error')}


def choose_word(word):
    st.session_state['search-query'] = word
    st.session_state['last-query'] = word

@st.cache_data(show_spinner=False)
def popular(serialized):
    return frequent_words([{'words':json.loads(words)} for words in serialized])


def video_import(widget_prefix, transcribe):
    with st.expander('Добавить видео', expanded=not bool(records)):
        st.caption('Выберите оригиналы на этом Mac. Видеофайлы остаются на своих местах.')
        if st.button('Выбрать видео на Mac', type='primary', key=widget_prefix+'-choose'):
            try:
                add_videos(choose_videos(), transcribe)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                st.error(str(error))
        with st.popover('Указать пути вручную'):
            paths = st.text_area('Пути к файлам, по одному на строку', placeholder='/Users/…/Видео.mov',
                                 key=widget_prefix+'-paths')
            if st.button('Добавить по путям', disabled=not paths.strip(), key=widget_prefix+'-submit'):
                add_videos((line.strip() for line in paths.splitlines() if line.strip()), transcribe)


def render_manual():
    video_import('manual-import', transcribe=False)
    available = [key for key, record in records.items() if Path(record['source']).is_file()]
    if not available:
        if records:
            st.info('Привяжите исходный файл в видеотеке, чтобы открыть его для монтажа.')
        return False
    available.sort(key=lambda k: records[k]['name'].casefold())
    selected = st.session_state.get('last-manual-source')
    index = available.index(selected) if selected in available else 0
    key = st.selectbox('Исходное видео', available, index=index,
                       format_func=lambda k: records[k]['name'], key='manual-source')
    st.session_state['last-manual-source'] = key
    record = records[key]
    try:
        if not record.get('media_duration'):
            record['media_duration'] = video_duration(record['source'])
            update_record(LIBRARY, key, {'media_duration': record['media_duration']})
    except (OSError, ValueError) as error:
        st.error(str(error))
        return False

    duration = record['media_duration']
    acknowledgements = st.session_state.setdefault('manual-ack', {})
    ack = acknowledgements.get(key)
    value = editor(record['source'], 0, duration, duration, key='manual-editor-'+key,
                   theme=theme, manual=True, ack=ack)
    if not isinstance(value, dict) or value.get('type') != 'add':
        return False
    sequence = value.get('sequence')
    if type(sequence) is not int or (ack and sequence <= ack['sequence']):
        return False
    try:
        start, end = clip_bounds(value['start'], value['end'], duration)
    except (KeyError, TypeError, ValueError):
        acknowledgements[key] = {'sequence': sequence, 'ok': False}
        return True
    identity = 'manual-' + uuid4().hex
    project[identity] = {'id': identity, 'record_key': key, 'start': start, 'end': end, 'text': ''}
    acknowledgements[key] = {'sequence': sequence, 'ok': True}
    return True


def render_search():
    if not any(not r.get('pending') for r in records.values()):
        st.info('Добавьте видео в видеотеку. Поиск станет доступен после первой расшифровки.')
        return False, False
    active = [r for r in records.values() if not r.get('pending')]
    if active:
        st.caption('Частые слова во всех готовых записях · формы слов объединены, предлоги и служебные слова исключены')
        common = popular(tuple(json.dumps(r['words'], ensure_ascii=False) for r in active))
        for offset in range(0, len(common), 6):
            columns = st.columns(6)
            for column, (word, count) in zip(columns, common[offset:offset+6]):
                column.button(f'{word} · {count}', key='popular-'+word, on_click=choose_word, args=(word,))
    if 'search-query' not in st.session_state:
        st.session_state['search-query'] = st.session_state.get('last-query', '')
    query = st.text_input('Человек, тема или варианты фразы',
                          placeholder='дамир редизайн · или: редизайн, redesign', key='search-query')
    st.session_state['last-query'] = query
    people = sorted({r.get('person', '') for r in records.values()} - {''})
    saved_filters = [person for person in st.session_state.get('last-filters', []) if person in people]
    if 'search-filters' not in st.session_state:
        st.session_state['search-filters'] = saved_filters
    filters = st.multiselect('Искать у людей', people,
                             placeholder='Все люди', key='search-filters') if people else []
    st.session_state['last-filters'] = filters
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
        st.caption(f'Найдено: {len(matches)}')
        page = st.number_input('Страница результатов', min_value=1, max_value=max(1, math.ceil(len(matches)/5)), value=1, key='results-page-'+hashlib.sha256(repr((query, filters, len(matches))).encode()).hexdigest()[:12])
    else:
        page = 1
    saved = st.session_state['clip-edits']
    invalid = False
    changed_selection = False
    for identity, record, hit in matches[(page-1)*5:page*5]:
        existing = project.get(identity)
        duration = source_duration(record)
        default_start = timecode(max(0, hit['start'] - pre))
        default_end = timecode(min(duration, hit['end'] + post))
        edit = saved.setdefault(identity, {
            'start': timecode(existing['start']) if existing else default_start,
            'end': timecode(existing['end']) if existing else default_end,
        })
        with st.container(border=True, key='result-card-'+identity):
            heading, action = st.columns([3, 1.4], gap='small', vertical_alignment='center')
            heading.caption(f"{record.get('person') or record['name']} · {timecode(hit['start'])} · {hit['kind']}")
            add = action.button('Убрать из монтажа' if existing else 'Добавить в монтаж',
                                key='montage-toggle-'+identity, type='secondary' if existing else 'primary')
            st.write(hit['text'])
            try:
                start, end = parse_time(edit['start']), parse_time(edit['end'])
                source = record['source'] if Path(record['source']).is_file() else None
                value = editor(source, start, end, duration, key='editor-'+identity, theme=theme)
                if value and value.get('sequence') != edit.get('sequence'):
                    new_start, new_end = clip_bounds(value['start'], value['end'], duration)
                    edit.update(start=timecode(new_start), end=timecode(new_end), sequence=value['sequence'])
            except Exception as error:
                st.error(f'Редактор: {error}')

            if existing and add:
                project.pop(identity)
                changed_selection = True
            elif existing or add:
                try:
                    start, end = clip_bounds(parse_time(edit['start']), parse_time(edit['end']), duration)
                    project[identity] = {'id': identity, 'record_key': hit['record_key'],
                                         'start': start, 'end': end, 'text': hit['text']}
                    changed_selection |= add
                except ValueError as error:
                    st.error(str(error))
                    invalid |= bool(existing)
    return invalid, changed_selection

for error in st.session_state.pop('import-errors', []):
    st.error(error)

with st.container(key='app-navigation'):
    page = st.segmented_control('Раздел', ['Видеотека', 'Ручной монтаж', 'Поиск'],
                                default='Видеотека', label_visibility='collapsed',
                                key='app-navigation-choice') or 'Видеотека'
invalid = changed_selection = manual_changed = False
if page == 'Видеотека':
    polling = any(status.state in ('queued', 'running') for status in jobs_at_load.values())
    for key, record in records.items():
        needs_frame = (('frame', key) not in jobs_at_load and Path(record['source']).is_file()
                       and not (LIBRARY / key / 'thumbnail.jpg').is_file())
        needs_summary = (('summary', key) not in jobs_at_load and record['passages']
                         and not record.get('summary'))
        polling = polling or bool(needs_frame or needs_summary)
    # Keep the fragment at a fixed position even when the queue finishes.
    st.fragment(render_library, run_every=1 if polling else None)()
    video_import('library-import', transcribe=True)

elif page == 'Ручной монтаж':
    manual_changed = render_manual()

else:
    @st.fragment(run_every=1 if any(kind == 'transcript' and job.state in ('queued', 'running')
                                    for (kind, _), job in jobs_at_load.items()) else None)
    def refresh_search():
        def transcripts(states):
            return {key: status for key, status in finished_jobs(states).items() if key[0] == 'transcript'}
        if transcripts(processing.snapshot()) != transcripts(jobs_at_load):
            st.rerun()

    refresh_search()
    invalid, changed_selection = render_search()

st.divider()
items = validate(list(project.values()), records)
st.session_state['montage-v5'] = items
atomic_json(project_path, items)
atomic_json(edits_path, st.session_state['clip-edits'])
if changed_selection or manual_changed:
    st.rerun()

def clear_montage():
    st.session_state['montage-v5'] = []
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
                                         source_duration=source_duration(record)))
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
        st.info('Добавьте фрагменты во вкладке «Ручной монтаж» или «Поиск».')

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
        include_subtitles = st.checkbox('Субтитры для After Effects',
                                        value=bool(settings.get('ae_subtitles', False)),
                                        key='ae-subtitles',
                                        help='Добавляет редактируемый текст по таймкодам расшифровки. Снимите флажок и обновите файл, чтобы убрать эту дорожку из AE.')
        if include_subtitles != settings.get('ae_subtitles', False):
            settings['ae_subtitles'] = include_subtitles
            atomic_json(settings_path, settings)
        actions = st.columns([2.5, 2, 1.25])
        button_label = 'Обновить файл для AE' if has_exchange else 'Экспорт в After Effects'
        if actions[0].button(button_label, type='primary', disabled=invalid or bool(missing)):
            try:
                path = write_exchange(DATA, items, records, include_subtitles=include_subtitles)
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
            actions[2].button('Очистить', on_click=clear_montage, help='Убрать все фрагменты из монтажа.')
        notice = st.session_state.pop('ae-notice', None)
        if notice:
            st.toast(notice + ' Запустите VideoWords.jsx в After Effects.')
