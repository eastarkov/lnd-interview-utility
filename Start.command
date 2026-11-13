#!/bin/bash
set -e
export HF_HUB_DISABLE_XET=1
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Установите Python 3.11 или 3.12 с https://www.python.org/downloads/macos/'
  read -r -p 'Нажмите Enter для выхода'
  exit 1
fi
if [ ! -d .venv ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt; then
  echo 'Установка не удалась. Проверьте интернет и используйте Python 3.11 или 3.12.'
  read -r -p 'Нажмите Enter для выхода'
  exit 1
fi
video_words_theme=$(.venv/bin/python - <<'PYMODE'
import json, os
from pathlib import Path
try:
    prefs=json.loads((Path(os.environ.get("VIDEO_WORDS_DATA_DIR", ".data"))/"settings.json").read_text())
except (OSError, ValueError):
    prefs={}
print("dark" if prefs.get("theme")=="dark" else "light")
PYMODE
)
.venv/bin/python -m streamlit run app.py --theme.base "$video_words_theme" --server.address 127.0.0.1 --browser.gatherUsageStats false
