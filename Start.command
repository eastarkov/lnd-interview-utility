#!/bin/bash
set -e
export HF_HUB_DISABLE_XET=1
cd "$(dirname "$0")"

fail() {
  printf '%s\n' "$1" >&2
  if [ -t 0 ]; then read -r -p 'Нажмите Enter для выхода' || true; fi
  exit 1
}

mac_arch="$(/usr/bin/uname -m)"
if [ "$(/usr/sbin/sysctl -in sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then
  mac_arch=arm64
fi
runtime_dir="$PWD/.runtime"
managed_python="$runtime_dir/python/bin/python3"

usable_python() {
  [ -n "$1" ] && [ "$1" != /usr/bin/python3 ] && [ -x "$1" ] &&
    "$1" -I -c 'import platform, ssl, sys, venv
sys.exit(not ((3, int(sys.argv[1])) <= sys.version_info[:2] <= (3, 12)
              and platform.machine() == sys.argv[2]))' "$2" "$mac_arch" >/dev/null 2>&1
}

install_python() (
  # Pinned archives and SHA-256 hashes from the Astral 20261110 release.
  case "$mac_arch" in
    arm64)
      target=aarch64-apple-darwin
      checksum=2f11b8aa3726bdbd87dd9eb76568412faddf4f47e26913fa45d6a2e424972acd
      ;;
    x86_64)
      target=x86_64-apple-darwin
      checksum=b78a4c479f1964ec484880bee567568266f68da102269fb93a05fe8716b45f91
      ;;
    *) fail 'Для автоматической установки нужен Mac с процессором Apple или Intel.' ;;
  esac
  mkdir -p "$runtime_dir" || exit 1
  staging="$(mktemp -d "$runtime_dir/download.XXXXXX")" || exit 1
  trap 'rm -rf "$staging"' EXIT
  trap 'exit 1' HUP INT TERM
  archive="cpython-3.12.16%2B20261110-$target-install_only_stripped.tar.gz"
  url="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/$archive"
  echo 'Скачиваю Python 3.12 для приложения…'
  /usr/bin/curl --fail --location --proto '=https' --retry 2 --connect-timeout 15 \
    --output "$staging/python.tar.gz" "$url" || exit 1
  (cd "$staging" && printf '%s  python.tar.gz\n' "$checksum" | /usr/bin/shasum -a 256 -c -) || exit 1
  /usr/bin/tar -xzf "$staging/python.tar.gz" -C "$staging" || exit 1
  usable_python "$staging/python/bin/python3" 11 || exit 1
  if [ -e "$runtime_dir/python" ]; then
    mv "$runtime_dir/python" "$staging/previous-python" || exit 1
  fi
  mv "$staging/python" "$runtime_dir/"
)

venv_dir="$PWD/.venv"
if ! usable_python "$venv_dir/bin/python" 9; then
  if [ -e "$venv_dir" ]; then
    venv_dir="$runtime_dir/venv"
  fi
  if ! usable_python "$venv_dir/bin/python" 11; then
    base_python=
    for candidate in "$managed_python" \
      "$(type -P python3.12 || true)" "$(type -P python3.11 || true)" "$(type -P python3 || true)" \
      /opt/homebrew/bin/python3.{12,11} /usr/local/bin/python3.{12,11} \
      /Library/Frameworks/Python.framework/Versions/3.{12,11}/bin/python3; do
      if usable_python "$candidate" 11; then
        base_python="$candidate"
        break
      fi
    done
    if [ -z "$base_python" ]; then
      install_python || fail 'Не удалось загрузить или подготовить Python. Проверьте доступ к github.com и запустите Start.command ещё раз.'
      base_python="$managed_python"
    fi
    "$base_python" -I -m venv "$venv_dir" || fail 'Не удалось создать окружение Python. Проверьте, что папка приложения доступна для записи.'
  fi
fi

python="$venv_dir/bin/python"
if ! "$python" -m pip install --disable-pip-version-check -r requirements.txt; then
  fail 'Не удалось установить зависимости. Проверьте интернет и запустите Start.command ещё раз.'
fi
video_words_theme=$("$python" - <<'PYMODE'
import json, os
from pathlib import Path
try:
    prefs=json.loads((Path(os.environ.get("VIDEO_WORDS_DATA_DIR", ".data"))/"settings.json").read_text())
except (OSError, ValueError):
    prefs={}
print("dark" if prefs.get("theme")=="dark" else "light")
PYMODE
)
exec "$python" -m streamlit run app.py --theme.base "$video_words_theme" --server.address 127.0.0.1 --browser.gatherUsageStats false
