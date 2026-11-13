"""Seekable, read-only access to videos explicitly registered by the app."""
import mimetypes
import os
import re
import secrets
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


def _range(value, size):
    if not value:
        return 0, size
    match = re.fullmatch(r'bytes=(\d{0,20})-(\d{0,20})', value.strip())
    if not match or not size:
        raise ValueError('Invalid byte range')
    first, last = match.groups()
    if first:
        start, end = int(first), min(size, int(last) + 1) if last else size
    elif last:
        start, end = max(0, size - int(last)), size
    else:
        raise ValueError('Empty byte range')
    if not 0 <= start < end <= size:
        raise ValueError('Unsatisfiable byte range')
    return start, end


class _MediaHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self._serve(False)

    def do_GET(self):
        self._serve(True)

    def _serve(self, body):
        addresses = (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')
        if (self.headers.get('Host') not in addresses
                or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
            self.send_error(403)
            return
        path = self.server.sources.get(self.path)
        if path is None:
            self.send_error(404)
            return
        try:
            source = path.open('rb')
        except OSError:
            self.send_error(404)
            return
        with source:
            size = os.fstat(source.fileno()).st_size
            requested = self.headers.get('Range')
            try:
                start, end = _range(requested, size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            mime = ('video/mp4' if path.suffix.lower() in ('.mov', '.m4v')
                    else mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type', mime)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end-start))
            self.send_header('Cache-Control', 'private, no-store')
            origin = self.headers.get('Origin', '')
            if re.fullmatch(r'http://(?:localhost|127\.0\.0\.1):\d{1,5}', origin):
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end-1}/{size}')
            self.end_headers()
            if body:
                try:
                    source.seek(start)
                    while start < end:
                        chunk = source.read(min(256 * 1024, end-start))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        start += len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass


@lru_cache(maxsize=1)
def _server():
    server = ThreadingHTTPServer(('127.0.0.1', 0), _MediaHandler)
    server.sources = {}
    Thread(target=server.serve_forever, name='VideoWords media', daemon=True).start()
    return server


@lru_cache(maxsize=1024)
def _url(path):
    server = _server()
    route = '/video/' + secrets.token_urlsafe(24)
    server.sources[route] = path
    return f':{server.server_port}{route}'


def media_url(source):
    return _url(Path(source).resolve(strict=True))
