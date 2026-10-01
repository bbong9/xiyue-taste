import argparse
import functools
import http.server
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .ask import AskError


class _Handler(http.server.SimpleHTTPRequestHandler):
    _status = None

    def __init__(self, *args, data=None, state=None, index=None, asker=None, butler=None, **kwargs):
        self._data = data
        self._state = state
        self._index = index
        self._asker = asker
        self._butler = butler
        super().__init__(*args, **kwargs)

    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def guess_type(self, path):
        if str(path).endswith(".json.gz"):
            return "application/json"
        return super().guess_type(path)

    def end_headers(self):
        if self._status == 200 and self.path.split("?", 1)[0].endswith(".json.gz"):
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()

    def log_message(self, format, *args):
        return

    def _panel_available(self):
        return self._data is not None and self._state is not None and self._index is not None

    def _send_json(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self, limit):
        try:
            length = int(self.headers.get("Content-Length"))
        except (TypeError, ValueError):
            self._send_json({"error": "too_large"}, 413)
            return None
        if length > limit:
            self._send_json({"error": "too_large"}, 413)
            return None
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Invalid object")
        except (ValueError, UnicodeDecodeError):
            error = "bad_query" if urlsplit(self.path).path == "/api/ask" else "bad_request"
            self._send_json({"error": error}, 400)
            return None
        return body

    @staticmethod
    def _integer(query, name, default):
        try:
            return int(query.get(name, [str(default)])[0])
        except ValueError:
            return default

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            if not self._panel_available():
                self.send_error(404)
                return
            payload = (Path(__file__).parent / "static" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif url.path.startswith("/api/"):
            if not self._panel_available():
                self._send_json({"error": "not_found"}, 404)
                return
            query = parse_qs(url.query, keep_blank_values=True)
            if url.path == "/api/status":
                self._send_json({
                    **self._state.snapshot(),
                    "trackCount": len(self._index.load()),
                    "failedCount": len(self._index.failures(self._data)),
                })
            elif url.path == "/api/stats":
                self._send_json(self._index.stats())
            elif url.path == "/api/tracks":
                self._send_json(self._index.tracks(
                    query.get("q", [""])[0],
                    self._integer(query, "offset", 0),
                    self._integer(query, "limit", 50),
                ))
            elif url.path == "/api/track":
                track = self._index.track(self._integer(query, "i", -1))
                if track is None:
                    self._send_json({"error": "not_found"}, 404)
                else:
                    self._send_json(track)
            elif url.path == "/api/failures":
                self._send_json(self._index.failures(self._data))
            elif url.path == "/api/ask/status":
                self._send_json(
                    self._asker.status() if self._asker is not None else {"configured": False, "model": ""}
                )
            else:
                self._send_json({"error": "not_found"}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            if path == "/api/scan" and self._panel_available():
                self._state.request_scan()
                self._send_json({"ok": True})
            elif path == "/api/ask" and self._panel_available():
                if self._asker is None:
                    self._send_json({"error": "ask_unconfigured"}, 503)
                    return
                body = self._read_json(4096)
                if body is None:
                    return
                q = body.get("q") if isinstance(body, dict) else None
                if not isinstance(q, str) or not q.strip() or len(q.strip()) > 100:
                    self._send_json({"error": "bad_query"}, 400)
                    return
                taste = body.get("taste", [])
                if not isinstance(taste, list) or len(taste) > 20 or not all(
                    isinstance(name, str) and 1 <= len(name) <= 40 for name in taste
                ):
                    self._send_json({"error": "bad_query"}, 400)
                    return
                try:
                    self._send_json(self._asker.ask(q.strip(), taste))
                except AskError as error:
                    self._send_json({"error": error.code}, error.status)
            elif path in ("/api/butler/artists", "/api/butler/songs") and self._panel_available():
                if self._butler is None:
                    self._send_json({"error": "ask_unconfigured"}, 503)
                    return
                body = self._read_json(262144)
                if body is None:
                    return
                if path == "/api/butler/artists":
                    items = body.get("artists")
                    valid = isinstance(items, list) and 1 <= len(items) <= 600 and all(
                        isinstance(item, dict)
                        and isinstance(item.get("name"), str) and 1 <= len(item["name"]) <= 100
                        and type(item.get("songs")) is int and item["songs"] >= 0
                        for item in items
                    )
                    method = self._butler.artists
                else:
                    items = body.get("songs")
                    valid = isinstance(items, list) and 1 <= len(items) <= 60 and all(
                        isinstance(item, dict)
                        and isinstance(item.get("id"), str) and bool(item["id"])
                        and all(isinstance(item.get(key), str) and len(item[key]) <= 500
                                for key in ("title", "album", "path"))
                        and isinstance(item.get("artists"), list)
                        and all(isinstance(name, str) for name in item["artists"])
                        for item in items
                    )
                    method = self._butler.songs
                if not valid:
                    self._send_json({"error": "bad_request"}, 400)
                    return
                try:
                    self._send_json(method(items))
                except AskError as error:
                    self._send_json({"error": error.code}, error.status)
            else:
                self._send_json({"error": "not_found"}, 404)
        else:
            self.send_error(501, "Unsupported method ('POST')")


def make_server(out, port, data=None, state=None, index=None, asker=None, butler=None):
    handler = functools.partial(
        _Handler, directory=str(out), data=data, state=state, index=index, asker=asker, butler=butler
    )
    return http.server.ThreadingHTTPServer(("", port), handler)


def main():
    parser = argparse.ArgumentParser(prog="python -m analyzer.serve")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    print(f"Serving {args.out} on port {args.port}.", flush=True)
    make_server(args.out, args.port).serve_forever()


if __name__ == "__main__":
    main()
