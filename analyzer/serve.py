import argparse
import functools
import http.server
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class _Handler(http.server.SimpleHTTPRequestHandler):
    _status = None

    def __init__(self, *args, data=None, state=None, index=None, **kwargs):
        self._data = data
        self._state = state
        self._index = index
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
            else:
                self._send_json({"error": "not_found"}, 404)
        else:
            self.send_error(501, "Unsupported method ('POST')")


def make_server(out, port, data=None, state=None, index=None):
    handler = functools.partial(
        _Handler, directory=str(out), data=data, state=state, index=index
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
