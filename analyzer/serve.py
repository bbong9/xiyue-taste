import argparse
import functools
import http.server
import ipaddress
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import log
from .access import AccessError, AccessSettings
from .accounts import AccountError
from .ask import ASK_PARTS, AskError
from .downloads import DownloadError
from .log import LOGGER
from .personal import PersonalError
from .resolver import FAMILY_DEFAULTS, OWNER, ResolveError
from .settings import SettingsError
from .sources import SourceError


_SOURCE_ITEM = re.compile(r"/api/sources/([0-9a-f]{16})(/test|/replace|/refresh|/rollback)?")
_ACCOUNT_ITEM = re.compile(r"/api/accounts/(owner|[0-9a-f]{16})(?:/devices/([0-9a-f]{16}))?")
_DISABLED = object()
_STREAM_PREFIX = "/api/source/stream/"
_auth_rejections = {}
_auth_rejections_lock = threading.Lock()


def _log_auth_reject(address, path, headers):
    now = time.monotonic()
    with _auth_rejections_lock:
        previous = _auth_rejections.get(address)
        if previous is not None and now - previous < 60:
            return
        _auth_rejections[address] = now
        if len(_auth_rejections) > 1000:
            _auth_rejections.clear()
            _auth_rejections[address] = now
    shown = urlsplit(path).path
    if unquote(shown).startswith(_STREAM_PREFIX.rstrip("/")):
        # A relay path carries its ticket.
        shown = _STREAM_PREFIX + "-"
    LOGGER.warning(
        "AUTH-REJECT addr=%s path=%s forwarded=%s token=%s",
        address, shown,
        "是" if any(header in headers for header in ("X-Forwarded-For", "X-Real-IP", "Forwarded")) else "否",
        "带了" if headers.get("Authorization") else "没带",
    )


class _Handler(http.server.SimpleHTTPRequestHandler):
    _status = None
    _identity = None

    def __init__(
        self, *args, data=None, state=None, index=None, asker=None, butler=None, llm=None, downloads=None,
        access=None, sources=None, resolver=None, accounts=None, personal=None, **kwargs,
    ):
        self._data = data
        self._state = state
        self._index = index
        self._asker = asker
        self._butler = butler
        self._llm = llm
        self._downloads = downloads
        self._access = access
        self._sources = sources
        self._resolver = resolver
        self._accounts = accounts
        self._personal = personal
        super().__init__(*args, **kwargs)

    def parse_request(self):
        self._identity = None
        return super().parse_request()

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

    def _is_trusted(self) -> bool:
        return self._access.trusted(self.client_address[0], self.headers)

    def _authorized(self) -> bool:
        account = self._account()
        return account is not None and account is not _DISABLED

    def _reject_auth(self):
        if self._account() is _DISABLED:
            self._send_json({"error": "account_disabled"}, 403)
        elif self._identity[0] == "replaced":
            self._send_json({"error": "signed_in_elsewhere"}, 401)
        else:
            self._send_json({"error": "unauthorized"}, 401)

    def _send_json(self, value, status=200):
        if status == 401:
            _log_auth_reject(self.client_address[0], self.path, self.headers)
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json" if status == 401 else "application/json; charset=utf-8")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
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

    def _account(self):
        if self._identity is None:
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer xyd_"):
                self._identity = self._accounts.lookup(authorization[7:]) if self._accounts is not None else ("unknown", None)
            elif self._access.token_matches(self.headers) or self._is_trusted():
                self._identity = ("ok", OWNER)
            else:
                self._identity = ("unknown", None)
        state, account = self._identity
        if state == "disabled":
            return _DISABLED
        return account if state == "ok" else None

    def _owner_only(self):
        if self._account() == OWNER:
            return True
        self._send_json({"error": "owner_only"}, 403)
        return False

    def _home_only(self):
        """The sources are seen and changed at home only, like the connection settings."""
        if not self._owner_only():
            return False
        if self._is_trusted():
            return True
        LOGGER.warning("SOURCE-PANEL rejected addr=%s", self.client_address[0])
        self._send_json({"error": "home_only"}, 403)
        return False

    def _accounts_configured(self, path):
        if path.startswith(("/api/account", "/api/me/")) and (self._accounts is None or self._personal is None):
            self._send_json({"error": "accounts_unconfigured"}, 503)
            return False
        return True

    def _account_info(self, account):
        limits = self._resolver.ledger.limits() if self._resolver is not None else FAMILY_DEFAULTS
        today = self._resolver.ledger.recent(1)[0]["accounts"].get(account, {}) if self._resolver is not None else {}
        return {
            "account": {
                "id": account, "name": (self._accounts.owner_login_name() or "主账户") if account == OWNER else self._accounts.name_of(account),
                "owner": account == OWNER,
            },
            "limits": None if account == OWNER else {
                "downloadsPerDay": limits["familyDownloadsPerDay"], "playsPerDay": limits["familyPlaysPerDay"],
            },
            "today": {kind: today.get(kind, 0) for kind in ("download", "play", "ask")},
            "listening": self._personal.listening_status(account),
            "collections": {key: value for key, value in self._personal.collections(account).items() if key in ("revision", "updatedAt")},
        }

    def _get_account(self, url):
        account = self._account()
        if url.path == "/api/account/me":
            self._send_json(self._account_info(account))
        elif url.path == "/api/me/listening":
            query = parse_qs(url.query, keep_blank_values=True)
            self._send_json(self._personal.listening(
                account, self._integer(query, "after", 0), self._integer(query, "limit", 1000),
            ))
        elif url.path == "/api/me/collections":
            self._send_json(self._personal.collections(account))
        elif url.path == "/api/accounts" and self._home_only():
            rows = [{"id": OWNER, "name": "主账户（我）", "owner": True, **self._accounts.owner_panel()}, *self._accounts.panel_rows()]
            for row in rows:
                info = self._account_info(row["id"])
                row.update(owner=row["id"] == OWNER, limits=info["limits"], today=info["today"])
                row.update(self._personal.panel_status(row["id"]))
            self._send_json({"accounts": rows})
        elif url.path != "/api/accounts":
            self._send_json({"error": "not_found"}, 404)

    def _post_account(self, path):
        try:
            if path == "/api/account/login":
                body = self._read_json(4096)
                if body is not None:
                    self._send_json(self._accounts.login(body.get("name"), body.get("password"), body.get("device")))
            elif path == "/api/account/logout":
                if not self.headers.get("Authorization", "").startswith("Bearer xyd_"):
                    self._send_json({"error": "not_a_device"}, 400)
                else:
                    self._accounts.logout(self.headers["Authorization"][7:])
                    self._send_json({"ok": True})
            elif path in ("/api/me/listening", "/api/me/collections"):
                body = self._read_json((2 if path.endswith("/listening") else 6) * 1024 * 1024)
                if body is not None:
                    result = (
                        self._personal.append(self._account(), body.get("events")) if path.endswith("/listening")
                        else self._personal.save_collections(self._account(), body)
                    )
                    self._send_json(result)
            elif path == "/api/accounts" or _ACCOUNT_ITEM.fullmatch(path):
                if not self._home_only():
                    return
                body = self._read_json(4096)
                if body is None:
                    return
                item = _ACCOUNT_ITEM.fullmatch(path)
                if path == "/api/accounts":
                    self._send_json(self._accounts.create(body.get("name"), body.get("password")))
                elif item.group(2) is not None:
                    self._send_json({"error": "not_found"}, 404)
                elif item.group(1) == OWNER:
                    if "enabled" in body or "name" not in body or "password" not in body:
                        raise AccountError("bad_request")
                    self._accounts.set_owner_login(body["name"], body["password"])
                    self._send_json({"ok": True})
                else:
                    if "enabled" in body and type(body["enabled"]) is not bool:
                        raise AccountError("bad_request")
                    if "password" in body:
                        self._accounts.set_password(item.group(1), body["password"])
                    if "enabled" in body:
                        self._accounts.set_enabled(item.group(1), body["enabled"])
                    self._send_json({"ok": True})
            else:
                self._send_json({"error": "not_found"}, 404)
        except (AccountError, PersonalError) as error:
            self._send_json({"error": error.code, **(error.details if isinstance(error, PersonalError) else {})}, error.status)

    def _delete_account(self, path):
        item = _ACCOUNT_ITEM.fullmatch(path)
        if item is None:
            self._send_json({"error": "not_found"}, 404)
        elif self._home_only():
            try:
                if item.group(2) is not None:
                    self._accounts.revoke_device(item.group(1), item.group(2))
                else:
                    self._accounts.delete(item.group(1))
                    self._personal.delete_account_data(item.group(1))
            except AccountError as error:
                self._send_json({"error": error.code}, error.status)
                return
            self._send_json({"ok": True})

    def _get_source(self, path):
        if self._sources is None or self._resolver is None:
            self._send_json({"error": "source_unconfigured"}, 503)
        elif path.startswith(_STREAM_PREFIX):
            self._relay(path[len(_STREAM_PREFIX):])
        elif path == "/api/source/capabilities":
            self._send_json(self._resolver.capabilities())
        elif path not in ("/api/sources", "/api/source/limits", "/api/source/usage"):
            self._send_json({"error": "not_found"}, 404)
        elif self._home_only():
            if path == "/api/sources":
                self._send_json(self._resolver.panel_sources())
            elif path == "/api/source/limits":
                self._send_json(self._resolver.limits())
            else:
                usage = self._resolver.usage()
                for day in usage["days"]:
                    for row in day["accounts"]:
                        row["name"] = self._accounts.name_of(row["account"]) if self._accounts is not None else (
                            "主账户" if row["account"] == OWNER else "已删除的账户"
                        )
                self._send_json(usage)

    def _post_source(self, path):
        if self._sources is None or self._resolver is None:
            self._send_json({"error": "source_unconfigured"}, 503)
            return
        item = _SOURCE_ITEM.fullmatch(path)
        try:
            if path == "/api/source/resolve":
                body = self._read_json(256 * 1024)
                if body is not None:
                    self._send_json(self._resolver.resolve(body, self._account()))
            elif path == "/api/source/download":
                body = self._read_json(6 * 1024 * 1024)
                if body is not None:
                    self._send_json(self._resolver.download(body, self._account(), self._downloads))
            elif path == "/api/source/transport":
                body = self._read_json(4096)
                if body is not None:
                    self._send_json(self._resolver.report_transport(body))
            elif path not in ("/api/sources", "/api/sources/import", "/api/source/limits") and item is None:
                self._send_json({"error": "not_found"}, 404)
            elif not self._home_only():
                return
            elif item is not None and item.group(2) == "/test":
                self._send_json(self._resolver.test_source(item.group(1)))
            elif item is not None and item.group(2) == "/refresh":
                self._send_json(self._sources.refresh(item.group(1)))
            elif item is not None and item.group(2) == "/rollback":
                self._send_json(self._sources.rollback(item.group(1)))
            else:
                # A script is at most 1 MiB, but JSON may spell a byte in six.
                is_script = path == "/api/sources" or (item is not None and item.group(2) == "/replace")
                body = self._read_json(6 * 1024 * 1024 if is_script else 4096)
                if body is None:
                    return
                if path == "/api/sources":
                    self._send_json(self._sources.add(body.get("script")))
                elif path == "/api/sources/import":
                    self._send_json(self._sources.import_url(body.get("url")))
                elif item is not None and item.group(2) == "/replace":
                    self._send_json(self._sources.replace(item.group(1), body.get("script")))
                elif path == "/api/source/limits":
                    self._send_json(self._resolver.set_limits(body))
                else:
                    self._send_json(self._sources.update(item.group(1), body))
        except (ResolveError, SourceError, DownloadError) as error:
            self._send_json({"error": error.code}, error.status)

    def _relay(self, ticket):
        """The audio behind a ticket, Range and all, straight through: nothing is kept on disk."""
        try:
            stream = self._resolver.open_stream(ticket, self.headers.get("Range"))
        except ResolveError as error:
            if error.code == "not_found" and not self._authorized():
                self._reject_auth()
            else:
                self._send_json({"error": error.code}, error.status)
            return
        try:
            # A phone that stops reading for a minute lets its slot go.
            self.connection.settimeout(60)
            self.send_response(stream.status)
            for name, value in stream.headers.items():
                self.send_header(name, value)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for chunk in stream.chunks():
                self.wfile.write(chunk)
        except OSError:
            pass
        finally:
            stream.close()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if not self._accounts_configured(path):
            return
        if path.startswith(_STREAM_PREFIX):
            if self.headers.get("Authorization", "").startswith("Bearer xyd_") and not self._authorized():
                self._reject_auth()
                return
            self._get_source(path)
            return
        if path not in ("/", "/index.html") and not self._authorized():
            self._reject_auth()
            return
        if path in ("/api/logs", "/api/logs/download", "/api/llm") and not self._owner_only():
            return
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
        elif url.path.startswith(("/api/account", "/api/me/")):
            self._get_account(url)
        elif url.path in ("/api/logs", "/api/logs/download"):
            if self._data is None:
                self._send_json({"error": "not_found"}, 404)
                return
            if url.path == "/api/logs":
                query = parse_qs(url.query, keep_blank_values=True)
                self._send_json({"lines": log.tail(self._data, self._integer(query, "lines", 300))})
            else:
                payload = log.export(self._data)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="xiyue-container.log"')
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        elif url.path == "/api/sources" or url.path.startswith("/api/source/"):
            self._get_source(url.path)
        elif url.path.startswith("/api/"):
            if not self._panel_available():
                self._send_json({"error": "not_found"}, 404)
                return
            query = parse_qs(url.query, keep_blank_values=True)
            if url.path == "/api/access":
                self._send_json({**self._access.status(), "home": self._is_trusted()})
            elif url.path == "/api/status":
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
            elif url.path == "/api/downloads":
                if self._downloads is None:
                    self._send_json({"available": False, "jobs": []})
                else:
                    account = self._account()
                    jobs = self._downloads.snapshot()
                    if account == OWNER:
                        jobs = [{
                            **job, "accountName": (
                                self._accounts.name_of(job["account"]) if self._accounts is not None else "已删除的账户"
                            ) if job["account"] != OWNER else "",
                        } for job in jobs]
                    else:
                        jobs = [job for job in jobs if job["account"] == account]
                    self._send_json({"available": self._downloads.available(), "jobs": jobs})
            elif url.path == "/api/llm" and self._llm is not None:
                self._send_json(self._llm.status())
            else:
                self._send_json({"error": "not_found"}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        if not self._accounts_configured(path):
            return
        if path not in ("/", "/index.html", "/api/account/login") and not self._authorized():
            self._reject_auth()
            return
        if path in ("/api/llm", "/api/llm/test", "/api/scan", "/api/butler/artists", "/api/butler/songs") and not self._owner_only():
            return
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            if path.startswith(("/api/account", "/api/me/")):
                self._post_account(path)
            elif path == "/api/access":
                if not self._owner_only():
                    return
                if not self._is_trusted():
                    LOGGER.warning("SETTINGS access rejected addr=%s", self.client_address[0])
                    self._send_json({"error": "home_only"}, 403)
                    return
                body = self._read_json(4096)
                if body is None:
                    return
                try:
                    self._access.update(body.get("hostIP"), body.get("network"), body.get("token"))
                except AccessError:
                    self._send_json({"error": "bad_request"}, 400)
                    return
                status = self._access.status()
                LOGGER.info(
                    "SETTINGS access saved network=%s hostIP=%s token=%s",
                    status["network"], status["hostIP"], "已设" if status["tokenSet"] else "未设",
                )
                self._send_json({**status, "home": self._is_trusted()})
            elif path == "/api/sources" or path.startswith(("/api/sources/", "/api/source/")):
                self._post_source(path)
            elif path == "/api/scan" and self._panel_available():
                self._state.request_scan()
                self._send_json({"ok": True})
            elif path in ("/api/downloads", "/api/downloads/cancel", "/api/downloads/locate"):
                if self._downloads is None:
                    self._send_json({"error": "download_unconfigured"}, 503)
                    return
                body = self._read_json(6 * 1024 * 1024)
                if body is None:
                    return
                if path == "/api/downloads/cancel":
                    account = self._account()
                    self._send_json({"ok": self._downloads.cancel(body.get("id"), None if account == OWNER else account)})
                    return
                try:
                    if path == "/api/downloads/locate":
                        self._send_json({"directory": self._downloads.locate(body.get("name"))})
                    else:
                        self._send_json({"id": self._downloads.submit(body, self._account())})
                except DownloadError as error:
                    self._send_json({"error": error.code}, error.status)
            elif path == "/api/llm" and self._llm is not None:
                body = self._read_json(4096)
                if body is None:
                    return
                try:
                    self._llm.update(body.get("baseURL"), body.get("model"), body.get("apiKey"))
                except SettingsError:
                    self._send_json({"error": "bad_request"}, 400)
                    return
                LOGGER.info("SETTINGS llm saved model=%s", self._llm.status()["model"])
                self._send_json(self._llm.status())
            elif path == "/api/llm/test" and self._llm is not None:
                self._send_json(self._llm.test())
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
                part = body.get("part", "all")
                if part not in ASK_PARTS:
                    self._send_json({"error": "bad_query"}, 400)
                    return
                try:
                    answer = self._asker.ask(q.strip(), taste, part)
                    if self._resolver is not None:
                        self._resolver.ledger.note(self._account(), "ask")
                    self._send_json(answer)
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

    def do_DELETE(self):
        path = unquote(urlsplit(self.path).path)
        if not self._accounts_configured(path):
            return
        if path not in ("/", "/index.html") and not self._authorized():
            self._reject_auth()
            return
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            self.send_error(501, "Unsupported method ('DELETE')")
            return
        if path.startswith("/api/accounts"):
            self._delete_account(path)
            return
        item = _SOURCE_ITEM.fullmatch(path)
        if item is None or item.group(2):
            self._send_json({"error": "not_found"}, 404)
        elif self._sources is None or self._resolver is None:
            self._send_json({"error": "source_unconfigured"}, 503)
        elif self._home_only():
            try:
                self._sources.delete(item.group(1))
            except SourceError as error:
                self._send_json({"error": error.code}, error.status)
                return
            self._send_json({"ok": True})


def make_server(
    out, port, data=None, state=None, index=None, asker=None, butler=None, llm=None, downloads=None,
    access=None, sources=None, resolver=None, accounts=None, personal=None,
):
    if access is None:
        access = AccessSettings(None, network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2")
    handler = functools.partial(
        _Handler, directory=str(out), data=data, state=state, index=index, asker=asker, butler=butler, llm=llm,
        downloads=downloads,
        access=access, sources=sources, resolver=resolver, accounts=accounts, personal=personal,
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
