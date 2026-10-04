"""Plays and downloads songs with the owner's LX sources: picks a source, steps down tier by tier,
keeps the books, and relays the audio a phone cannot fetch itself.

A source tries each tier it has before the next source is asked; sources take turns by what is left
of their daily quota and how they have done lately. The owner is counted and never refused; any
other account stops at the family limits set in the panel. No log line or panel answer carries an
address, a ticket or a script: a line names the host and the source id only.
"""

import collections
import concurrent.futures
import datetime
import http.client
import json
import re
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import lxnet
from .downloads import DownloadError
from .log import LOGGER
from .lxhost import PLATFORMS, QUALITIES, RunnerError
from .sources import SourceError, write_private

OWNER = "owner"
KINDS = ("download", "play")
TRANSPORTS = ("direct", "relay")
# kg addresses only play from the network that asked for them, so the NAS relays kg from the start.
DEFAULT_TRANSPORTS = {"kw": "direct", "kg": "relay", "tx": "direct", "wy": "direct", "mg": "direct"}
FAMILY_LIMITS = {"download": "familyDownloadsPerDay", "play": "familyPlaysPerDay"}
FAMILY_DEFAULTS = {"familyDownloadsPerDay": 200, "familyPlaysPerDay": 300}
MAX_FAMILY_LIMIT = 10_000
USAGE_DAYS = 31
CACHE_SECONDS = 600
CACHE_ENTRIES = 500
MAX_PLAYS = 2000
TICKET_SECONDS = 600
MAX_TICKETS = 1000
FAILURE_WINDOW = 600
COOLDOWN_AFTER = 3
COOLDOWN_SECONDS = 600
REPORTED_SECONDS = 7 * 86400
MAX_STREAMS = 6
CHUNK = 64 * 1024
MAX_MUSIC_INFO_BYTES = 64 * 1024
UPSTREAM_TIMEOUT = 15.0
# What iOS LXHTTPMediaRelay sends the upstream.
RELAY_HEADERS = {"Accept": "*/*", "User-Agent": "Xiyue-LX-Relay/1"}
DOWNLOAD_FIELDS = ("filename", "directory", "lyrics", "cover", "minDurationMs")
TICKET = re.compile(r"[A-Za-z0-9_-]{43}")
TEST_TIER = "320k"
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_RANGE = re.compile(r"bytes=(?:\d{1,18}-\d{0,18}|-\d{1,18})")
_CONTENT_RANGE = re.compile(r"bytes (?:\d{1,18}-\d{1,18}|\*)/(?:\d{1,18}|\*)")
_LENGTH = re.compile(r"[0-9]{1,18}")
_PRINTABLE = re.compile(r"[\x20-\x7e]{1,200}")
# QQ hands FLAC out under these types.
_FLAC_DISGUISES = ("audio/x-ogg", "application/octet-stream")
# Failures the source answered for. The runner going away mid-call blames nobody unless it looks
# like the script's doing (RunnerError.suspect).
_SOURCE_FAULTS = frozenset(("script_failed", "invalid_result", "timeout", "bad_address", "unplayable_format"))
# The script was never asked.
_NOT_CALLED = frozenset(("runner_unavailable", "unavailable"))

try:
    SHANGHAI = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    # China has kept UTC+8 without daylight saving since 1991.
    SHANGHAI = datetime.timezone(datetime.timedelta(hours=8), "Asia/Shanghai")


def _sample(platform, ids):
    # The panel's test song (海阔天空) as iOS TrackIdentity builds a musicInfo, its tiers unknown.
    return {
        "name": "海阔天空", "singer": "BEYOND", "source": platform, "interval": "05:24",
        "albumName": "乐与怒", "albumId": "", "img": "", "types": [], "_types": {}, "typeUrl": {}, **ids,
    }


SAMPLES = {
    "kw": _sample("kw", {"songmid": "5886682"}),
    "tx": _sample("tx", {
        "songmid": "001yS0N33yPm1B", "songId": 4835784, "strMediaMid": "002MX8Ea4e5RDS", "albumMid": "",
    }),
    "kg": _sample("kg", {"songmid": "3261014", "hash": "C41E80A18D1448FA47086372999C7F43"}),
    "wy": _sample("wy", {"songmid": "1357375695"}),
}


class ResolveError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path, value):
    write_private(path, json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _count(value):
    return value if type(value) is int and value >= 0 else 0


def _usage_days(saved):
    days = {}
    found = saved.get("days") if isinstance(saved, dict) else None
    for day, book in found.items() if isinstance(found, dict) else ():
        if not isinstance(day, str) or not _DAY.fullmatch(day) or not isinstance(book, dict):
            continue
        sources, accounts = book.get("sources"), book.get("accounts")
        days[day] = {
            "sources": {
                key: _count(value) for key, value in sources.items() if isinstance(key, str) and 0 < len(key) <= 64
            } if isinstance(sources, dict) else {},
            "accounts": {
                key: {kind: _count(value.get(kind)) for kind in KINDS}
                for key, value in accounts.items()
                if isinstance(key, str) and 0 < len(key) <= 64 and isinstance(value, dict)
            } if isinstance(accounts, dict) else {},
        }
    return days


class SourceLedger:
    """Counts per Shanghai day: musicUrl calls by source and songs by account; and the family limits.

    The owner is counted and never refused. Counts for the last 31 days live in
    <data>/source-usage.json, the limits in <data>/source-limits.json.
    """

    def __init__(self, data, clock=time.time):
        self._clock = clock
        self._usage_path = Path(data) / "source-usage.json"
        self._limits_path = Path(data) / "source-limits.json"
        self._lock = threading.Lock()
        self._warned = None
        self._days = _usage_days(_read_json(self._usage_path))
        self._prune(self.today())
        saved = _read_json(self._limits_path)
        saved = saved if isinstance(saved, dict) else {}
        self._limits = {
            name: value if type(value := saved.get(name)) is int and 0 <= value <= MAX_FAMILY_LIMIT else default
            for name, default in FAMILY_DEFAULTS.items()
        }

    def today(self):
        return datetime.datetime.fromtimestamp(self._clock(), SHANGHAI).date().isoformat()

    def _prune(self, today):
        first = (datetime.date.fromisoformat(today) - datetime.timedelta(days=USAGE_DAYS - 1)).isoformat()
        for day in [day for day in self._days if day < first]:
            del self._days[day]

    def _book(self, day):
        # Under self._lock: the day's counts, made (and the oldest days dropped) on first use.
        book = self._days.get(day)
        if book is None:
            book = self._days[day] = {"sources": {}, "accounts": {}}
            self._prune(day)
        return book

    def _save(self):
        # Under self._lock. A count that cannot be saved still holds until the next restart.
        try:
            _write_json(self._usage_path, {"days": self._days})
        except OSError as error:
            now = time.monotonic()
            if self._warned is None or now - self._warned >= 600:
                self._warned = now
                LOGGER.warning("SOURCE-USAGE save-failed error=%s", type(error).__name__)

    def charge(self, account, kind):
        """One song resolved for an account (kind is "download" or "play"). Any account but the
        owner is refused at its daily limit. Returns a receipt for refund()."""
        with self._lock:
            day = self.today()
            accounts = self._book(day)["accounts"]
            counts = accounts.get(account) or {name: 0 for name in KINDS}
            if account != OWNER and counts[kind] >= self._limits[FAMILY_LIMITS[kind]]:
                raise ResolveError("quota_exceeded", 429)
            counts[kind] += 1
            accounts[account] = counts
            self._save()
        return day, account, kind

    def refund(self, receipt):
        day, account, kind = receipt
        with self._lock:
            counts = self._days.get(day, {}).get("accounts", {}).get(account)
            if counts is not None and counts[kind] > 0:
                counts[kind] -= 1
                self._save()

    def take_source(self, source_id, quota, enforce):
        """One musicUrl call on a source's books; with enforce, a source at its quota is left alone (None)."""
        with self._lock:
            day = self.today()
            sources = self._book(day)["sources"]
            used = sources.get(source_id, 0)
            if enforce and used >= quota:
                return None
            sources[source_id] = used + 1
            self._save()
        return day, source_id

    def untake_source(self, receipt):
        """The call never reached the script after all."""
        day, source_id = receipt
        with self._lock:
            sources = self._days.get(day, {}).get("sources", {})
            if sources.get(source_id, 0) > 0:
                sources[source_id] -= 1
                self._save()

    def used_today(self):
        with self._lock:
            return dict(self._days.get(self.today(), {}).get("sources", {}))

    def recent(self, count):
        """The last count days, today first, empty days included."""
        with self._lock:
            today = datetime.date.fromisoformat(self.today())
            days = []
            for back in range(count):
                day = (today - datetime.timedelta(days=back)).isoformat()
                book = self._days.get(day, {"sources": {}, "accounts": {}})
                days.append({
                    "date": day, "sources": dict(book["sources"]),
                    "accounts": {account: dict(counts) for account, counts in book["accounts"].items()},
                })
            return days

    def limits(self):
        with self._lock:
            return dict(self._limits)

    def set_limits(self, changes):
        with self._lock:
            limits = {**self._limits, **changes}
            _write_json(self._limits_path, limits)
            self._limits = limits


class Transports:
    """How each platform's addresses reach the phone: direct, or relayed by the NAS.

    The phone reporting that a direct address would not play moves its platform to relay for 7 days;
    the panel can set either for good. Kept in <data>/source-transport.json.
    """

    def __init__(self, data, clock=time.time):
        self._path = Path(data) / "source-transport.json"
        self._clock = clock
        self._lock = threading.Lock()
        saved = _read_json(self._path)
        self._overrides = {
            platform: {"transport": value["transport"], "origin": value["origin"], "at": int(value["at"])}
            for platform, value in (saved.items() if isinstance(saved, dict) else ())
            if platform in PLATFORMS and isinstance(value, dict) and value.get("transport") in TRANSPORTS
            and value.get("origin") in ("manual", "reported") and type(value.get("at")) is int
        }

    def _current(self, platform, now):
        # Under self._lock.
        override = self._overrides.get(platform)
        if override is not None and override["origin"] == "reported" and now >= override["at"] + REPORTED_SECONDS:
            return None
        return override

    def _save(self):
        # Under self._lock.
        now = self._clock()
        _write_json(self._path, {
            platform: override for platform, override in self._overrides.items()
            if self._current(platform, now) is not None
        })

    def mode(self, platform):
        with self._lock:
            override = self._current(platform, self._clock())
        return override["transport"] if override is not None else DEFAULT_TRANSPORTS[platform]

    def report(self, platform, transport, ok):
        """Only "a direct address would not play" changes anything."""
        with self._lock:
            now = self._clock()
            override = self._current(platform, now)
            current = override["transport"] if override is not None else DEFAULT_TRANSPORTS[platform]
            if transport != "direct" or ok is not False or current != "direct":
                return
            self._overrides[platform] = {"transport": "relay", "origin": "reported", "at": int(now)}
            try:
                self._save()
            except OSError as error:
                LOGGER.warning("SOURCE-TRANSPORT save-failed error=%s", type(error).__name__)
        LOGGER.info("SOURCE-TRANSPORT platform=%s transport=relay origin=reported", platform)

    def set(self, changes):
        """{platform: "direct" | "relay" | "default"} from the panel."""
        with self._lock:
            now = int(self._clock())
            for platform, value in changes.items():
                if value == "default":
                    self._overrides.pop(platform, None)
                else:
                    self._overrides[platform] = {"transport": value, "origin": "manual", "at": now}
            self._save()

    def status(self):
        with self._lock:
            now = self._clock()
            rows = []
            for platform in PLATFORMS:
                override = self._current(platform, now)
                reported = override is not None and override["origin"] == "reported"
                rows.append({
                    "platform": platform,
                    "transport": override["transport"] if override is not None else DEFAULT_TRANSPORTS[platform],
                    "default": DEFAULT_TRANSPORTS[platform],
                    "origin": override["origin"] if override is not None else "default",
                    "until": override["at"] + REPORTED_SECONDS if reported else None,
                })
            return rows


_Request = collections.namedtuple("_Request", "platform info songmid quality exact purpose available")
_Resolved = collections.namedtuple("_Resolved", "url quality source_id source_name expires cached")
_Cached = collections.namedtuple("_Cached", "url source_id expires")
_Landing = collections.namedtuple("_Landing", "tier expires")
_Mark = collections.namedtuple("_Mark", "expires")


class _Ticket:
    __slots__ = ("url", "expires", "flac")

    def __init__(self, url, expires):
        self.url, self.expires = url, expires
        # Whether the file starts with fLaC, once a stream has seen its first bytes.
        self.flac = None


class _Health:
    """A source's recent failures, in memory only."""

    def __init__(self):
        self.failures = collections.deque()
        self.streak = []
        self.cooldown_until = 0.0
        self.last_error = ""
        self.last_error_at = None

    def recent(self, now):
        while self.failures and now - self.failures[0] >= FAILURE_WINDOW:
            self.failures.popleft()
        return len(self.failures)


def _present(value):
    """An id as a script reads one: a number, or a string with something in it."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) or (isinstance(value, str) and value.strip() != "")


def _request(body, purpose=None):
    """The phone's song, checked for shape only: musicInfo reaches the script as it came."""
    platform, info, quality = body.get("platform"), body.get("musicInfo"), body.get("quality")
    exact = body.get("exact", False)
    purpose = purpose or body.get("purpose", "play")
    if (
        platform not in PLATFORMS or not isinstance(info, dict) or quality not in QUALITIES
        or not isinstance(exact, bool) or purpose not in KINDS
    ):
        raise ResolveError("bad_request")
    try:
        size = len(json.dumps(info, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    except ValueError:
        raise ResolveError("bad_request") from None
    if size > MAX_MUSIC_INFO_BYTES:
        raise ResolveError("too_large", 413)
    # iOS TrackIdentity: tx needs songId and strMediaMid; kg's hash is only there when iOS has it.
    if not _present(info.get("songmid")) or (platform == "tx" and not (
        _present(info.get("songId")) and isinstance(info.get("strMediaMid"), str) and info["strMediaMid"].strip()
    )):
        raise ResolveError("bad_request")
    types = info.get("_types")
    available = frozenset(types) if isinstance(types, dict) and types else None
    return _Request(platform, info, str(info["songmid"]), quality, exact, purpose, available)


def _tiers(quality, exact, declared, available):
    """iOS playQualities: the preferred tier, then (unless exact) each lower one, as far as the source
    declares it and the song is known to have it."""
    wanted = [quality] if exact else QUALITIES[:QUALITIES.index(quality) + 1][::-1]
    return [tier for tier in wanted if tier in declared and (available is None or tier in available)]


def _put(table, key, value, now, limit):
    table.pop(key, None)
    table[key] = value
    _trim(table, now, limit)


def _trim(table, now, limit):
    # Entries go in with the same lifetime, so the oldest (first) expire first.
    while table and (len(table) > limit or next(iter(table.values())).expires <= now):
        table.popitem(last=False)


def _relay_headers(response):
    """The upstream headers the phone gets: type, length and range, nothing else."""
    headers = {}
    content_type = response.getheader("Content-Type", "")
    headers["Content-Type"] = content_type if _PRINTABLE.fullmatch(content_type) else "application/octet-stream"
    length = response.getheader("Content-Length", "").strip()
    if _LENGTH.fullmatch(length):
        headers["Content-Length"] = length
    content_range = response.getheader("Content-Range", "").strip()
    if _CONTENT_RANGE.fullmatch(content_range):
        headers["Content-Range"] = content_range
    accept = response.getheader("Accept-Ranges", "").strip().lower()
    if accept in ("bytes", "none"):
        headers["Accept-Ranges"] = accept
    return headers


def _read_head(response):
    head = b""
    while len(head) < 4:
        piece = response.read1(4 - len(head))
        if not piece:
            break
        head += piece
    return head


class Stream:
    """One relayed answer: send status and headers, then chunks(); close() however that ends."""

    def __init__(self, status, headers, connection, response, head, host, release):
        self.status, self.headers = status, headers
        self._connection, self._response, self._head = connection, response, head
        self._host, self._release = host, release
        self._lock = threading.Lock()
        self._closed = False

    def chunks(self):
        """The body straight from the upstream in pieces of at most 64 KiB; nothing touches the disk."""
        if self._head:
            yield self._head
        try:
            while True:
                piece = self._response.read1(CHUNK)
                if not piece:
                    return
                yield piece
        except (OSError, ValueError, http.client.HTTPException) as error:
            if not self._closed:
                LOGGER.warning("SOURCE-STREAM cut host=%s error=%s", self._host, type(error).__name__)

    def close(self):
        """Cuts the upstream too, so a phone that went away leaves nothing running."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._connection.close()
        self._release()


class Resolver:
    """Resolves songs with the enabled sources; the phone's and the panel's view of them."""

    def __init__(self, sources, runner, data, *, clock=time.time, allow_private=False):
        self._sources = sources
        self._runner = runner
        self._clock = clock
        self._allow_private = allow_private
        self.ledger = SourceLedger(data, clock)
        self.transports = Transports(data, clock)
        self._lock = threading.Lock()
        self._urls = collections.OrderedDict()
        self._landings = collections.OrderedDict()
        self._plays = collections.OrderedDict()
        self._tickets = collections.OrderedDict()
        self._health = {}
        self._streams = threading.BoundedSemaphore(MAX_STREAMS)

    # The phone's side.

    def capabilities(self):
        platforms = {}
        for entry in self._sources.list():
            if not entry["enabled"] or entry["dailyQuota"] <= 0:
                continue
            for platform, qualities in entry["platforms"].items():
                if platform in PLATFORMS:
                    platforms.setdefault(platform, set()).update(qualities)
        return {"platforms": {
            platform: ordered for platform, qualities in platforms.items()
            if (ordered := [quality for quality in QUALITIES if quality in qualities])
        }}

    def resolve(self, body, account):
        """POST /api/source/resolve: the address itself, or a relay path standing in for it."""
        request = _request(body)
        transport = self.transports.mode(request.platform)
        found, _ = self._logged(request, account, transport)
        if transport == "direct" and urlsplit(found.url).scheme == "http":
            transport = "relay"
            LOGGER.info("SOURCE-RESOLVE http_relay platform=%s", request.platform)
        answer = {
            "quality": found.quality, "sourceID": found.source_id, "sourceName": found.source_name,
            "transport": transport,
        }
        if transport == "direct":
            answer.update(url=found.url, expiresAt=int(found.expires))
        else:
            ticket, expires = self._issue(found.url)
            answer.update(path=f"/api/source/stream/{ticket}", expiresAt=int(expires))
        return answer

    def download(self, body, account, downloads):
        """POST /api/source/download: resolves for a download and hands the address to the NAS's queue."""
        if downloads is None or not downloads.available():
            raise ResolveError("download_unconfigured", 503)
        request = _request(body, purpose="download")
        found, receipt = self._logged(request, account, "nas")
        job = {
            "url": found.url, **{name: body[name] for name in DOWNLOAD_FIELDS if name in body},
            "sourceName": found.source_name,
        }
        try:
            job_id = downloads.submit(job)
        except DownloadError:
            self.ledger.refund(receipt)
            raise
        return {"id": job_id, "quality": found.quality, "sourceName": found.source_name}

    def report_transport(self, body):
        """POST /api/source/transport: only "direct would not play" changes anything."""
        platform, transport, ok = body.get("platform"), body.get("transport"), body.get("ok")
        if platform not in PLATFORMS or transport not in TRANSPORTS or not isinstance(ok, bool):
            raise ResolveError("bad_request")
        self.transports.report(platform, transport, ok)
        return {"platform": platform, "transport": self.transports.mode(platform)}

    def open_stream(self, ticket, range_header):
        """GET /api/source/stream/<ticket>: the upstream's answer to the phone's Range.

        A seek is a new Range on the same ticket: relayed again, never resolved or counted again.
        """
        now = self._clock()
        with self._lock:
            entry = self._tickets.get(ticket) if isinstance(ticket, str) and TICKET.fullmatch(ticket) else None
            if entry is not None and entry.expires <= now:
                del self._tickets[ticket]
                entry = None
        if entry is None:
            raise ResolveError("not_found", 404)
        if not self._streams.acquire(blocking=False):
            LOGGER.warning("SOURCE-STREAM busy streams=%s", MAX_STREAMS)
            raise ResolveError("busy", 429)
        host = lxnet.host_of(entry.url)
        headers = dict(RELAY_HEADERS)
        wanted = range_header.strip() if isinstance(range_header, str) else ""
        if _RANGE.fullmatch(wanted):
            headers["Range"] = wanted
        connection = stream = None
        try:
            connection, response = lxnet.open_media(
                entry.url, headers, allow_private=self._allow_private, timeout=UPSTREAM_TIMEOUT,
            )
            if response.status in (200, 206, 416):
                relayed = _relay_headers(response)
                head = b""
                disguised = relayed["Content-Type"].split(";", 1)[0].strip().lower() in _FLAC_DISGUISES
                if response.status != 416 and disguised:
                    if entry.flac is None:
                        if response.status == 200 or relayed.get("Content-Range", "").startswith("bytes 0-"):
                            head = _read_head(response)
                            entry.flac = head == b"fLaC"
                        else:
                            entry.flac = self._starts_with_flac(entry.url)
                    if entry.flac:
                        relayed["Content-Type"] = "audio/flac"
                stream = Stream(response.status, relayed, connection, response, head, host, self._streams.release)
                return stream
            LOGGER.warning("SOURCE-STREAM failed host=%s status=%s", host, response.status)
        except (lxnet.NetError, OSError, http.client.HTTPException) as error:
            LOGGER.warning(
                "SOURCE-STREAM failed host=%s error=%s", host,
                error.code if isinstance(error, lxnet.NetError) else type(error).__name__,
            )
        finally:
            if stream is None:
                if connection is not None:
                    connection.close()
                self._streams.release()
        raise ResolveError("upstream_failed", 502)

    def _starts_with_flac(self, url):
        """For a stream that starts past the file's head: asks for its first 4 bytes. None when unknown."""
        headers = {**RELAY_HEADERS, "Range": "bytes=0-3"}
        try:
            connection, response = lxnet.open_media(
                url, headers, allow_private=self._allow_private, timeout=UPSTREAM_TIMEOUT,
            )
        except (lxnet.NetError, OSError, http.client.HTTPException):
            return None
        try:
            return _read_head(response) == b"fLaC" if response.status in (200, 206) else None
        except (OSError, http.client.HTTPException):
            return None
        finally:
            connection.close()

    # The panel's side.

    def panel_sources(self):
        """GET /api/sources: every source with today's use and how it has done lately; never a script."""
        entries = self._sources.list()
        used = self.ledger.used_today()
        now = self._clock()
        rows = []
        with self._lock:
            for entry in entries:
                health = self._health.get(entry["id"])
                rows.append({
                    **entry,
                    "used": used.get(entry["id"], 0),
                    "failures": health.recent(now) if health is not None else 0,
                    "cooldownUntil": (
                        int(health.cooldown_until) if health is not None and health.cooldown_until > now else None
                    ),
                    "lastError": health.last_error if health is not None else "",
                    "lastErrorAt": int(health.last_error_at) if health is not None and health.last_error_at else None,
                })
        return {"runner": self._runner.running, "sources": rows}

    def usage(self):
        """GET /api/source/usage: today and the 6 days before, by source and by account."""
        entries = self._sources.list()
        names = {entry["id"]: entry["displayName"] for entry in entries}
        days = [
            {
                "date": book["date"],
                "used": sum(book["sources"].values()),
                "sources": [
                    {"id": source_id, "name": names.get(source_id, ""), "used": count}
                    for source_id, count in sorted(book["sources"].items(), key=lambda item: -item[1])
                ],
                "accounts": [{"account": account, **counts} for account, counts in sorted(book["accounts"].items())],
            }
            for book in self.ledger.recent(7)
        ]
        quota = sum(entry["dailyQuota"] for entry in entries if entry["enabled"])
        return {"today": days[0]["date"], "quota": quota, "days": days}

    def limits(self):
        """GET /api/source/limits: the family limits and each platform's transport. The owner has none."""
        return {**self.ledger.limits(), "transports": self.transports.status()}

    def set_limits(self, body):
        """POST /api/source/limits: everything is checked before anything is saved."""
        limits = {}
        for name in FAMILY_DEFAULTS:
            value = body.get(name)
            if value is None:
                continue
            if type(value) is not int or not 0 <= value <= MAX_FAMILY_LIMIT:
                raise ResolveError("bad_request")
            limits[name] = value
        transports = body.get("transports")
        if transports is not None and (not isinstance(transports, dict) or not all(
            platform in PLATFORMS and value in (*TRANSPORTS, "default") for platform, value in transports.items()
        )):
            raise ResolveError("bad_request")
        if not limits and not transports:
            raise ResolveError("bad_request")
        try:
            if limits:
                self.ledger.set_limits(limits)
            if transports:
                self.transports.set(transports)
        except OSError:
            raise ResolveError("save_failed", 500) from None
        LOGGER.info(
            "SOURCE-LIMITS downloads=%s plays=%s transports=%s",
            limits.get("familyDownloadsPerDay", "-"), limits.get("familyPlaysPerDay", "-"),
            ",".join(f"{platform}:{value}" for platform, value in (transports or {}).items()) or "-",
        )
        return self.limits()

    def test_source(self, source_id):
        """POST /api/sources/<id>/test: the sample song at 320k on each sampled platform the source has."""
        try:
            entry = self._sources.ensure_loaded(source_id)
        except SourceError as error:
            raise ResolveError(error.code, error.status) from None
        platforms = [platform for platform in SAMPLES if platform in entry["platforms"]]
        results = []
        if platforms:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(platforms)) as pool:
                results = list(pool.map(lambda platform: self._try_sample(entry, platform), platforms))
        LOGGER.info(
            "SOURCE-TEST id=%s results=%s", source_id,
            ",".join(f"{item['platform']}:{'ok' if item['ok'] else item['error']}" for item in results) or "-",
        )
        return {"id": source_id, "results": results}

    def _try_sample(self, entry, platform):
        result = {"platform": platform, "ok": False, "ms": 0, "quality": TEST_TIER, "host": "", "error": ""}
        if TEST_TIER not in entry["platforms"][platform]:
            result["error"] = "unsupported_quality"
            return result
        receipt = self.ledger.take_source(entry["id"], entry["dailyQuota"], enforce=False)
        started = time.monotonic()
        try:
            url = self._runner.call(entry["id"], platform, TEST_TIER, SAMPLES[platform])
        except RunnerError as error:
            if error.code in _NOT_CALLED:
                self.ledger.untake_source(receipt)
            result.update(ms=round((time.monotonic() - started) * 1000), error=error.code)
            return result
        result.update(ms=round((time.monotonic() - started) * 1000), host=lxnet.host_of(url))
        problem = self._address_problem(url)
        result.update(ok=problem is None, error=problem or "")
        return result

    # Resolving.

    def _logged(self, request, account, via):
        started = time.monotonic()
        try:
            found, receipt = self._resolve(request, account)
        except ResolveError as error:
            LOGGER.warning(
                "SOURCE-RESOLVE platform=%s purpose=%s quality=%s exact=%s error=%s ms=%s account=%s",
                request.platform, request.purpose, request.quality, "是" if request.exact else "否",
                error.code, round((time.monotonic() - started) * 1000), account,
            )
            raise
        LOGGER.info(
            "SOURCE-RESOLVE platform=%s purpose=%s quality=%s exact=%s got=%s source=%s via=%s cache=%s ms=%s "
            "host=%s account=%s",
            request.platform, request.purpose, request.quality, "是" if request.exact else "否", found.quality,
            found.source_id, via, "是" if found.cached else "否", round((time.monotonic() - started) * 1000),
            lxnet.host_of(found.url), account,
        )
        return found, receipt

    def _resolve(self, request, account):
        """(the song's address, the account's receipt or None). The account is charged before any
        source is asked, so a family account past its limit costs no source a call."""
        hit = self._cached(request)
        if hit is not None:
            receipt = None
            if request.purpose == "download" or self._first_play(account, request, hit.quality):
                try:
                    receipt = self.ledger.charge(account, request.purpose)
                except ResolveError:
                    self._forget_play(account, request, hit.quality)
                    raise
            return hit, receipt
        receipt = self.ledger.charge(account, request.purpose)
        try:
            found = self._ask_sources(request, account == OWNER)
        except BaseException:
            self.ledger.refund(receipt)
            raise
        if request.purpose == "play" and not self._first_play(account, request, found.quality):
            self.ledger.refund(receipt)
            receipt = None
        return found, receipt

    def _cached(self, request):
        now = self._clock()
        with self._lock:
            _trim(self._urls, now, CACHE_ENTRIES)
            _trim(self._landings, now, CACHE_ENTRIES)
            tier = request.quality
            cached = self._urls.get((request.platform, request.songmid, tier))
            if cached is None:
                landing = self._landings.get((request.platform, request.songmid, request.quality, request.exact))
                if landing is not None:
                    tier = landing.tier
                    cached = self._urls.get((request.platform, request.songmid, tier))
        if cached is None:
            return None
        # A source switched off or deleted since does not answer from the cache either.
        entry = self._sources.get(cached.source_id)
        if entry is None or not entry["enabled"] or entry["dailyQuota"] <= 0:
            return None
        return _Resolved(cached.url, tier, cached.source_id, entry["displayName"], cached.expires, True)

    def _first_play(self, account, request, tier):
        """Marks a play; False when the account played this song at this tier in the last 10 minutes."""
        key = (account, request.platform, request.songmid, tier)
        now = self._clock()
        with self._lock:
            _trim(self._plays, now, MAX_PLAYS)
            if key in self._plays:
                return False
            _put(self._plays, key, _Mark(now + CACHE_SECONDS), now, MAX_PLAYS)
            return True

    def _forget_play(self, account, request, tier):
        with self._lock:
            self._plays.pop((account, request.platform, request.songmid, tier), None)

    def _ask_sources(self, request, owner):
        """Each source in turn, each of its tiers from the preferred one down, until an address passes.

        A source that fails every tier counts one failure toward its cooldown.
        """
        used = self.ledger.used_today()
        candidates = [
            entry for entry in self._sources.list()
            if entry["enabled"] and entry["dailyQuota"] > 0 and request.platform in entry["platforms"]
        ]
        if not candidates:
            raise ResolveError("no_source", 404)
        if not owner:
            candidates = [entry for entry in candidates if used.get(entry["id"], 0) < entry["dailyQuota"]]
            if not candidates:
                raise ResolveError("sources_exhausted", 429)
        now = self._clock()
        with self._lock:
            candidates.sort(key=lambda entry: self._rank(entry, used, now))
        called = exhausted = skipped = False
        for entry in candidates:
            source_id = entry["id"]
            blamed = None
            for tier in _tiers(request.quality, request.exact, entry["platforms"][request.platform], request.available):
                taken = self.ledger.take_source(source_id, entry["dailyQuota"], enforce=not owner)
                if taken is None:
                    exhausted = True
                    break
                try:
                    url = self._runner.call(source_id, request.platform, tier, request.info)
                except RunnerError as error:
                    if error.code in _NOT_CALLED:
                        self.ledger.untake_source(taken)
                        skipped = True
                        break
                    code, blame = error.code, error.code in _SOURCE_FAULTS or error.suspect
                else:
                    code = self._address_problem(url)
                    if code is None:
                        self._succeeded(source_id)
                        expires = self._store(request, tier, url, source_id)
                        return _Resolved(url, tier, source_id, entry["displayName"], expires, False)
                    blame = True
                called = True
                if blame:
                    blamed = code
            if blamed is not None:
                self._failed(source_id, request.platform, blamed)
        if called:
            raise ResolveError("resolve_failed", 502)
        if exhausted:
            raise ResolveError("sources_exhausted", 429)
        if skipped:
            raise ResolveError("runner_unavailable", 503)
        raise ResolveError("unsupported_quality", 404)

    def _rank(self, entry, used, now):
        # Under self._lock. Cooling sources last; then those with quota left; then fewer failures in the
        # last 10 minutes; then the smaller share of the quota used; then the panel's order.
        health = self._health.get(entry["id"])
        spent, quota = used.get(entry["id"], 0), entry["dailyQuota"]
        cooling = health is not None and health.cooldown_until > now
        failures = health.recent(now) if health is not None else 0
        return cooling, spent >= quota, failures, spent / quota, entry["order"]

    def _address_problem(self, url):
        """iOS canonicalMediaURL and isUnplayableMediaFormat on a musicUrl answer."""
        try:
            lxnet.media_url(url, allow_private=self._allow_private)
        except lxnet.NetError:
            return "bad_address"
        return "unplayable_format" if lxnet.is_unplayable(url) else None

    def _store(self, request, tier, url, source_id):
        now = self._clock()
        expires = now + CACHE_SECONDS
        with self._lock:
            _put(self._urls, (request.platform, request.songmid, tier), _Cached(url, source_id, expires), now, CACHE_ENTRIES)
            _put(
                self._landings, (request.platform, request.songmid, request.quality, request.exact),
                _Landing(tier, expires), now, CACHE_ENTRIES,
            )
        return expires

    def _issue(self, url):
        ticket = secrets.token_urlsafe(32)
        now = self._clock()
        expires = now + TICKET_SECONDS
        with self._lock:
            _put(self._tickets, ticket, _Ticket(url, expires), now, MAX_TICKETS)
        return ticket, expires

    def _succeeded(self, source_id):
        with self._lock:
            health = self._health.get(source_id)
            if health is not None:
                health.streak = []
                health.cooldown_until = 0.0

    def _failed(self, source_id, platform, code):
        now = self._clock()
        with self._lock:
            health = self._health.setdefault(source_id, _Health())
            health.failures.append(now)
            health.streak = [moment for moment in health.streak if now - moment < FAILURE_WINDOW] + [now]
            health.last_error, health.last_error_at = code, now
            cooled = len(health.streak) >= COOLDOWN_AFTER
            if cooled:
                health.streak = []
                health.cooldown_until = now + COOLDOWN_SECONDS
        LOGGER.warning("SOURCE-FAIL source=%s platform=%s error=%s", source_id, platform, code)
        if cooled:
            LOGGER.warning("SOURCE-COOLDOWN source=%s minutes=%s", source_id, COOLDOWN_SECONDS // 60)
