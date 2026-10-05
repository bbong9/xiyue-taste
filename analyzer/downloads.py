import base64
import binascii
import ipaddress
import os
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from pathlib import Path

import mutagen
from mutagen.aac import AAC
from mutagen.flac import FLAC
from mutagen.monkeysaudio import MonkeysAudio
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from . import loudness
from .log import LOGGER

AUDIO_EXTENSIONS = ("flac", "mp3", "m4a", "aac", "ogg", "opus", "wav", "ape")
MAX_AUDIO_BYTES = 600 * 1024 * 1024
MIN_AUDIO_BYTES = 64 * 1024
MAX_COVER_BYTES = 3 * 1024 * 1024
MAX_LYRICS_CHARS = 200_000
MAX_WAITING = 500
KEPT_JOBS = 200
SPARE_BYTES = 2 * 1024 * 1024 * 1024
USER_AGENT = "Mozilla/5.0"
SUFFIXES = {
    "FLAC": "flac", "MP3": "mp3", "MP4": "m4a", "OggVorbis": "ogg", "OggOpus": "opus",
    "WAVE": "wav", "MonkeysAudio": "ape", "AAC": "aac",
}


class DownloadError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _check_shape(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise DownloadError("bad_url")
    return parts


def check_public(url):
    """Only addresses out on the internet: never this NAS or the home network."""
    parts = _check_shape(url)
    try:
        found = socket.getaddrinfo(
            parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError):
        raise DownloadError("bad_url")
    if not found or not all(ipaddress.ip_address(item[4][0].split("%")[0]).is_global for item in found):
        raise DownloadError("bad_url")


class _Guard(urllib.request.HTTPRedirectHandler):
    """Every hop of a redirect is checked like the first address."""

    def __init__(self, check):
        self._check = check

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _is_audio(head):
    return (
        head.startswith((b"fLaC", b"ID3", b"OggS", b"RIFF", b"MAC "))
        or head[4:8] == b"ftyp"
        or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
    )


def _facts(path):
    """What the file really is, whatever its address or its name said."""
    # By name and first bytes; then each kind in turn, for a song whose name
    # says one format and whose bytes are another.
    for kind in (mutagen.File, FLAC, MP4, OggOpus, OggVorbis, WAVE, MonkeysAudio, MP3, AAC):
        try:
            audio = kind(path)
        except Exception:
            continue
        info = getattr(audio, "info", None)
        suffix = SUFFIXES.get(type(audio).__name__)
        length = getattr(info, "length", 0) or 0
        if suffix is not None and length > 0:
            break
    else:
        raise DownloadError("not_audio")
    return {
        "format": suffix,
        "durationMs": int(length * 1000),
        "sampleRate": int(getattr(info, "sample_rate", 0) or 0),
        "bitDepth": int(getattr(info, "bits_per_sample", 0) or 0),
        "bitRate": int(getattr(info, "bitrate", 0) or 0),
    }


def _cover_suffix(data):
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return None


def _text(value, limit):
    if not isinstance(value, str) or len(value) > limit or any(c in value for c in "\r\n\0"):
        raise DownloadError("bad_request")
    return value


def _filename(value):
    name = _text(value, 200)
    if (
        not name or name != name.strip() or name.startswith(".")
        or any(c in name for c in "/\\")
        or name.rsplit(".", 1)[-1].lower() not in AUDIO_EXTENSIONS or "." not in name
    ):
        raise DownloadError("bad_request")
    return name


def _directory(value):
    directory = _text(value, 300)
    if not directory:
        return ""
    parts = directory.split("/")
    if len(parts) > 6 or "\\" in directory or any(
        not part or part != part.strip() or part.startswith(".") for part in parts
    ):
        raise DownloadError("bad_request")
    return directory


class Downloads:
    """Songs the phone asks this NAS to fetch itself: one at a time, into the
    downloads folder and nowhere else. Addresses are used, never listed."""

    def __init__(self, root, allow_private=False):
        self._root = Path(root)
        self._check = _check_shape if allow_private else check_public
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._jobs = {}
        self._waiting = deque()
        threading.Thread(target=self._work, daemon=True).start()

    def available(self):
        return self._root.is_dir() and os.access(self._root, os.W_OK)

    def submit(self, body, account="owner"):
        if not self.available():
            raise DownloadError("download_unconfigured", 503)
        url = _text(body.get("url"), 4000)
        _check_shape(url)
        lyrics = body.get("lyrics")
        if lyrics is not None and (not isinstance(lyrics, str) or len(lyrics) > MAX_LYRICS_CHARS):
            raise DownloadError("bad_request")
        cover = body.get("cover")
        if cover is not None:
            try:
                cover = base64.b64decode(_text(cover, MAX_COVER_BYTES * 2), validate=True)
            except (binascii.Error, ValueError):
                raise DownloadError("bad_request")
            if len(cover) > MAX_COVER_BYTES or _cover_suffix(cover) is None:
                raise DownloadError("bad_request")
        minimum = body.get("minDurationMs", 0)
        if type(minimum) is not int or not 0 <= minimum <= 86_400_000:
            raise DownloadError("bad_request")
        source_name = body.get("sourceName", "")
        if not isinstance(source_name, str):
            raise DownloadError("bad_request")
        source_name = source_name.strip()
        if len(source_name) > 64:
            raise DownloadError("bad_request")
        job = {
            "id": uuid.uuid4().hex,
            "account": account,
            "filename": _filename(body.get("filename")),
            "directory": _directory(body.get("directory", "")),
            "state": "queued", "received": 0, "total": 0, "error": "", "path": "",
            "host": urllib.parse.urlsplit(url).hostname,
            "createdAt": time.time(), "finishedAt": None,
            "format": "", "durationMs": 0, "sampleRate": 0, "bitDepth": 0, "bitRate": 0,
            "minDurationMs": minimum,
            "sourceName": source_name,
            "url": url,
            "userAgent": _text(body.get("userAgent", ""), 500),
            "referer": _text(body.get("referer", ""), 2000),
            "lyrics": lyrics, "cover": cover, "cancelled": False,
        }
        with self._wake:
            if len(self._waiting) >= MAX_WAITING:
                raise DownloadError("busy", 429)
            self._jobs[job["id"]] = job
            self._waiting.append(job["id"])
            for old in [key for key, item in self._jobs.items() if item["finishedAt"] is not None]:
                if len(self._jobs) <= KEPT_JOBS:
                    break
                del self._jobs[old]
            LOGGER.info(
                "DOWNLOAD-START id=%s host=%s dir=%s file=%s account=%s",
                job["id"], job["host"], job["directory"], job["filename"], account,
            )
            self._wake.notify()
        return job["id"]

    def snapshot(self):
        names = ("id", "filename", "directory", "state", "received", "total", "error", "path", "host",
                 "createdAt", "finishedAt", "format", "durationMs", "sampleRate", "bitDepth", "bitRate", "sourceName", "account")
        with self._lock:
            return [{name: job[name] for name in names} for job in list(self._jobs.values())[::-1][:100]]

    def cancel(self, job_id, account=None):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] not in ("queued", "downloading"):
                return False
            if account is not None and job["account"] != account:
                return False
            job["cancelled"] = True
            if job["state"] == "queued":
                self._finish(job, "cancelled")
            return True

    def locate(self, name):
        """Where under the downloads folder a marker is that the phone just put
        there over its own connection: proof that both mean the same folder."""
        if not isinstance(name, str) or not re.fullmatch(r"\.xiyue-probe-[0-9a-f]{32}", name):
            raise DownloadError("bad_request")
        seen = 0
        for folder, folders, files in os.walk(self._root):
            relative = Path(folder).relative_to(self._root)
            folders[:] = [] if len(relative.parts) >= 6 else sorted(
                item for item in folders if not item.startswith(".")
            )
            if name in files:
                directory = "" if not relative.parts else relative.as_posix()
                LOGGER.info("DOWNLOAD-LOCATE found=是 dir=%s", directory)
                return directory
            seen += 1
            if seen > 20000:
                break
        LOGGER.info("DOWNLOAD-LOCATE found=否 dir=")
        raise DownloadError("not_found", 404)

    def _finish(self, job, state, error="", path=""):
        job.update(state=state, error=error, path=path, finishedAt=time.time(), url="", lyrics=None, cover=None)
        LOGGER.info(
            "DOWNLOAD-END id=%s state=%s error=%s bytes=%s format=%s seconds=%.2f",
            job["id"], state, error, job.get("received", ""), job.get("format", ""),
            job["finishedAt"] - job["createdAt"],
        )

    def _work(self):
        while True:
            with self._wake:
                while not self._waiting:
                    self._wake.wait()
                job = self._jobs.get(self._waiting.popleft())
                if job is None or job["state"] != "queued":
                    continue
                job["state"] = "downloading"
            state, error, path = "failed", "", ""
            try:
                path = self._fetch(job)
                state = "done"
            except DownloadError as failure:
                state, error = ("cancelled", "") if failure.code == "cancelled" else ("failed", failure.code)
            except urllib.error.HTTPError as failure:
                error = f"http_{failure.code}"
            except urllib.error.URLError:
                error = "unreachable"
            except (OSError, ValueError):
                error = "io_failed"
            with self._lock:
                self._finish(job, state, error, path)

    def _own(self, path, folder=False):
        # As whoever owns the downloads folder, so the NAS's own apps and
        # WebDAV can still rename and delete what lands here.
        try:
            owner = self._root.stat()
            os.chown(path, owner.st_uid, owner.st_gid)
            os.chmod(path, owner.st_mode & (0o777 if folder else 0o666))
        except OSError:
            pass

    def _fetch(self, job):
        self._check(job["url"])
        directory = self._root / job["directory"]
        made, current = [], directory
        while current != self._root and not current.exists():
            made.append(current)
            current = current.parent
        # A link inside the downloads folder must not lead the song out of it.
        if not current.resolve().is_relative_to(self._root.resolve()):
            raise DownloadError("outside")
        directory.mkdir(parents=True, exist_ok=True)
        for folder in made:
            self._own(folder, folder=True)
        # Named like the song so the format is recognised by more than its first bytes.
        part = directory / f".xiyue-part-{job['id']}.{job['filename'].rsplit('.', 1)[-1].lower()}"
        headers = {"User-Agent": job["userAgent"] or USER_AGENT}
        if job["referer"]:
            headers["Referer"] = job["referer"]
        opener = urllib.request.build_opener(_Guard(self._check))
        try:
            with opener.open(urllib.request.Request(job["url"], headers=headers), timeout=30) as response:
                try:
                    total = int(response.headers.get("Content-Length") or 0)
                except ValueError:
                    total = 0
                if total > MAX_AUDIO_BYTES:
                    raise DownloadError("too_large")
                if shutil.disk_usage(directory).free < total + SPARE_BYTES:
                    raise DownloadError("no_space")
                with self._lock:
                    job["total"] = total
                received = 0
                with open(part, "wb") as file:
                    while True:
                        chunk = response.read(262144)
                        if not chunk:
                            break
                        if received == 0 and not _is_audio(chunk):
                            raise DownloadError("not_audio")
                        received += len(chunk)
                        if received > MAX_AUDIO_BYTES:
                            raise DownloadError("too_large")
                        file.write(chunk)
                        with self._lock:
                            job["received"] = received
                            if job["cancelled"]:
                                raise DownloadError("cancelled")
            if received < MIN_AUDIO_BYTES or (total and received != total):
                raise DownloadError("incomplete")
            facts = _facts(part)
            if facts["durationMs"] < job["minDurationMs"]:
                raise DownloadError("too_short")
            if facts["format"] in loudness.WRITABLE:
                try:
                    gains = loudness.measure_audio(part)
                    loudness.write_tags(part, facts["format"], gains, track_only=True)
                except Exception as error:
                    LOGGER.warning("LOUD-DOWNLOAD-FAIL %s %s", job["filename"], loudness._reason(error))
            with self._lock:
                job.update(facts)
            stem, suffix = job["filename"].rsplit(".", 1)[0], facts["format"]
            final = directory / f"{stem}.{suffix}"
            for number in range(2, 100):
                if not final.exists():
                    break
                final = directory / f"{stem} ({number}).{suffix}"
            else:
                raise DownloadError("name_taken")
            os.replace(part, final)
        finally:
            part.unlink(missing_ok=True)
        self._own(final)
        sidecars = []
        if job["lyrics"] and job["lyrics"].strip():
            sidecars.append((final.with_suffix(".lrc"), job["lyrics"].strip().encode("utf-8")))
        if job["cover"]:
            sidecars.append((final.with_suffix(_cover_suffix(job["cover"])), job["cover"]))
        for path, data in sidecars:
            try:
                if path.is_symlink():
                    continue
                path.write_bytes(data)
                self._own(path)
            except OSError:
                pass
        return str(final.relative_to(self._root))
