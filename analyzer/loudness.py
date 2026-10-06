"""ReplayGain 2.0 at -18 LUFS, measured without changing existing music.

Album gain uses a duration-weighted energy average of track loudness. It can
differ by a few tenths of a dB from measuring the concatenated album.
"""

import errno
import fcntl
import gzip
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import ID3, ID3NoHeaderError, TXXX
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4, MP4FreeForm

from .log import LOGGER
from .scan import AUDIO_EXTENSIONS


WRITABLE = {"flac", "mp3", "m4a"}
_MEASURE_RETRY_SECONDS = 30
_FOLLOWUP_SECONDS = (45, 150, 420)
TAG_NAMES = {
    "track_gain": "REPLAYGAIN_TRACK_GAIN", "track_peak": "REPLAYGAIN_TRACK_PEAK",
    "album_gain": "REPLAYGAIN_ALBUM_GAIN", "album_peak": "REPLAYGAIN_ALBUM_PEAK",
}
MP4_PREFIX = "----:com.apple.iTunes:"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|inf|nan)"
_COLUMNS = (
    "path", "size", "mtime", "format", "duration", "album_key", "track_gain",
    "track_peak", "album_gain", "album_peak", "source", "error", "write_error", "updated",
)


class LoudnessError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _reason(error):
    if isinstance(error, LoudnessError):
        return error.code
    if isinstance(error, OSError):
        reasons = {
            errno.ENAMETOOLONG: "name_too_long",
            errno.EACCES: "permission_denied", errno.EPERM: "permission_denied",
            errno.ENOSPC: "no_space", errno.EDQUOT: "no_space",
            errno.EROFS: "read_only",
        }
        return reasons.get(error.errno, type(error).__name__)
    return type(error).__name__


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path, document, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".xiyue-loud-", delete=False) as file:
            temporary = Path(file.name)
        opener = gzip.open if compressed else open
        with opener(temporary, "wt", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _acquire(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    lock = open(data / "loudness.lock", "a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise LoudnessError("busy") from None
    except Exception:
        lock.close()
        raise
    return lock


@contextmanager
def _database(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(data / "loudness.sqlite")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("""
            CREATE TABLE IF NOT EXISTS loudness(
                path TEXT PRIMARY KEY, size INTEGER, mtime REAL, format TEXT,
                duration REAL, album_key TEXT, track_gain REAL, track_peak REAL,
                album_gain REAL, album_peak REAL, source TEXT, error TEXT,
                write_error TEXT, updated REAL
            )
        """)
        connection.commit()
        yield connection


def _save_row(connection, row):
    connection.execute(
        f"INSERT INTO loudness ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)}) "
        "ON CONFLICT(path) DO UPDATE SET "
        + ", ".join(f"{name}=excluded.{name}" for name in _COLUMNS if name != "path"),
        [row.get(name) for name in _COLUMNS],
    )
    connection.commit()


def _rows(data):
    path = Path(data) / "loudness.sqlite"
    if not path.exists():
        return []
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='loudness'").fetchone():
            return []
        return [dict(row) for row in connection.execute("SELECT * FROM loudness ORDER BY path")]


def parse_tag(value, peak=False):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    try:
        number = float(re.sub(r"\s*db\s*$", "", str(value).strip(), flags=re.I))
    except ValueError:
        return None
    low, high = (0, 10) if peak else (-60, 60)
    return number if math.isfinite(number) and low <= number <= high else None


def read_gains(path, format):
    """Reads the four tags, including an ID3-only fixture without MPEG frames."""
    try:
        if format == "mp3":
            raw = {frame.desc.upper(): frame.text for frame in ID3(path).getall("TXXX")}
        elif format == "m4a":
            raw = {
                key[len(MP4_PREFIX):].upper(): value for key, value in MP4(path).items()
                if key.casefold().startswith(MP4_PREFIX.casefold())
            }
        elif format == "flac":
            raw = {key.upper(): value for key, value in FLAC(path).items()}
        else:
            raw = {}
    except (OSError, mutagen.MutagenError):
        raw = {}
    return {field: parse_tag(raw.get(name), peak=field.endswith("peak")) for field, name in TAG_NAMES.items()}


def _metadata(path):
    audio = mutagen.File(path, easy=True)
    return {
        name: list(audio.get(name, [])) if audio is not None else []
        for name in ("title", "artist", "album")
    }


def _describe(path, relative):
    duration, album = None, ""
    try:
        audio = mutagen.File(path)
        length = getattr(getattr(audio, "info", None), "length", None)
        if length is not None and math.isfinite(length) and length > 0:
            duration = float(length)
    except (OSError, mutagen.MutagenError):
        pass
    try:
        albums = _metadata(path)["album"]
        album = albums[0].strip().casefold() if albums else ""
    except (OSError, mutagen.MutagenError):
        pass
    return duration, f"{Path(relative).parent.as_posix()}\x1f{album}" if album else None


def measure_audio(path):
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-threads", "1", "-i", str(path),
             "-map", "0:a:0", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise LoudnessError("timeout") from None
    except FileNotFoundError:
        raise LoudnessError("ffmpeg_missing") from None
    if result.returncode:
        raise LoudnessError(f"ffmpeg_exit_{result.returncode}")
    stderr = result.stderr
    if "Integrated loudness:" not in stderr:
        raise LoudnessError("no_summary")
    summary = stderr.rsplit("Integrated loudness:", 1)[1]
    integrated = re.search(rf"\bI:\s*({_NUMBER})\s+LUFS", summary, re.I)
    peak = re.search(rf"True peak:\s*Peak:\s*({_NUMBER})\s+dBFS", summary, re.I)
    if integrated is None or peak is None:
        raise LoudnessError("no_summary")
    loudness, peak_db = float(integrated[1]), float(peak[1])
    if loudness <= -69.9:
        raise LoudnessError("silent")
    if not math.isfinite(loudness) or not math.isfinite(peak_db):
        raise LoudnessError("no_summary")
    try:
        peak_linear = round(10 ** (peak_db / 20), 6)
    except OverflowError:
        raise LoudnessError("no_summary") from None
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    duration = None
    if times:
        hours, minutes, seconds = map(float, times[-1])
        duration = hours * 3600 + minutes * 60 + seconds
    return {"track_gain": round(-18 - loudness, 2), "track_peak": peak_linear, "duration": duration}


def _library_path(library, relative):
    path = Path(library) / relative
    if not path.resolve().is_relative_to(Path(library).resolve()):
        raise LoudnessError("outside_library")
    return path


def _measure_one(library, item, remeasure):
    path, relative, stat = item
    row = {
        "path": relative, "size": stat.st_size if stat else None, "mtime": stat.st_mtime if stat else None,
        "format": path.suffix.lower().lstrip("."), "source": "measured", "updated": time.time(),
    }
    try:
        _library_path(library, relative)
        if stat is None:
            raise LoudnessError("unreadable")
        duration, album_key = _describe(path, relative)
        row.update(duration=duration, album_key=album_key)
        gains = read_gains(path, row["format"])
        if not remeasure and gains["track_gain"] is not None:
            row.update(gains, source="tag")
        else:
            measured = measure_audio(path)
            row.update(measured, duration=duration or measured["duration"])
    except Exception as error:
        row["error"] = _reason(error)
        LOGGER.warning("LOUD-FAIL %s %s", relative, row["error"])
    return row


def _albums(connection, library):
    groups = {}
    for row in connection.execute("SELECT * FROM loudness WHERE error IS NULL AND track_gain IS NOT NULL"):
        if row["album_key"]:
            groups.setdefault(row["album_key"], []).append(row)
    for rows in groups.values():
        weights = [row["duration"] or 1 for row in rows]
        energy = sum(weight * 10 ** ((-18 - row["track_gain"]) / 10) for row, weight in zip(rows, weights))
        gain = round(-18 - 10 * math.log10(energy / sum(weights)), 2)
        peaks = [row["track_peak"] for row in rows if row["track_peak"] is not None]
        peak = max(peaks) if peaks else None
        for row in rows:
            # A previous pass may have calculated album_gain for a tagged track.
            # Preserve only album gains actually present in the file.
            if row["source"] == "tag" and read_gains(
                _library_path(library, row["path"]), row["format"],
            )["album_gain"] is not None:
                continue
            connection.execute(
                "UPDATE loudness SET album_gain=?, album_peak=? WHERE path=?", (gain, peak, row["path"]),
            )
    connection.commit()


def _export(connection, out):
    tracks = []
    for row in connection.execute("SELECT * FROM loudness WHERE error IS NULL ORDER BY path"):
        track = {"path": row["path"], "trackGain": row["track_gain"], "source": row["source"]}
        for column, key in (("track_peak", "trackPeak"), ("album_gain", "albumGain"), ("album_peak", "albumPeak")):
            if row[column] is not None:
                track[key] = row[column]
        tracks.append(track)
    _atomic_json(Path(out) / "xiyue-loudness-v1.json.gz", {
        "version": 1, "reference": -18, "generatedAt": _now(), "tracks": tracks,
    }, compressed=True)


def _measure(library, data, out, remeasure, workers, progress):
    started = time.monotonic()
    library = Path(library)
    if not library.is_dir() or not os.access(library, os.R_OK | os.X_OK):
        raise LoudnessError("no_library")
    with _database(data) as connection:
        cached = {row["path"]: row for row in connection.execute("SELECT * FROM loudness")}
        present, pending = set(), []
        def walk_error(error):
            raise error

        for directory, directories, filenames in os.walk(library, onerror=walk_error):
            directories[:] = sorted(name for name in directories if not name.startswith((".", "@", "#")))
            for name in sorted(filenames):
                path = Path(directory) / name
                if name.startswith(".") or path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                relative = path.relative_to(library).as_posix()
                present.add(relative)
                try:
                    stat = path.stat()
                except OSError:
                    stat = None
                previous = cached.get(relative)
                if not remeasure and stat is not None and previous is not None and not previous["error"] and (
                    previous["size"], previous["mtime"]
                ) == (stat.st_size, stat.st_mtime):
                    continue
                pending.append((path, relative, stat))
        for removed in cached.keys() - present:
            connection.execute("DELETE FROM loudness WHERE path=?", (removed,))
        connection.commit()
        LOGGER.info("LOUD-START pending=%s remeasure=%s", len(pending), remeasure)
        progress(0, len(pending))
        counts = {"measured": 0, "fromTags": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_measure_one, library, item, remeasure) for item in pending]
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                _save_row(connection, row)
                counts["failed" if row.get("error") else "fromTags" if row["source"] == "tag" else "measured"] += 1
                progress(done, len(pending))
        _albums(connection, library)
        _export(connection, out)
    counts["seconds"] = round(time.monotonic() - started, 2)
    LOGGER.info("LOUD-END measured=%s fromTags=%s failed=%s seconds=%.2f", *counts.values())
    return counts


def write_tags(path, format, gains, track_only=False):
    fields = list(TAG_NAMES)[:2] if track_only else list(TAG_NAMES)
    names = {TAG_NAMES[field] for field in fields}
    values = {
        TAG_NAMES[field]: f"{gains[field]:.6f}" if field.endswith("peak") else f"{gains[field]:+.2f} dB"
        for field in fields if gains.get(field) is not None
    }
    if format == "flac":
        audio = FLAC(path)
        for key in list(audio):
            if key.upper() in names:
                del audio[key]
        for key, value in values.items():
            audio[key] = [value]
        audio.save()
    elif format == "mp3":
        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        for frame in tags.getall("TXXX"):
            if frame.desc.upper() in names:
                del tags[frame.HashKey]
        for key, value in values.items():
            tags.add(TXXX(encoding=3, desc=key, text=[value]))
        tags.save(path)
    elif format == "m4a":
        audio = MP4(path)
        if audio.tags is None:
            audio.add_tags()
        for key in list(audio):
            if key.casefold().startswith(MP4_PREFIX.casefold()) and key[len(MP4_PREFIX):].upper() in names:
                del audio[key]
        for key, value in values.items():
            audio[MP4_PREFIX + key] = [MP4FreeForm(value.encode("utf-8"))]
        audio.save()
    else:
        raise LoudnessError("unsupported")


def _audio_signature(path, format):
    info = {"flac": FLAC, "mp3": MP3, "m4a": MP4}[format](path).info
    names = ("total_samples", "sample_rate", "md5_signature") if format == "flac" else ("length", "sample_rate", "channels")
    return {name: getattr(info, name) for name in names}


def _validate(path, format, gains, signature, metadata):
    readback = read_gains(path, format)
    for field in TAG_NAMES:
        expected, actual = gains.get(field), readback[field]
        if (expected is None) != (actual is None):
            return False
        if expected is not None and abs(expected - actual) > 0.005:
            return False
    actual = _audio_signature(path, format)
    for name, value in signature.items():
        if name == "length":
            if abs(actual[name] - value) >= 0.01:
                return False
        elif name != "md5_signature" or value:
            if actual[name] != value:
                return False
    return _metadata(path) == metadata


def _update_taste_cache(data, relative, old, new):
    path = Path(data) / "cache.sqlite"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection:
        for (cached_path,) in connection.execute(
            "SELECT path FROM tracks WHERE size=? AND mtime=?", (old.st_size, old.st_mtime),
        ).fetchall():
            if relative == cached_path or relative.endswith("/" + cached_path):
                connection.execute(
                    "UPDATE tracks SET size=?, mtime=? WHERE path=? AND size=? AND mtime=?",
                    (new.st_size, new.st_mtime, cached_path, old.st_size, old.st_mtime),
                )
        connection.commit()


def _write_one(library, data, connection, row):
    temporary = None
    path = _library_path(library, row["path"])
    old = path.stat()
    if (old.st_size, old.st_mtime) != (row["size"], row["mtime"]):
        connection.execute("UPDATE loudness SET size=NULL WHERE path=?", (row["path"],))
        connection.commit()
        return "changed"
    try:
        if path.is_symlink():
            raise LoudnessError("symlink")
        signature, metadata = _audio_signature(path, row["format"]), _metadata(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".xiyue-rg-", suffix=path.suffix, delete=False) as file:
            temporary = Path(file.name)
        shutil.copyfile(path, temporary)
        write_tags(temporary, row["format"], row)
        if not _validate(temporary, row["format"], row, signature, metadata):
            raise LoudnessError("validation_failed")
        try:
            os.chown(temporary, old.st_uid, old.st_gid)
        except OSError:
            pass
        os.chmod(temporary, old.st_mode & 0o7777)
        os.utime(temporary, ns=(old.st_atime_ns, old.st_mtime_ns))
        os.replace(temporary, path)
        new = path.stat()
        connection.execute(
            "UPDATE loudness SET source='written', size=?, mtime=?, write_error=NULL, updated=? WHERE path=?",
            (new.st_size, new.st_mtime, time.time(), row["path"]),
        )
        connection.commit()
        _update_taste_cache(data, row["path"], old, new)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return None


def _write(library, data, out, music, paths, progress):
    result = {"written": [], "failed": [], "skipped": []}
    with _database(data) as connection:
        rows = {row["path"]: dict(row) for row in connection.execute("SELECT * FROM loudness ORDER BY path")}
        if paths is None:
            paths = [path for path, row in rows.items() if _pending(row)]
        else:
            paths = list(dict.fromkeys(paths))
        progress(0, len(paths))
        for done, relative in enumerate(paths, 1):
            row = rows.get(relative)
            reason = (
                "unknown" if row is None else "unsupported" if row["format"] not in WRITABLE
                else "not_measured" if not _pending(row) else None
            )
            try:
                if reason is None:
                    reason = _write_one(library, data, connection, row)
                if reason is not None:
                    result["skipped"].append({"path": relative, "reason": reason})
                else:
                    result["written"].append(relative)
                    LOGGER.info("LOUD-WRITE %s ok", relative)
            except Exception as error:
                reason = _reason(error)
                connection.execute("UPDATE loudness SET write_error=? WHERE path=?", (reason, relative))
                connection.commit()
                result["failed"].append({"path": relative, "reason": reason})
                LOGGER.warning("LOUD-WRITE %s fail %s", relative, reason)
            progress(done, len(paths))
        _export(connection, out)
    return result


def _pending(row):
    return row["source"] == "measured" and row["format"] in WRITABLE and not row["error"] and row["track_gain"] is not None


def _read_state(data):
    try:
        return json.loads((Path(data) / "loudness-state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "idle", "done": 0, "total": 0, "lastMeasure": None, "lastWrite": None}


def _begin(data, operation):
    state = _read_state(data)
    state.update(state=operation, done=0, total=0)
    _atomic_json(Path(data) / "loudness-state.json", state)
    return state


def _operate(operation, library, data, out, music=None, paths=None, remeasure=False, workers=2, progress=None):
    state = _begin(data, operation)

    def report(done, total):
        state.update(done=done, total=total)
        _atomic_json(Path(data) / "loudness-state.json", state)
        if progress is not None:
            progress(done, total)

    try:
        if operation == "measuring":
            result = _measure(library, data, out, remeasure, workers, report)
            state["lastMeasure"] = {"at": _now(), **result}
        else:
            result = _write(library, data, out, music, paths, report)
            state["lastWrite"] = {"at": _now(), **result, "written": len(result["written"])}
        return result
    finally:
        state["state"] = "idle"
        _atomic_json(Path(data) / "loudness-state.json", state)


def measure(library, data, out, remeasure=False, workers=2, progress=None):
    with _acquire(data):
        return _operate("measuring", library, data, out, remeasure=remeasure, workers=workers, progress=progress)


def write(library, data, out, music, paths=None):
    with _acquire(data):
        return _operate("writing", library, data, out, music=music, paths=paths)


class Loudness:
    """HTTP jobs reserve the same OS lock before returning 202, including against CLI jobs."""

    def __init__(self, library, data, out, music):
        self.library, self.data, self.out, self.music = library, data, out, music
        self._measure_request_lock = threading.Lock()
        self._measure_retry = None
        self._followup_lock = threading.Lock()
        self._followup_files: dict[str, tuple | None] = {}
        self._followup_timer = None
        self._followup_generation = None
        self._followup_started = 0

    def request_measure(self, path=None):
        """Incrementally measure after a download, coalescing retries while busy."""
        if path:
            with self._followup_lock:
                self._followup_files[path] = self._followup_stat(path)
                if self._followup_timer is not None:
                    self._followup_timer.cancel()
                self._followup_generation = object()
                self._followup_started = time.monotonic()
                self._schedule_followup(0, self._followup_generation)
        with self._measure_request_lock:
            if self._measure_retry is None:
                self._request_measure(20)

    def _followup_stat(self, path):
        try:
            stat = os.stat(Path(self.library) / path)
        except OSError:
            return None
        return stat.st_size, stat.st_mtime

    def _schedule_followup(self, index, generation):
        delay = max(0, self._followup_started + _FOLLOWUP_SECONDS[index] - time.monotonic())
        self._followup_timer = threading.Timer(delay, self._check_followup, args=(index, generation))
        self._followup_timer.daemon = True
        self._followup_timer.start()

    def _check_followup(self, index, generation):
        with self._followup_lock:
            # A cancelled timer may already have entered its callback.
            if generation is not self._followup_generation:
                return
            changed = 0
            for path, previous in self._followup_files.items():
                current = self._followup_stat(path)
                if current != previous:
                    changed += 1
                    self._followup_files[path] = current
            total = len(self._followup_files)
        if changed:
            LOGGER.info("LOUD-FOLLOWUP run changed=%s files=%s", changed, total)
            try:
                self.request_measure()
            except Exception:
                LOGGER.warning("LOUD-AFTER-DOWNLOAD failed")
        else:
            LOGGER.info("LOUD-FOLLOWUP skipped files=%s", total)
        with self._followup_lock:
            if generation is not self._followup_generation:
                return
            if index + 1 < len(_FOLLOWUP_SECONDS):
                self._schedule_followup(index + 1, generation)
            else:
                self._followup_files.clear()
                self._followup_timer = None

    def _request_measure(self, retries_left):
        try:
            self.start("measuring", remeasure=False)
        except LoudnessError as error:
            if error.code != "busy":
                raise
            if retries_left == 0:
                LOGGER.warning("LOUD-AFTER-DOWNLOAD gave_up")
                return
            LOGGER.info("LOUD-AFTER-DOWNLOAD busy_retry")
            self._measure_retry = threading.Timer(
                _MEASURE_RETRY_SECONDS, self._retry_measure, args=(retries_left - 1,),
            )
            self._measure_retry.daemon = True
            self._measure_retry.start()
        else:
            LOGGER.info("LOUD-AFTER-DOWNLOAD started")

    def _retry_measure(self, retries_left):
        with self._measure_request_lock:
            self._measure_retry = None
            try:
                self._request_measure(retries_left)
            except Exception:
                LOGGER.warning("LOUD-AFTER-DOWNLOAD failed")

    def snapshot(self):
        state = _read_state(self.data)
        try:
            with _acquire(self.data):
                state["state"] = "idle"
        except LoudnessError:
            pass
        rows = _rows(self.data)
        good = [row for row in rows if not row["error"]]
        return {
            **state,
            "counts": {
                "fromTags": sum(row["source"] == "tag" for row in good),
                "measured": sum(row["source"] == "measured" for row in good),
                "written": sum(row["source"] == "written" for row in good),
                "failed": len(rows) - len(good),
                "unsupported": sum(row["format"] not in WRITABLE for row in good),
            },
            "pending": [row["path"] for row in rows if _pending(row)],
            "problems": [
                {"path": row["path"], "stage": "measure" if row["error"] else "write",
                 "reason": row["error"] or row["write_error"]}
                for row in rows if row["error"] or row["write_error"]
            ][:200],
        }

    def start(self, operation, **options):
        lock = _acquire(self.data)
        try:
            _begin(self.data, operation)

            def work():
                try:
                    with lock:
                        _operate(operation, self.library, self.data, self.out, music=self.music, **options)
                except Exception as error:
                    LOGGER.warning("LOUD-FAIL %s %s", operation, _reason(error))

            threading.Thread(target=work, name="loudness", daemon=True).start()
        except Exception:
            lock.close()
            raise
