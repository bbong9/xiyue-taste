"""The owner's LX source scripts, kept on the NAS so the family's phones never see them.

Scripts live in <data>/sources/<id>.js (0600) and their details in <data>/sources/index.json.
No method here returns or logs a script's text; a log line names a source by its id only.
"""

import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

from .log import LOGGER
from .lxhost import RunnerError

MAX_SCRIPT_BYTES = 1_048_576
DEFAULT_QUOTA = 1000
MAX_QUOTA = 100_000
SOURCE_ID = re.compile(r"[0-9a-f]{16}")
# Swift's Character.isNewline ("\r\n" is one break) and CharacterSet.whitespaces (Zs and tab),
# which iOS LXMusicScriptInspector splits and trims the header comment with.
_NEWLINE = re.compile("\r\n|[\n\x0b\x0c\r\x85  ]")
_TRIM = "\t \xa0              　"
_WHITE_SPACE = frozenset(_TRIM + "\n\x0b\x0c\r\x85  ")
# A script blamed this many times for hanging or killing the runner is not loaded again on its own.
QUARANTINE_AFTER = 2
# The runner was not there to ask; nothing is known about the script.
_RUNNER_CODES = ("runner_unavailable", "cancelled")


class SourceError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _header_value(key, comment):
    marker = "@" + key
    for line in _NEWLINE.split(comment):
        trimmed = line.strip(_TRIM)
        at = trimmed.find(marker)
        if at < 0 or not all(c == "*" or c in _WHITE_SPACE for c in trimmed[:at]):
            continue
        return trimmed[at + len(marker):].strip(_TRIM)
    return ""


def inspect(data):
    """iOS LXMusicScriptInspector: at most 1 MiB of UTF-8, with @name and @version in the first
    /* */ comment. homepage is read the same way, for the panel only."""
    if len(data) > MAX_SCRIPT_BYTES:
        raise SourceError("script_too_large", 413)
    if not data:
        raise SourceError("invalid_script")
    try:
        script = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SourceError("invalid_script") from None
    start = script.find("/*")
    end = script.find("*/", start + 2) if start >= 0 else -1
    comment = script[start:end + 2] if end >= 0 else ""
    name, version = _header_value("name", comment), _header_value("version", comment)
    if not name or not version:
        raise SourceError("invalid_script")
    return {
        "name": name[:64],
        "description": _header_value("description", comment)[:256],
        "version": version[:32],
        "author": _header_value("author", comment)[:64],
        "homepage": _header_value("homepage", comment)[:1024],
    }


def write_private(path, data):
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        os.fchmod(file.fileno(), 0o600)
        file.write(data)
    os.replace(temporary, path)


def _entry(saved):
    """One index.json entry as stored, or None when it is not one."""
    if not isinstance(saved, dict) or not isinstance(saved.get("id"), str) or not SOURCE_ID.fullmatch(saved["id"]):
        return None
    entry = {"id": saved["id"]}
    for name in ("name", "version", "author", "description", "homepage", "sha256", "loadError"):
        entry[name] = saved.get(name) if isinstance(saved.get(name), str) else ""
    entry["enabled"] = saved.get("enabled") is True
    quota = saved.get("dailyQuota")
    entry["dailyQuota"] = quota if type(quota) is int and 0 <= quota <= MAX_QUOTA else DEFAULT_QUOTA
    order = saved.get("order")
    entry["order"] = order if type(order) is int else 0
    platforms = saved.get("platforms")
    entry["platforms"] = {
        platform: list(tiers) for platform, tiers in platforms.items()
        if isinstance(platform, str) and isinstance(tiers, list) and all(isinstance(tier, str) for tier in tiers)
    } if isinstance(platforms, dict) else {}
    added = saved.get("addedAt")
    entry["addedAt"] = added if type(added) is int else 0
    return entry


class SourceStore:
    """The scripts and their settings; loads the enabled ones into the runner.

    Changes go one at a time (a load can take the script's 12 s to init); reading the list never
    waits for them.
    """

    def __init__(self, data, runner):
        self._directory = Path(data) / "sources"
        self._index = self._directory / "index.json"
        self._runner = runner
        self._lock = threading.Lock()
        self._ops = threading.Lock()
        # Sources whose loads hung or killed the runner: left alone until switched on again.
        self._suspects = {}
        self._quarantined = set()
        self._directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._directory, 0o700)
        except OSError:
            pass
        try:
            saved = json.loads(self._index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = []
        self._entries = {}
        dropped = 0
        for item in saved if isinstance(saved, list) else []:
            entry = _entry(item)
            if entry is not None and entry["id"] not in self._entries and self._script_path(entry["id"]).is_file():
                self._entries[entry["id"]] = entry
            else:
                dropped += 1
        self._renumber()
        if dropped:
            LOGGER.warning("SOURCE-INDEX dropped=%s", dropped)

    def _script_path(self, source_id):
        return self._directory / f"{source_id}.js"

    def _sorted(self):
        return sorted(self._entries.values(), key=lambda entry: (entry["order"], entry["addedAt"], entry["id"]))

    def _renumber(self):
        for position, entry in enumerate(self._sorted()):
            entry["order"] = position

    def _save(self):
        # Under self._lock.
        payload = json.dumps(self._sorted(), ensure_ascii=False, indent=1).encode("utf-8")
        write_private(self._index, payload)

    def list(self):
        """Every source's details, with whether it is loaded right now; never a script."""
        loaded = self._runner.loaded()
        with self._lock:
            return [{**entry, "platforms": dict(entry["platforms"]), "loaded": entry["id"] in loaded}
                    for entry in self._sorted()]

    def get(self, source_id):
        with self._lock:
            entry = self._entries.get(source_id)
            return None if entry is None else {**entry, "platforms": dict(entry["platforms"])}

    def add(self, script):
        """Stores a new script and loads it. A script that fails to load is kept, switched off,
        with the reason."""
        if not isinstance(script, str):
            raise SourceError("invalid_script")
        try:
            data = script.encode("utf-8")
        except UnicodeEncodeError:
            raise SourceError("invalid_script") from None
        meta = inspect(data)
        digest = hashlib.sha256(data).hexdigest()
        with self._ops:
            with self._lock:
                if any(entry["sha256"] == digest for entry in self._entries.values()):
                    raise SourceError("duplicate", 409)
                source_id = secrets.token_hex(8)
                last = max((entry["order"] for entry in self._entries.values()), default=-1)
            path = self._script_path(source_id)
            write_private(path, data)
            entry = {
                "id": source_id, **meta, "sha256": digest, "enabled": False, "dailyQuota": DEFAULT_QUOTA,
                "order": last + 1, "platforms": {}, "addedAt": int(time.time()), "loadError": "",
            }
            try:
                platforms, error = self._load(entry, script)
                entry.update(enabled=error is None, platforms=platforms or {}, loadError=error or "")
                with self._lock:
                    self._entries[source_id] = entry
                    self._save()
            except BaseException:
                with self._lock:
                    self._entries.pop(source_id, None)
                self._runner.unload(source_id)
                path.unlink(missing_ok=True)
                raise
        LOGGER.info("SOURCE-ADD id=%s ok=%s error=%s", source_id, "是" if error is None else "否", error or "")
        return self.get(source_id)

    def update(self, source_id, body):
        """Switches a source on or off, sets its daily quota, or moves it to another place."""
        enabled, quota, order = body.get("enabled"), body.get("dailyQuota"), body.get("order")
        if (
            (enabled is not None and not isinstance(enabled, bool))
            or (quota is not None and (type(quota) is not int or not 0 <= quota <= MAX_QUOTA))
            or (order is not None and (type(order) is not int or order < 0))
            or (enabled is None and quota is None and order is None)
        ):
            raise SourceError("bad_request")
        with self._ops:
            current = self.get(source_id)
            if current is None:
                raise SourceError("not_found", 404)
            changes = {}
            if enabled is True and (not current["enabled"] or source_id not in self._runner.loaded()):
                self._suspects.pop(source_id, None)
                self._quarantined.discard(source_id)
                platforms, error = self._load(current, self._read_script(source_id))
                changes.update(enabled=error is None, loadError=error or "")
                if platforms is not None:
                    changes["platforms"] = platforms
            elif enabled is False:
                self._runner.unload(source_id)
                changes["enabled"] = False
            if quota is not None:
                changes["dailyQuota"] = quota
            with self._lock:
                entry = self._entries.get(source_id)
                if entry is None:
                    raise SourceError("not_found", 404)
                entry.update(changes)
                if order is not None:
                    others = [item for item in self._sorted() if item is not entry]
                    others.insert(min(order, len(others)), entry)
                    for position, item in enumerate(others):
                        item["order"] = position
                self._save()
        LOGGER.info(
            "SOURCE-UPDATE id=%s enabled=%s quota=%s order=%s error=%s", source_id,
            "-" if enabled is None else ("是" if changes.get("enabled") else "否"),
            "-" if quota is None else quota, "-" if order is None else order, changes.get("loadError", ""),
        )
        return self.get(source_id)

    def delete(self, source_id):
        with self._ops:
            with self._lock:
                if source_id not in self._entries:
                    raise SourceError("not_found", 404)
            self._runner.unload(source_id)
            with self._lock:
                del self._entries[source_id]
                self._renumber()
                self._save()
            self._suspects.pop(source_id, None)
            self._quarantined.discard(source_id)
            self._script_path(source_id).unlink(missing_ok=True)
        LOGGER.info("SOURCE-DELETE id=%s", source_id)

    def ensure_loaded(self, source_id):
        """For a test from the panel: an enabled source that is not loaded is loaded now."""
        with self._ops:
            entry = self.get(source_id)
            if entry is None:
                raise SourceError("not_found", 404)
            if source_id in self._runner.loaded():
                return entry
            if not entry["enabled"]:
                raise SourceError("disabled", 409)
            self._suspects.pop(source_id, None)
            self._quarantined.discard(source_id)
            error = self._reload(entry)
            if source_id not in self._runner.loaded():
                raise SourceError(error or "not_loaded", 409)
            return self.get(source_id)

    def reload_all(self):
        """After the runner (re)starts: loads every enabled source, one by one."""
        loaded = failed = 0
        with self._lock:
            ids = [entry["id"] for entry in self._sorted()]
        for source_id in ids:
            with self._ops:
                entry = self.get(source_id)
                if entry is None or not entry["enabled"] or source_id in self._quarantined:
                    continue
                if source_id in self._runner.loaded():
                    loaded += 1
                    continue
                error = self._reload(entry)
            if error in _RUNNER_CODES:
                # The runner is gone again; its next start loads everything.
                return
            if error is None:
                loaded += 1
            else:
                failed += 1
        LOGGER.info("SOURCE-RELOAD loaded=%s failed=%s", loaded, failed)

    def retry_failed(self):
        """Every so often: enabled sources whose last load failed get another try."""
        with self._lock:
            ids = [entry["id"] for entry in self._sorted() if entry["enabled"] and entry["loadError"]]
        for source_id in ids:
            with self._ops:
                entry = self.get(source_id)
                if (
                    entry is None or not entry["enabled"] or source_id in self._quarantined
                    or source_id in self._runner.loaded()
                ):
                    continue
                if self._reload(entry) in _RUNNER_CODES:
                    return

    def _reload(self, entry):
        """Loads an enabled source again; it stays enabled whatever happens. Returns the error."""
        source_id = entry["id"]
        try:
            script = self._read_script(source_id)
        except SourceError as failure:
            platforms, error = None, failure.code
        else:
            platforms, error = self._load(entry, script)
        if error in _RUNNER_CODES:
            return error
        with self._lock:
            current = self._entries.get(source_id)
            if current is not None:
                current["loadError"] = error or ""
                if platforms is not None:
                    current["platforms"] = platforms
                self._save()
        return error

    def _read_script(self, source_id):
        try:
            return self._script_path(source_id).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            raise SourceError("script_missing", 500) from None

    def _load(self, entry, script):
        """Returns (platforms, None) or (None, error code)."""
        meta = {name: entry[name] for name in ("name", "description", "version", "author")}
        # iOS hands lx_setup an empty homepage.
        meta["homepage"] = ""
        source_id = entry["id"]
        try:
            platforms = self._runner.load(source_id, script, meta)
        except RunnerError as error:
            if error.suspect:
                self._suspects[source_id] = self._suspects.get(source_id, 0) + 1
                if self._suspects[source_id] >= QUARANTINE_AFTER:
                    self._quarantined.add(source_id)
                    LOGGER.warning("SOURCE-QUARANTINE id=%s error=%s", source_id, error.code)
            return None, error.code
        self._suspects.pop(source_id, None)
        return platforms, None
