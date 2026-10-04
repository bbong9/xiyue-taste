"""Opaque personal JSON documents and an append-only listening log per account."""

import bisect
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from .log import LOGGER
from .sources import write_private

_ACCOUNT = re.compile(r"(?:owner|[0-9a-f]{16})")
_EVENT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class PersonalError(Exception):
    def __init__(self, code, status=400, **details):
        super().__init__(code)
        self.code, self.status, self.details = code, status, details


def _encode(value):
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise PersonalError("bad_request") from None


class PersonalStore:
    def __init__(self, data, clock=time.time, max_events=50_000, kept_events=40_000):
        self._root = Path(data) / "accounts"
        self._clock = clock
        self._max_events, self._kept_events = max_events, kept_events
        self._lock = threading.Lock()
        self._locks, self._indexes = {}, {}

    def _account_lock(self, account):
        if not isinstance(account, str) or not _ACCOUNT.fullmatch(account):
            raise PersonalError("bad_request")
        with self._lock:
            return self._locks.setdefault(account, threading.Lock())

    def _directory(self, account):
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)
        folder = self._root / account
        folder.mkdir(exist_ok=True, mode=0o700)
        folder.chmod(0o700)
        return folder

    def _index(self, account):
        index = self._indexes.get(account)
        if index is None:
            index = {"ids": set(), "positions": [], "lastSeq": 0}
            path = self._root / account / "listening.jsonl"
            if path.exists():
                skipped, cut = 0, None
                with path.open("rb") as file:
                    while True:
                        position = file.tell()
                        line = file.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            # A write cut short: drop it, or the next append would run into it.
                            cut = position
                            break
                        try:
                            row = json.loads(line)
                            seq, event_id = row["seq"], row["id"]
                            if type(seq) is not int or seq <= index["lastSeq"] or not isinstance(event_id, str):
                                raise ValueError("Invalid row")
                        except (ValueError, KeyError, TypeError):
                            skipped += 1
                            continue
                        index["ids"].add(event_id)
                        index["positions"].append((seq, position))
                        index["lastSeq"] = seq
                if cut is not None:
                    os.truncate(path, cut)
                if skipped or cut is not None:
                    LOGGER.warning(
                        "PERSONAL-INDEX account=%s skipped=%s cut=%s", account, skipped, "是" if cut is not None else "否",
                    )
            self._indexes[account] = index
        return index

    def listening_status(self, account):
        with self._account_lock(account):
            index = self._index(account)
            return {"total": len(index["positions"]), "lastSeq": index["lastSeq"]}

    def append(self, account, events):
        if not isinstance(events, list) or len(events) > 500:
            raise PersonalError("bad_request")
        for row in events:
            if (
                not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not _EVENT_ID.fullmatch(row["id"]) or not isinstance(row.get("event"), dict)
                or len(_encode(row["event"])) > 4096
            ):
                raise PersonalError("bad_request")
        with self._account_lock(account):
            index = self._index(account)
            path = self._directory(account) / "listening.jsonl"
            accepted = duplicates = 0
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "ab") as file:
                os.fchmod(file.fileno(), 0o600)
                for row in events:
                    if row["id"] in index["ids"]:
                        duplicates += 1
                        continue
                    seq = index["lastSeq"] + 1
                    position = file.tell()
                    file.write(_encode({"seq": seq, "id": row["id"], "at": int(self._clock()), "event": row["event"]}) + b"\n")
                    index["ids"].add(row["id"])
                    index["positions"].append((seq, position))
                    index["lastSeq"] = seq
                    accepted += 1
            if len(index["positions"]) > self._max_events:
                with path.open("rb") as file:
                    file.seek(index["positions"][-self._kept_events][1])
                    kept = file.read()
                write_private(path, kept)
                del self._indexes[account]
                index = self._index(account)
                LOGGER.info("PERSONAL-COMPACT account=%s kept=%s", account, len(index["positions"]))
            result = {
                "accepted": accepted, "duplicates": duplicates,
                "total": len(index["positions"]), "lastSeq": index["lastSeq"],
            }
        LOGGER.info(
            "PERSONAL-LISTENING account=%s accepted=%s duplicates=%s total=%s",
            account, accepted, duplicates, result["total"],
        )
        return result

    def listening(self, account, after=0, limit=1000):
        limit = min(2000, max(1, limit))
        with self._account_lock(account):
            index = self._index(account)
            start = bisect.bisect_left(index["positions"], (after + 1, 0))
            positions = index["positions"][start:start + limit]
            events = []
            if positions:
                with (self._root / account / "listening.jsonl").open("rb") as file:
                    for _, position in positions:
                        file.seek(position)
                        row = json.loads(file.readline())
                        events.append({key: row[key] for key in ("seq", "id", "event")})
            return {
                "events": events, "lastSeq": events[-1]["seq"] if events else index["lastSeq"],
                "hasMore": start + len(positions) < len(index["positions"]),
            }

    def _collections(self, account):
        path = self._root / account / "collections.json"
        if not path.exists():
            return {"revision": 0, "updatedAt": None, "summary": None, "document": None}
        return json.loads(path.read_text(encoding="utf-8"))

    def collections(self, account):
        with self._account_lock(account):
            return self._collections(account)

    def panel_status(self, account):
        with self._account_lock(account):
            index = self._index(account)
            collections = self._collections(account)
            last_sync = collections["updatedAt"]
            if index["positions"]:
                with (self._root / account / "listening.jsonl").open("rb") as file:
                    file.seek(index["positions"][-1][1])
                    last_sync = max(last_sync or 0, json.loads(file.readline())["at"])
            return {
                "listening": {"total": len(index["positions"]), "lastSeq": index["lastSeq"]},
                "collections": {key: collections[key] for key in ("revision", "updatedAt", "summary")},
                "lastSyncAt": last_sync,
            }

    def save_collections(self, account, body):
        revision, document, summary = body.get("baseRevision"), body.get("document"), body.get("summary")
        if type(revision) is not int or revision < 0 or not isinstance(document, dict):
            raise PersonalError("bad_request")
        if summary is not None and (
            not isinstance(summary, dict) or set(summary) != {"liked", "playlists", "tracks"}
            or any(type(value) is not int or not 0 <= value <= 1_000_000 for value in summary.values())
        ):
            raise PersonalError("bad_request")
        size = len(_encode(body))
        if size > 6 * 1024 * 1024:
            raise PersonalError("too_large", 413)
        with self._account_lock(account):
            current = self._collections(account)
            if current["revision"] != revision:
                LOGGER.info(
                    "PERSONAL-COLLECTIONS conflict account=%s base=%s current=%s",
                    account, revision, current["revision"],
                )
                raise PersonalError("conflict", 409, revision=current["revision"], updatedAt=current["updatedAt"])
            result = {"revision": revision + 1, "updatedAt": int(self._clock())}
            write_private(self._directory(account) / "collections.json", _encode({
                **result, "summary": summary, "document": document,
            }))
        LOGGER.info("PERSONAL-COLLECTIONS account=%s revision=%s bytes=%s", account, result["revision"], size)
        return result

    def delete_account_data(self, account):
        with self._account_lock(account):
            folder = self._root / account
            if folder.exists():
                shutil.rmtree(folder)
            self._indexes.pop(account, None)
