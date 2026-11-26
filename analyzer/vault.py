"""Persistent library butler metadata & artwork vault stored under <data>/butler-vault.

Keeps owner-filled artist photos, artist bios, album covers, filled lyrics, and
Fly Music connection avatars on the NAS so app reinstalls and family devices
automatically restore them without depending on WebDAV folder layouts.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from .log import LOGGER
from .sources import write_private

VAULT_VERSION = 1
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_LYRICS_BYTES = 256 * 1024
MAX_BIO_CHARS = 8_000
MAX_KEY_CHARS = 256
MAX_LABEL_CHARS = 300
MAX_BATCH_ITEMS = 50
MAX_DELETE_KEYS = 500
MAX_LYRICS_BATCH_IDS = 100
MAX_TRACK_GUIDS = 500
MAX_ONLINE_ARTISTS = 50

_BLOB_ID = re.compile(r"[0-9a-f]{32}")
_KINDS = ("photos", "covers", "avatars", "lyrics")


class ButlerVaultError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _blob_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _image_content_type(data: bytes):
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _clean_key(value, limit=MAX_KEY_CHARS):
    if not isinstance(value, str):
        raise ButlerVaultError("bad_request")
    trimmed = value.strip()
    if not trimmed or len(trimmed) > limit or "\x00" in trimmed:
        raise ButlerVaultError("bad_request")
    return trimmed


def _clean_label(value, default="", limit=MAX_LABEL_CHARS):
    if value is None:
        return default
    if not isinstance(value, str) or "\x00" in value:
        raise ButlerVaultError("bad_request")
    trimmed = value.strip()
    if len(trimmed) > limit:
        raise ButlerVaultError("bad_request")
    return trimmed


def _clean_timestamp(value, fallback):
    if value is None:
        return int(fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ButlerVaultError("bad_request")
    ts = int(value)
    if ts < 0 or ts > 4_102_444_800:  # year 2100 sanity bound
        raise ButlerVaultError("bad_request")
    return ts


def _clean_string_list(value, max_items, max_chars):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > max_items:
        raise ButlerVaultError("bad_request")
    result = []
    for entry in value:
        if not isinstance(entry, str) or "\x00" in entry:
            raise ButlerVaultError("bad_request")
        trimmed = entry.strip()
        if not trimmed or len(trimmed) > max_chars:
            raise ButlerVaultError("bad_request")
        result.append(trimmed)
    return result


def _decode_image(value):
    if not isinstance(value, str) or not value:
        raise ButlerVaultError("bad_request")
    # Quick upper-bound check before base64 decoding.
    if len(value) > (MAX_IMAGE_BYTES * 4) // 3 + 16:
        raise ButlerVaultError("image_too_large", 413)
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ButlerVaultError("bad_image") from None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ButlerVaultError("image_too_large" if raw else "bad_image", 413 if raw else 400)
    content_type = _image_content_type(raw)
    if content_type is None:
        raise ButlerVaultError("unsupported_image")
    return raw, content_type


def _encode_json(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ButlerVaultError("bad_request") from None


def _empty_index():
    return {
        "version": VAULT_VERSION,
        "revision": 0,
        "updatedAt": 0,
        "photos": {},
        "bios": {},
        "covers": {},
        "lyrics": {},
        "avatars": {},
    }


class ButlerVaultStore:
    def __init__(self, data, clock=time.time):
        self._root = Path(data) / "butler-vault"
        self._clock = clock
        self._lock = threading.RLock()
        self._index = self._load_index()

    def _index_path(self) -> Path:
        return self._root / "index.json"

    def _blob_path(self, kind: str, blob_id: str) -> Path:
        ext = ".json" if kind == "lyrics" else ".img"
        return self._root / kind / (blob_id + ext)

    def _ensure_dir(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            folder.chmod(0o700)
        except OSError:
            pass

    def _load_index(self):
        path = self._index_path()
        if not path.exists():
            return _empty_index()
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                return _empty_index()
            base = _empty_index()
            base["revision"] = int(loaded.get("revision", 0)) if type(loaded.get("revision")) is int else 0
            base["updatedAt"] = int(loaded.get("updatedAt", 0)) if type(loaded.get("updatedAt")) is int else 0
            for section in ("photos", "bios", "covers", "lyrics", "avatars"):
                raw_section = loaded.get(section)
                if isinstance(raw_section, dict):
                    base[section] = raw_section
            return base
        except (OSError, ValueError, TypeError):
            LOGGER.warning("BUTLER-VAULT index unreadable; starting empty")
            return _empty_index()

    def _persist_index(self):
        self._ensure_dir(self._root)
        write_private(self._index_path(), _encode_json(self._index))

    def _counts(self):
        return {
            section: len(self._index[section])
            for section in ("photos", "bios", "covers", "lyrics", "avatars")
        }

    def manifest(self, since=None):
        with self._lock:
            revision = self._index["revision"]
            updated_at = self._index["updatedAt"]
            counts = self._counts()
            if since is not None and since == revision:
                return {
                    "version": VAULT_VERSION,
                    "unchanged": True,
                    "revision": revision,
                    "updatedAt": updated_at,
                    "counts": counts,
                }
            return {
                "version": VAULT_VERSION,
                "unchanged": False,
                "revision": revision,
                "updatedAt": updated_at,
                "counts": counts,
                "photos": [self._index["photos"][k] for k in sorted(self._index["photos"])],
                "bios": [self._index["bios"][k] for k in sorted(self._index["bios"])],
                "covers": [self._index["covers"][k] for k in sorted(self._index["covers"])],
                "lyrics": [self._index["lyrics"][k] for k in sorted(self._index["lyrics"])],
                "avatars": [self._index["avatars"][k] for k in sorted(self._index["avatars"])],
            }

    def read_blob(self, kind: str, blob_id: str):
        if kind not in _KINDS or not isinstance(blob_id, str) or not _BLOB_ID.fullmatch(blob_id):
            raise ButlerVaultError("not_found", 404)
        with self._lock:
            entry = None
            for candidate in self._index[kind].values():
                if candidate.get("blobId") == blob_id:
                    entry = candidate
                    break
            if entry is None:
                raise ButlerVaultError("not_found", 404)
            path = self._blob_path(kind, blob_id)
            if not path.is_file():
                raise ButlerVaultError("not_found", 404)
            try:
                data = path.read_bytes()
            except OSError:
                raise ButlerVaultError("not_found", 404) from None
            content_type = "application/json; charset=utf-8" if kind == "lyrics" else entry.get("contentType", "image/jpeg")
            etag = entry.get("revision") or _revision(data)
            return data, content_type, etag

    def read_lyrics_batch(self, track_ids):
        if not isinstance(track_ids, list) or len(track_ids) > MAX_LYRICS_BATCH_IDS:
            raise ButlerVaultError("bad_request")
        cleaned = []
        for raw_id in track_ids:
            cleaned.append(_clean_key(raw_id))
        items = []
        with self._lock:
            for track_id in cleaned:
                entry = self._index["lyrics"].get(track_id)
                if entry is None:
                    continue
                path = self._blob_path("lyrics", entry["blobId"])
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("raw"), str):
                    item = {**entry, "raw": payload["raw"]}
                    if isinstance(payload.get("translatedRaw"), str):
                        item["translatedRaw"] = payload["translatedRaw"]
                    items.append(item)
        return {"items": items}

    def upsert(self, body):
        if not isinstance(body, dict):
            raise ButlerVaultError("bad_request")
        now = int(self._clock())

        raw_photos = body.get("photos") or []
        raw_bios = body.get("bios") or []
        raw_covers = body.get("covers") or []
        raw_lyrics = body.get("lyrics") or []
        raw_avatars = body.get("avatars") or []

        if not all(isinstance(x, list) for x in (raw_photos, raw_bios, raw_covers, raw_lyrics, raw_avatars)):
            raise ButlerVaultError("bad_request")
        total = len(raw_photos) + len(raw_bios) + len(raw_covers) + len(raw_lyrics) + len(raw_avatars)
        if total == 0:
            with self._lock:
                return {
                    "ok": True,
                    "revision": self._index["revision"],
                    "updatedAt": self._index["updatedAt"],
                    "saved": {k: 0 for k in ("photos", "bios", "covers", "lyrics", "avatars")},
                }
        if total > MAX_BATCH_ITEMS:
            raise ButlerVaultError("too_many_items", 400)

        # Validate every entry upfront before mutating disk or index.
        parsed_photos = []
        for item in raw_photos:
            if not isinstance(item, dict):
                raise ButlerVaultError("bad_request")
            key = _clean_key(item.get("key"))
            name = _clean_label(item.get("name"))
            if not name:
                raise ButlerVaultError("bad_request")
            source_name = _clean_label(item.get("sourceName"), default="NAS")
            filled_at = _clean_timestamp(item.get("filledAt"), now)
            data, content_type = _decode_image(item.get("data"))
            parsed_photos.append((
                key,
                data,
                {
                    "key": key,
                    "name": name,
                    "sourceName": source_name,
                    "blobId": _blob_id(key),
                    "revision": _revision(data),
                    "contentType": content_type,
                    "bytes": len(data),
                    "filledAt": filled_at,
                },
            ))

        parsed_bios = []
        for item in raw_bios:
            if not isinstance(item, dict):
                raise ButlerVaultError("bad_request")
            key = _clean_key(item.get("key"))
            name = _clean_label(item.get("name"))
            text = item.get("text")
            if not name or not isinstance(text, str) or "\x00" in text:
                raise ButlerVaultError("bad_request")
            text = text.strip()
            if not text or len(text) > MAX_BIO_CHARS:
                raise ButlerVaultError("bad_request")
            source_name = _clean_label(item.get("sourceName"), default="NAS")
            filled_at = _clean_timestamp(item.get("filledAt"), now)
            parsed_bios.append((
                key,
                {
                    "key": key,
                    "name": name,
                    "text": text,
                    "sourceName": source_name,
                    "revision": _revision(text.encode("utf-8")),
                    "filledAt": filled_at,
                },
            ))

        parsed_covers = []
        for item in raw_covers:
            if not isinstance(item, dict):
                raise ButlerVaultError("bad_request")
            album_guid = _clean_key(item.get("albumGUID"))
            track_guids = _clean_string_list(item.get("trackGUIDs"), MAX_TRACK_GUIDS, 128)
            title = _clean_label(item.get("title"))
            artist = _clean_label(item.get("artist"))
            source_name = _clean_label(item.get("sourceName"), default="NAS")
            filled_at = _clean_timestamp(item.get("filledAt"), now)
            data, content_type = _decode_image(item.get("data"))
            parsed_covers.append((
                album_guid,
                data,
                {
                    "albumGUID": album_guid,
                    "trackGUIDs": track_guids,
                    "title": title,
                    "artist": artist,
                    "sourceName": source_name,
                    "blobId": _blob_id(album_guid),
                    "revision": _revision(data),
                    "contentType": content_type,
                    "bytes": len(data),
                    "filledAt": filled_at,
                },
            ))

        parsed_lyrics = []
        for item in raw_lyrics:
            if not isinstance(item, dict):
                raise ButlerVaultError("bad_request")
            track_id = _clean_key(item.get("trackID"))
            raw_text = item.get("raw")
            if not isinstance(raw_text, str) or "\x00" in raw_text:
                raise ButlerVaultError("bad_request")
            raw_text = raw_text.strip()
            raw_bytes = raw_text.encode("utf-8")
            if not raw_bytes or len(raw_bytes) > MAX_LYRICS_BYTES:
                raise ButlerVaultError("lyrics_too_large" if raw_bytes else "bad_request", 413 if raw_bytes else 400)
            translated_raw = item.get("translatedRaw")
            if translated_raw is not None:
                if not isinstance(translated_raw, str) or "\x00" in translated_raw:
                    raise ButlerVaultError("bad_request")
                translated_raw = translated_raw.strip() or None
                if translated_raw and len(translated_raw.encode("utf-8")) > MAX_LYRICS_BYTES:
                    raise ButlerVaultError("lyrics_too_large", 413)
            source_name = _clean_label(item.get("sourceName"), default="NAS")
            online_title = _clean_label(item.get("onlineTitle"))
            online_artists = _clean_string_list(item.get("onlineArtists"), MAX_ONLINE_ARTISTS, 200)
            filled_at = _clean_timestamp(item.get("filledAt"), now)
            blob_payload = {"trackID": track_id, "raw": raw_text}
            if translated_raw:
                blob_payload["translatedRaw"] = translated_raw
            blob_bytes = _encode_json(blob_payload)
            rev_source = raw_bytes if not translated_raw else raw_bytes + b"\x00" + translated_raw.encode("utf-8")
            parsed_lyrics.append((
                track_id,
                blob_bytes,
                {
                    "trackID": track_id,
                    "sourceName": source_name,
                    "onlineTitle": online_title,
                    "onlineArtists": online_artists,
                    "blobId": _blob_id(track_id),
                    "revision": _revision(rev_source),
                    "filledAt": filled_at,
                },
            ))

        parsed_avatars = []
        for item in raw_avatars:
            if not isinstance(item, dict):
                raise ButlerVaultError("bad_request")
            key = _clean_key(item.get("key"))
            display_name = _clean_label(item.get("displayName"))
            updated_at = _clean_timestamp(item.get("updatedAt"), now)
            data, content_type = _decode_image(item.get("data"))
            parsed_avatars.append((
                key,
                data,
                {
                    "key": key,
                    "displayName": display_name,
                    "blobId": _blob_id(key),
                    "revision": _revision(data),
                    "contentType": content_type,
                    "bytes": len(data),
                    "updatedAt": updated_at,
                },
            ))

        saved = {k: 0 for k in ("photos", "bios", "covers", "lyrics", "avatars")}
        changed = False
        with self._lock:
            for kind, items in (
                ("photos", parsed_photos),
                ("covers", parsed_covers),
                ("lyrics", parsed_lyrics),
                ("avatars", parsed_avatars),
            ):
                if not items:
                    continue
                self._ensure_dir(self._root / kind)
                for key, blob_bytes, entry in items:
                    existing = self._index[kind].get(key)
                    blob_path = self._blob_path(kind, entry["blobId"])
                    if existing == entry and blob_path.is_file():
                        continue
                    write_private(blob_path, blob_bytes)
                    self._index[kind][key] = entry
                    saved[kind] += 1
                    changed = True

            for key, entry in parsed_bios:
                if self._index["bios"].get(key) == entry:
                    continue
                self._index["bios"][key] = entry
                saved["bios"] += 1
                changed = True

            if changed:
                self._index["revision"] += 1
                self._index["updatedAt"] = now
                self._persist_index()
                LOGGER.info(
                    "BUTLER-VAULT-UPSERT rev=%s photos=%s bios=%s covers=%s lyrics=%s avatars=%s",
                    self._index["revision"],
                    saved["photos"], saved["bios"], saved["covers"], saved["lyrics"], saved["avatars"],
                )
            return {
                "ok": True,
                "revision": self._index["revision"],
                "updatedAt": self._index["updatedAt"],
                "saved": saved,
            }

    def delete(self, body):
        if not isinstance(body, dict):
            raise ButlerVaultError("bad_request")
        keys_by_section = {}
        total = 0
        for section in ("photos", "bios", "covers", "lyrics", "avatars"):
            raw = body.get(section) or []
            if not isinstance(raw, list):
                raise ButlerVaultError("bad_request")
            cleaned = [_clean_key(k) for k in raw]
            keys_by_section[section] = cleaned
            total += len(cleaned)
        if total > MAX_DELETE_KEYS:
            raise ButlerVaultError("too_many_items", 400)

        removed = {k: 0 for k in ("photos", "bios", "covers", "lyrics", "avatars")}
        changed = False
        now = int(self._clock())
        with self._lock:
            for section, keys in keys_by_section.items():
                for key in keys:
                    entry = self._index[section].pop(key, None)
                    if entry is None:
                        continue
                    removed[section] += 1
                    changed = True
                    if section in _KINDS:
                        blob_id = entry.get("blobId")
                        if isinstance(blob_id, str) and _BLOB_ID.fullmatch(blob_id):
                            try:
                                self._blob_path(section, blob_id).unlink()
                            except FileNotFoundError:
                                pass
            if changed:
                self._index["revision"] += 1
                self._index["updatedAt"] = now
                self._persist_index()
                LOGGER.info(
                    "BUTLER-VAULT-DELETE rev=%s photos=%s bios=%s covers=%s lyrics=%s avatars=%s",
                    self._index["revision"],
                    removed["photos"], removed["bios"], removed["covers"], removed["lyrics"], removed["avatars"],
                )
            return {
                "ok": True,
                "revision": self._index["revision"],
                "updatedAt": self._index["updatedAt"],
                "removed": removed,
            }
