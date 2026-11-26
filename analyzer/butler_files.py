"""Safe local-disk file trash, restore, purge, move, and sidecar write inside the writable NAS library root."""

import base64
import binascii
import os
import stat
import threading
import time
import uuid
from pathlib import Path

from .downloads import majority_owner
from .log import LOGGER
from .scan import AUDIO_EXTENSIONS

MAX_BATCH_ITEMS = 500
MAX_PATH_CHARS = 1024
MAX_WRITE_BYTES = 6 * 1024 * 1024
TRASH_DIR_NAME = ".trash"
STAGING_DIR_NAME = ".xiyue-staging"
FOLDER_IMAGES = {
    "cover.jpg", "cover.jpeg", "cover.png", "cover.webp",
    "folder.jpg", "folder.jpeg", "folder.png",
    "front.jpg", "front.jpeg", "front.png",
}


class ButlerFileError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _clean_path_str(value):
    if not isinstance(value, str):
        raise ButlerFileError("bad_request")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_PATH_CHARS or any(c in cleaned for c in "\x00\r\n\\"):
        raise ButlerFileError("bad_request")
    return cleaned


def _segments(raw):
    parts = [part for part in raw.split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        raise ButlerFileError("outside")
    return parts


class ButlerFileOps:
    """Performs atomic move-to-trash, restore, and .trash-only purge directly on
    the NAS disk mounted at `root` (the writable `--downloads` library mount).

    Security invariants:
    - Never follows symlinks out of `root` or operates on symlinks.
    - Never touches `.xiyue-staging`.
    - `purge` ONLY unlinks regular files whose immediate parent directory is
      literally `.trash` inside `root`.
    """

    def __init__(self, root, clock=time.time):
        self._root = Path(root)
        self._clock = clock
        self._lock = threading.RLock()
        self._owner = None
        self._owner_checked_at = None

    @property
    def root(self) -> Path:
        return self._root

    def available(self) -> bool:
        return self._root.is_dir() and os.access(self._root, os.W_OK)

    def _require_available(self):
        if not self.available():
            raise ButlerFileError("files_unavailable", 503)

    def own(self, path: Path, folder: bool = False):
        """Applies the library's majority non-root uid/gid/mode so NAS & Fly Music
        keep full access to files and `.trash` folders created by the container."""
        try:
            now = time.monotonic()
            with self._lock:
                if self._owner_checked_at is None or now - self._owner_checked_at >= 3600:
                    self._owner = majority_owner(self._root)
                    self._owner_checked_at = now
                owner = self._owner
            if owner is not None:
                uid, gid, mode = owner
                if folder:
                    mode |= 0o100 | (0o010 if mode & 0o040 else 0) | (0o001 if mode & 0o004 else 0)
            else:
                info = self._root.stat()
                uid, gid = info.st_uid, info.st_gid
                mode = info.st_mode & (0o777 if folder else 0o666)
            os.chown(path, uid, gid)
            os.chmod(path, mode)
        except OSError:
            pass

    def _verify_no_symlink_escape(self, candidate: Path):
        root_resolved = self._root.resolve()
        current = candidate
        while current != self._root:
            if current.is_symlink():
                raise ButlerFileError("outside")
            parent = current.parent
            if parent == current:
                raise ButlerFileError("outside")
            current = parent
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ButlerFileError("outside")

    def resolve_existing_file(self, raw_path: str):
        """Resolves either a container-relative path or a full Fly Music NAS path
        (e.g. `/vol1/1000/Music/Artist/Album/song.mp3`) to `(Path, relative_posix)`
        by checking longest suffixes first."""
        self._require_available()
        cleaned = _clean_path_str(raw_path)
        parts = _segments(cleaned)
        root_resolved = self._root.resolve()
        for start in range(len(parts)):
            suffix = parts[start:]
            if STAGING_DIR_NAME in suffix:
                continue
            candidate = self._root.joinpath(*suffix)
            try:
                if candidate.is_symlink():
                    continue
                info = candidate.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            try:
                self._verify_no_symlink_escape(candidate)
            except ButlerFileError:
                continue
            if candidate.resolve().is_relative_to(root_resolved):
                return candidate, candidate.relative_to(self._root).as_posix()
        raise ButlerFileError("missing", 404)

    def resolve_relative_target(self, raw_rel: str, *, allow_trash: bool = False):
        """Resolves a strict container-relative path (does not need to exist yet)."""
        self._require_available()
        cleaned = _clean_path_str(raw_rel)
        if cleaned.startswith("/"):
            cleaned = cleaned.lstrip("/")
        parts = _segments(cleaned)
        if STAGING_DIR_NAME in parts:
            raise ButlerFileError("outside")
        if not allow_trash and TRASH_DIR_NAME in parts:
            raise ButlerFileError("outside")
        candidate = self._root.joinpath(*parts)
        # Verify existing ancestor prefix has no symlink escape.
        current = candidate
        while current != self._root and not current.exists():
            current = current.parent
        self._verify_no_symlink_escape(current)
        if candidate.is_symlink():
            raise ButlerFileError("outside")
        return candidate, "/".join(parts)

    def resolve_batch(self, raw_paths):
        self._require_available()
        if not isinstance(raw_paths, list) or len(raw_paths) > 2000:
            raise ButlerFileError("bad_request")
        matched = {}
        prefixes = {}
        for raw in raw_paths:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                _, rel = self.resolve_existing_file(raw)
            except ButlerFileError:
                continue
            matched[raw] = rel
            if raw.endswith(rel) and len(raw) > len(rel):
                prefix = raw[: len(raw) - len(rel)]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        best_prefix = max(prefixes.items(), key=lambda item: (item[1], -len(item[0])))[0] if prefixes else ""
        return {"available": True, "matched": matched, "prefix": best_prefix}

    def status(self):
        writable = self.available()
        if not writable:
            return {
                "available": False,
                "writable": False,
                "supportsMove": False,
                "supportsWrite": False,
                "trashCount": 0,
                "trashBytes": 0,
            }
        count, total_bytes = 0, 0
        for item in self._iter_trash_files():
            count += 1
            total_bytes += item["sizeBytes"]
        return {
            "available": True,
            "writable": True,
            "supportsMove": True,
            "supportsWrite": True,
            "trashCount": count,
            "trashBytes": total_bytes,
        }

    def list_trash(self):
        self._require_available()
        items = list(self._iter_trash_files())
        items.sort(key=lambda entry: entry["modifiedAt"], reverse=True)
        return {"items": items}

    def _iter_trash_files(self):
        if not self._root.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=False):
            current = Path(dirpath)
            if current.name == STAGING_DIR_NAME:
                dirnames[:] = []
                continue
            if current.name == TRASH_DIR_NAME:
                dirnames[:] = []
                for name in sorted(filenames):
                    entry = current / name
                    try:
                        info = entry.lstat()
                    except OSError:
                        continue
                    if stat.S_ISREG(info.st_mode):
                        yield {
                            "relativePath": entry.relative_to(self._root).as_posix(),
                            "name": name,
                            "sizeBytes": int(info.st_size),
                            "modifiedAt": int(info.st_mtime),
                        }
            else:
                dirnames[:] = [
                    d for d in sorted(dirnames)
                    if d != STAGING_DIR_NAME and not (current / d).is_symlink()
                ]

    def unique_trash_path(self, src: Path, stamp: int) -> Path:
        trash_dir = src.parent / TRASH_DIR_NAME
        if trash_dir.is_symlink():
            raise ButlerFileError("outside")
        trash_dir.mkdir(exist_ok=True)
        self.own(trash_dir, folder=True)
        base = f"{stamp}-{src.name}"
        candidate = trash_dir / base
        if not candidate.exists():
            return candidate
        stem = src.stem
        suffix = src.suffix
        for number in range(2, 1000):
            candidate = trash_dir / f"{stamp}-{stem}-{number}{suffix}"
            if not candidate.exists():
                return candidate
        raise ButlerFileError("trash_name_taken", 409)

    def _find_own_lyrics(self, audio_path: Path):
        lrc = audio_path.with_suffix(".lrc")
        if not lrc.is_file() or lrc.is_symlink():
            return None
        stem = audio_path.stem
        try:
            for sibling in audio_path.parent.iterdir():
                if sibling == audio_path or sibling.is_symlink() or not sibling.is_file():
                    continue
                if sibling.suffix.lower() in AUDIO_EXTENSIONS and sibling.stem == stem:
                    return None
        except OSError:
            return None
        return lrc

    def trash(self, body):
        self._require_available()
        if not isinstance(body, dict):
            raise ButlerFileError("bad_request")
        items = body.get("items")
        if not isinstance(items, list) or len(items) == 0 or len(items) > MAX_BATCH_ITEMS:
            raise ButlerFileError("bad_request")
        raw_stamp = body.get("stamp")
        stamp = int(raw_stamp) if isinstance(raw_stamp, (int, float)) and raw_stamp > 0 else int(self._clock())

        results = []
        done = 0
        failed = 0
        with self._lock:
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise ButlerFileError("bad_request")
                raw_path = item["path"]
                keep_raw = item.get("keepPath")
                include_lyrics = item.get("includeLyrics", True) is not False
                try:
                    src, rel_from = self.resolve_existing_file(raw_path)
                    if TRASH_DIR_NAME in src.relative_to(self._root).parts:
                        raise ButlerFileError("already_trashed")
                    if src.suffix.lower() == ".cue":
                        raise ButlerFileError("cue_unsupported")
                    if isinstance(keep_raw, str) and keep_raw.strip():
                        try:
                            keep_path, _ = self.resolve_existing_file(keep_raw)
                        except ButlerFileError:
                            raise ButlerFileError("keep_missing", 404)
                        if os.path.samefile(src, keep_path):
                            raise ButlerFileError("same_file", 409)
                    size_bytes = int(src.lstat().st_size)
                    lrc = self._find_own_lyrics(src) if include_lyrics else None
                    dst = self.unique_trash_path(src, stamp)
                    os.replace(src, dst)
                    rel_trash = dst.relative_to(self._root).as_posix()

                    lrc_from = None
                    lrc_trash = None
                    if lrc is not None:
                        try:
                            lrc_dst = self.unique_trash_path(lrc, stamp)
                            os.replace(lrc, lrc_dst)
                            lrc_from = lrc.relative_to(self._root).as_posix()
                            lrc_trash = lrc_dst.relative_to(self._root).as_posix()
                        except OSError:
                            pass

                    results.append({
                        "ok": True,
                        "path": raw_path,
                        "relativeFrom": rel_from,
                        "relativeTrash": rel_trash,
                        "sizeBytes": size_bytes,
                        "lyricsFrom": lrc_from,
                        "lyricsTrash": lrc_trash,
                    })
                    done += 1
                except ButlerFileError as error:
                    results.append({"ok": False, "path": raw_path, "error": error.code})
                    failed += 1
                except OSError:
                    results.append({"ok": False, "path": raw_path, "error": "io_failed"})
                    failed += 1
        LOGGER.info("BUTLER-FILES-TRASH done=%s failed=%s", done, failed)
        return {"ok": True, "done": done, "failed": failed, "results": results}

    def restore(self, body):
        self._require_available()
        if not isinstance(body, dict):
            raise ButlerFileError("bad_request")
        items = body.get("items")
        if not isinstance(items, list) or len(items) == 0 or len(items) > MAX_BATCH_ITEMS:
            raise ButlerFileError("bad_request")

        results = []
        done = 0
        failed = 0
        with self._lock:
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("trashPath"), str)
                    or not isinstance(item.get("restorePath"), str)
                ):
                    raise ButlerFileError("bad_request")
                raw_trash = item["trashPath"]
                raw_restore = item["restorePath"]
                try:
                    trash_file, rel_trash = self.resolve_relative_target(raw_trash, allow_trash=True)
                    if trash_file.parent.name != TRASH_DIR_NAME:
                        raise ButlerFileError("outside")
                    if not trash_file.exists():
                        raise ButlerFileError("missing", 404)
                    if trash_file.is_symlink() or not trash_file.is_file():
                        raise ButlerFileError("outside")
                    dst_file, rel_restore = self.resolve_relative_target(raw_restore, allow_trash=False)
                    if dst_file.exists():
                        raise ButlerFileError("exists", 409)
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    self.own(dst_file.parent, folder=True)
                    os.replace(trash_file, dst_file)
                    self._cleanup_empty_trash_dir(trash_file.parent)
                    results.append({
                        "ok": True,
                        "trashPath": rel_trash,
                        "restorePath": rel_restore,
                    })
                    done += 1
                except ButlerFileError as error:
                    results.append({
                        "ok": False,
                        "trashPath": raw_trash,
                        "restorePath": raw_restore,
                        "error": error.code,
                    })
                    failed += 1
                except OSError:
                    results.append({
                        "ok": False,
                        "trashPath": raw_trash,
                        "restorePath": raw_restore,
                        "error": "io_failed",
                    })
                    failed += 1
        LOGGER.info("BUTLER-FILES-RESTORE done=%s failed=%s", done, failed)
        return {"ok": True, "done": done, "failed": failed, "results": results}

    def purge(self, body):
        self._require_available()
        if not isinstance(body, dict):
            raise ButlerFileError("bad_request")
        paths = body.get("paths")
        if not isinstance(paths, list) or len(paths) == 0 or len(paths) > MAX_BATCH_ITEMS:
            raise ButlerFileError("bad_request")

        results = []
        done = 0
        failed = 0
        with self._lock:
            for raw_path in paths:
                if not isinstance(raw_path, str):
                    raise ButlerFileError("bad_request")
                try:
                    target, rel = self.resolve_relative_target(raw_path, allow_trash=True)
                    if target.parent.name != TRASH_DIR_NAME:
                        raise ButlerFileError("outside")
                    if not target.exists():
                        results.append({"ok": True, "path": rel, "alreadyGone": True})
                        done += 1
                        continue
                    if target.is_symlink() or not target.is_file():
                        raise ButlerFileError("outside")
                    target.unlink()
                    self._cleanup_empty_trash_dir(target.parent)
                    results.append({"ok": True, "path": rel, "alreadyGone": False})
                    done += 1
                except ButlerFileError as error:
                    results.append({"ok": False, "path": raw_path, "error": error.code})
                    failed += 1
                except OSError:
                    results.append({"ok": False, "path": raw_path, "error": "io_failed"})
                    failed += 1
        LOGGER.info("BUTLER-FILES-PURGE done=%s failed=%s", done, failed)
        return {"ok": True, "done": done, "failed": failed, "results": results}

    def move(self, body):
        """Moves tracks (and their own .lrc sidecar) inside the music root."""
        self._require_available()
        if not isinstance(body, dict):
            raise ButlerFileError("bad_request")
        items = body.get("items")
        if not isinstance(items, list) or len(items) == 0 or len(items) > MAX_BATCH_ITEMS:
            raise ButlerFileError("bad_request")

        results = []
        done = 0
        failed = 0
        with self._lock:
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("from"), str)
                    or not isinstance(item.get("to"), str)
                ):
                    raise ButlerFileError("bad_request")
                raw_from = item["from"]
                raw_to = item["to"]
                include_lyrics = item.get("includeLyrics", True) is not False
                try:
                    src, rel_from = self.resolve_existing_file(raw_from)
                    if TRASH_DIR_NAME in src.relative_to(self._root).parts:
                        raise ButlerFileError("already_trashed")
                    if src.suffix.lower() == ".cue":
                        raise ButlerFileError("cue_unsupported")

                    # Strip the same NAS prefix that matched raw_from if raw_to carries it.
                    target_rel_raw = raw_to
                    cleaned_from = _clean_path_str(raw_from) or ""
                    cleaned_to = _clean_path_str(raw_to) or ""
                    if cleaned_from.endswith(rel_from):
                        prefix = cleaned_from[: len(cleaned_from) - len(rel_from)]
                        if prefix and cleaned_to.startswith(prefix):
                            target_rel_raw = cleaned_to[len(prefix):]

                    dst, rel_to = self.resolve_relative_target(target_rel_raw, allow_trash=False)
                    if dst.exists():
                        if os.path.samefile(src, dst):
                            results.append({
                                "ok": True,
                                "from": raw_from,
                                "relativeFrom": rel_from,
                                "relativeTo": rel_to,
                                "alreadyThere": True,
                                "lyricsFrom": None,
                                "lyricsTo": None,
                                "removedFolders": [],
                            })
                            done += 1
                            continue
                        raise ButlerFileError("exists", 409)

                    self._mkdir_owned(dst.parent)
                    lrc = self._find_own_lyrics(src) if include_lyrics else None
                    os.replace(src, dst)

                    lrc_from = None
                    lrc_to = None
                    if lrc is not None:
                        dst_lrc = dst.with_suffix(".lrc")
                        if not dst_lrc.exists():
                            try:
                                os.replace(lrc, dst_lrc)
                                lrc_from = lrc.relative_to(self._root).as_posix()
                                lrc_to = dst_lrc.relative_to(self._root).as_posix()
                            except OSError:
                                pass

                    removed_folders = self._cleanup_after_move(src.parent, dst.parent)
                    results.append({
                        "ok": True,
                        "from": raw_from,
                        "relativeFrom": rel_from,
                        "relativeTo": rel_to,
                        "alreadyThere": False,
                        "lyricsFrom": lrc_from,
                        "lyricsTo": lrc_to,
                        "removedFolders": removed_folders,
                    })
                    done += 1
                except ButlerFileError as error:
                    results.append({"ok": False, "from": raw_from, "error": error.code})
                    failed += 1
                except OSError:
                    results.append({"ok": False, "from": raw_from, "error": "io_failed"})
                    failed += 1
        LOGGER.info("BUTLER-FILES-MOVE done=%s failed=%s", done, failed)
        return {"ok": True, "done": done, "failed": failed, "results": results}

    def write(self, body):
        """Writes sidecar files (artist.jpg, artist.nfo, cover.jpg, .lrc) on NAS disk."""
        self._require_available()
        if not isinstance(body, dict):
            raise ButlerFileError("bad_request")
        items = body.get("items")
        if not isinstance(items, list) or len(items) == 0 or len(items) > MAX_BATCH_ITEMS:
            raise ButlerFileError("bad_request")

        results = []
        done = 0
        failed = 0
        with self._lock:
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("relativePath"), str)
                    or not isinstance(item.get("contentBase64"), str)
                ):
                    raise ButlerFileError("bad_request")
                raw_rel = item["relativePath"]
                overwrite = bool(item.get("overwrite", False))
                try:
                    dst, rel = self.resolve_relative_target(raw_rel, allow_trash=False)
                    if dst.exists():
                        if dst.is_symlink() or not dst.is_file():
                            raise ButlerFileError("outside")
                        if not overwrite:
                            results.append({"ok": True, "relativePath": rel, "skipped": True})
                            continue
                    try:
                        payload = base64.b64decode(item["contentBase64"], validate=True)
                    except (binascii.Error, ValueError):
                        raise ButlerFileError("bad_request")
                    if not payload or len(payload) > MAX_WRITE_BYTES:
                        raise ButlerFileError("bad_request")
                    self._mkdir_owned(dst.parent)
                    tmp = dst.parent / f".xiyue-write-{uuid.uuid4().hex[:8]}.tmp"
                    try:
                        with open(tmp, "wb") as handle:
                            handle.write(payload)
                        self.own(tmp, folder=False)
                        os.replace(tmp, dst)
                    finally:
                        tmp.unlink(missing_ok=True)
                    results.append({"ok": True, "relativePath": rel, "skipped": False})
                    done += 1
                except ButlerFileError as error:
                    results.append({"ok": False, "relativePath": raw_rel, "error": error.code})
                    failed += 1
                except OSError:
                    results.append({"ok": False, "relativePath": raw_rel, "error": "io_failed"})
                    failed += 1
        LOGGER.info("BUTLER-FILES-WRITE done=%s failed=%s", done, failed)
        return {"ok": True, "done": done, "failed": failed, "results": results}

    def _mkdir_owned(self, folder: Path):
        missing = []
        cursor = folder
        while cursor != self._root and not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        folder.mkdir(parents=True, exist_ok=True)
        for created in reversed(missing):
            self.own(created, folder=True)

    def _cleanup_after_move(self, old_dir: Path, new_dir: Path):
        removed = []
        cursor = old_dir
        while cursor != self._root and cursor != new_dir:
            try:
                entries = list(cursor.iterdir())
            except OSError:
                break
            if any(
                e.is_dir() or e.is_symlink() or e.suffix.lower() in AUDIO_EXTENSIONS
                for e in entries
            ):
                break
            # Carry folder cover art along only from the immediate emptied album folder.
            if cursor == old_dir:
                for entry in entries:
                    if entry.is_file() and entry.name.lower() in FOLDER_IMAGES:
                        dst_img = new_dir / entry.name
                        if not dst_img.exists():
                            try:
                                os.replace(entry, dst_img)
                            except OSError:
                                pass
            try:
                if not any(cursor.iterdir()):
                    cursor.rmdir()
                    removed.append(cursor.relative_to(self._root).as_posix())
                    cursor = cursor.parent
                    continue
            except OSError:
                pass
            break
        return removed

    def _cleanup_empty_trash_dir(self, trash_dir: Path):
        if trash_dir.name != TRASH_DIR_NAME or trash_dir == self._root:
            return
        try:
            trash_dir.rmdir()
        except OSError:
            pass
