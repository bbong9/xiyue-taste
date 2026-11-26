"""NAS-side background audio quality upgrade (洗版) worker.

Downloads candidate tracks directly on the NAS, verifies recording identity &
quality, preserves tags/lyrics/cover + ReplayGain, and atomically swaps files
on disk without relying on iPhone foreground execution or WebDAV upload.
"""

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import mutagen
import numpy as np
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, TALB, TDRC, TIT2, TPE1, TPOS, TRCK, USLT
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4, MP4Cover

from . import loudness
from .sources import write_private
from .butler_files import ButlerFileError, ButlerFileOps, STAGING_DIR_NAME
from .downloads import (
    MAX_AUDIO_BYTES,
    MIN_AUDIO_BYTES,
    SPARE_BYTES,
    USER_AGENT,
    _Guard,
    _check_shape,
    _facts,
    _is_audio,
    check_public,
)
from .log import LOGGER
from .resolver import OWNER, ResolveError

MAX_UPGRADE_ITEMS = 100
MAX_CANDIDATES_PER_ITEM = 8
LOSSLESS_FORMATS = {"flac", "wav", "ape"}
TARGET_ALIASES = {
    "320k": "high",
    "high": "high",
    "flac": "lossless",
    "lossless": "lossless",
    "hires": "hires",
    "master": "master",
}


class ButlerUpgradeError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _normalize_target(value) -> str:
    if not isinstance(value, str):
        raise ButlerUpgradeError("bad_request")
    mapped = TARGET_ALIASES.get(value.strip().lower())
    if mapped is None:
        raise ButlerUpgradeError("bad_request")
    return mapped


def _format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return ""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{max(1, round(num_bytes / 1024))} KB"


def _describe_facts(facts: dict, size_bytes: int) -> str:
    fmt = (facts.get("format") or "").upper()
    parts = [fmt] if fmt else []
    bit_depth = int(facts.get("bitDepth") or 0)
    sample_rate = int(facts.get("sampleRate") or 0)
    bit_rate = int(facts.get("bitRate") or 0)
    if fmt.lower() in LOSSLESS_FORMATS:
        spec = []
        if bit_depth > 0:
            spec.append(f"{bit_depth}bit")
        if sample_rate > 0:
            khz = sample_rate / 1000.0
            spec.append(f"{int(khz) if khz.is_integer() else f'{khz:.1f}'}kHz")
        if spec:
            parts.append("/".join(spec))
    elif bit_rate > 0:
        parts.append(f"{round(bit_rate / 1000)} kbps")
    size_str = _format_size(size_bytes)
    if size_str:
        parts.append(size_str)
    return " · ".join(parts)


def _is_target_met(facts: dict, target: str) -> bool:
    fmt = (facts.get("format") or "").lower()
    bit_depth = int(facts.get("bitDepth") or 0)
    sample_rate = int(facts.get("sampleRate") or 0)
    bit_rate = int(facts.get("bitRate") or 0)
    is_lossless = fmt in LOSSLESS_FORMATS
    if target == "high":
        return is_lossless or bit_rate >= 256_000
    if target == "lossless":
        return is_lossless
    if target == "hires":
        return is_lossless and (bit_depth >= 24 or sample_rate > 48_000)
    if target == "master":
        return is_lossless and bit_depth >= 24 and sample_rate >= 88_200
    return False


def _accepts_new_quality(facts: dict, target: str) -> bool:
    fmt = (facts.get("format") or "").lower()
    bit_depth = int(facts.get("bitDepth") or 0)
    sample_rate = int(facts.get("sampleRate") or 0)
    bit_rate = int(facts.get("bitRate") or 0)
    is_lossless = fmt in LOSSLESS_FORMATS
    if target == "high":
        return is_lossless or bit_rate >= 256_000
    if target == "lossless":
        return is_lossless
    if target == "hires":
        return is_lossless and (bit_depth >= 24 or sample_rate > 48_000)
    if target == "master":
        return is_lossless and bit_depth >= 24 and sample_rate >= 88_200
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(262144)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _decode_pcm(path: Path, sample_rate: int, start_sec: float = 0.0, duration_sec: float = 0.0):
    cmd = ["ffmpeg", "-v", "error"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.2f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1"]
    if sample_rate > 0:
        cmd += ["-ar", str(sample_rate)]
    if duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.2f}"]
    cmd += ["-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    if proc.returncode != 0 or len(proc.stdout) < 400:
        return None
    usable = (len(proc.stdout) // 4) * 4
    return np.frombuffer(proc.stdout[:usable], dtype="<f4")


def _loudness_envelope(samples: np.ndarray, sample_rate: int = 8000, window_sec: float = 0.5):
    step = max(1, int(sample_rate * window_sec))
    count = len(samples) // step
    if count < 6:
        return None
    frames = samples[: count * step].reshape(count, step)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return rms


def _same_recording_correlation(old_path: Path, new_path: Path):
    old_pcm = _decode_pcm(old_path, sample_rate=8000, duration_sec=180.0)
    new_pcm = _decode_pcm(new_path, sample_rate=8000, duration_sec=180.0)
    if old_pcm is None or new_pcm is None:
        return None
    env_a = _loudness_envelope(old_pcm)
    env_b = _loudness_envelope(new_pcm)
    if env_a is None or env_b is None:
        return None
    best = -1.0
    for shift in range(-8, 9):
        if shift >= 0:
            a, b = env_a[shift:], env_b
        else:
            a, b = env_a, env_b[-shift:]
        length = min(len(a), len(b))
        if length < 6:
            continue
        x = a[:length] - np.mean(a[:length])
        y = b[:length] - np.mean(b[:length])
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= 1e-9:
            continue
        corr = float(np.dot(x, y) / denom)
        if corr > best:
            best = corr
    return best if best >= -0.99 else None


def _estimate_cutoff_hz(path: Path, sample_rate: int, duration_ms: int):
    if sample_rate <= 0:
        return None
    dur_sec = max(0.0, duration_ms / 1000.0)
    start_sec = max(0.0, dur_sec * 0.25) if dur_sec > 30 else 0.0
    pcm = _decode_pcm(path, sample_rate=sample_rate, start_sec=start_sec, duration_sec=20.0)
    if pcm is None or len(pcm) < 4096:
        return None
    n_fft = 4096
    hop = 2048
    n_frames = (len(pcm) - n_fft) // hop + 1
    if n_frames < 1:
        return None
    window = np.hanning(n_fft).astype(np.float32)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    limit = min(n_frames, 120)
    for idx in range(limit):
        chunk = pcm[idx * hop : idx * hop + n_fft] * window
        spec = np.abs(np.fft.rfft(chunk)) ** 2
        acc += spec
    acc /= limit
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    db = 10.0 * np.log10(acc + 1e-18)
    mid_mask = (freqs >= 1000.0) & (freqs <= 4000.0)
    if not np.any(mid_mask):
        return None
    ref_db = float(np.percentile(db[mid_mask], 75))
    threshold = ref_db - 48.0
    # Scan downward in 500 Hz bins to find the highest band with real musical energy.
    top_freq = min(sample_rate / 2.0, 44000.0)
    cutoff = 0.0
    freq = 1000.0
    while freq + 500.0 <= top_freq:
        band = (freqs >= freq) & (freqs < freq + 500.0)
        if np.any(band) and float(np.mean(db[band])) >= threshold:
            cutoff = freq + 500.0
        freq += 500.0
    return cutoff


def _extract_embedded_extras(old_path: Path):
    """Reads embedded lyrics, cover picture, and basic tags from the original file."""
    lyrics_text = None
    cover_bytes = None
    cover_mime = None
    tags = {}
    try:
        audio = mutagen.File(old_path)
    except Exception:
        return tags, lyrics_text, cover_bytes, cover_mime
    if audio is None:
        return tags, lyrics_text, cover_bytes, cover_mime

    if isinstance(audio, FLAC):
        if audio.pictures:
            pic = audio.pictures[0]
            if pic.data:
                cover_bytes = bytes(pic.data)
                cover_mime = pic.mime or "image/jpeg"
        for key in ("lyrics", "unsyncedlyrics"):
            vals = audio.get(key)
            if vals and str(vals[0]).strip():
                lyrics_text = str(vals[0]).strip()
                break
        tags = {
            "title": (audio.get("title") or [None])[0],
            "artists": list(audio.get("artist") or []),
            "album": (audio.get("album") or [None])[0],
            "trackNumber": (audio.get("tracknumber") or [None])[0],
            "discNumber": (audio.get("discnumber") or [None])[0],
            "year": (audio.get("date") or audio.get("year") or [None])[0],
        }
    elif isinstance(audio, MP3) and isinstance(audio.tags, ID3):
        uslt = audio.tags.getall("USLT")
        if uslt and uslt[0].text and uslt[0].text.strip():
            lyrics_text = uslt[0].text.strip()
        apics = audio.tags.getall("APIC")
        if apics and apics[0].data:
            cover_bytes = bytes(apics[0].data)
            cover_mime = apics[0].mime or "image/jpeg"
        tags = {
            "title": str(audio.tags["TIT2"].text[0]) if "TIT2" in audio.tags and audio.tags["TIT2"].text else None,
            "artists": [str(v) for v in audio.tags["TPE1"].text] if "TPE1" in audio.tags else [],
            "album": str(audio.tags["TALB"].text[0]) if "TALB" in audio.tags and audio.tags["TALB"].text else None,
            "trackNumber": str(audio.tags["TRCK"].text[0]) if "TRCK" in audio.tags and audio.tags["TRCK"].text else None,
            "discNumber": str(audio.tags["TPOS"].text[0]) if "TPOS" in audio.tags and audio.tags["TPOS"].text else None,
            "year": str(audio.tags["TDRC"].text[0]) if "TDRC" in audio.tags and audio.tags["TDRC"].text else None,
        }
    elif isinstance(audio, MP4) and audio.tags is not None:
        lyr = audio.tags.get("\xa9lyr")
        if lyr and str(lyr[0]).strip():
            lyrics_text = str(lyr[0]).strip()
        covr = audio.tags.get("covr")
        if covr:
            first = covr[0]
            cover_bytes = bytes(first)
            cover_mime = "image/png" if getattr(first, "imageformat", None) == MP4Cover.FORMAT_PNG else "image/jpeg"
        tags = {
            "title": (audio.tags.get("\xa9nam") or [None])[0],
            "artists": list(audio.tags.get("\xa9ART") or []),
            "album": (audio.tags.get("\xa9alb") or [None])[0],
            "year": (audio.tags.get("\xa9day") or [None])[0],
        }
    return tags, lyrics_text, cover_bytes, cover_mime


def _apply_preserved_tags(new_path: Path, old_path: Path, item: dict):
    """Writes Fly Music metadata (falling back to old file's tags) plus old file's
    embedded lyrics & cover into the upgraded file."""
    old_tags, lyrics_text, cover_bytes, cover_mime = _extract_embedded_extras(old_path)
    title = (item.get("title") or "").strip() or old_tags.get("title") or ""
    artists = [a.strip() for a in (item.get("artists") or []) if isinstance(a, str) and a.strip()] or old_tags.get("artists") or []
    album = (item.get("album") or "").strip() or old_tags.get("album") or ""
    track_no = item.get("trackNumber") or old_tags.get("trackNumber")
    disc_no = item.get("discNumber") or old_tags.get("discNumber")
    year = item.get("year") or old_tags.get("year")

    try:
        audio = mutagen.File(new_path)
    except Exception:
        return
    if audio is None:
        return

    try:
        if isinstance(audio, FLAC):
            if title:
                audio["TITLE"] = [title]
            if artists:
                audio["ARTIST"] = artists
            if album:
                audio["ALBUM"] = [album]
            if track_no:
                audio["TRACKNUMBER"] = [str(track_no)]
            if disc_no:
                audio["DISCNUMBER"] = [str(disc_no)]
            if year:
                audio["DATE"] = [str(year)]
            if lyrics_text:
                audio["LYRICS"] = [lyrics_text]
            if cover_bytes and not audio.pictures:
                pic = Picture()
                pic.type = 3
                pic.mime = cover_mime or "image/jpeg"
                pic.data = cover_bytes
                audio.add_picture(pic)
            audio.save()
        elif isinstance(audio, MP3):
            if audio.tags is None:
                audio.add_tags()
            if title:
                audio.tags.setall("TIT2", [TIT2(encoding=3, text=[title])])
            if artists:
                audio.tags.setall("TPE1", [TPE1(encoding=3, text=artists)])
            if album:
                audio.tags.setall("TALB", [TALB(encoding=3, text=[album])])
            if track_no:
                audio.tags.setall("TRCK", [TRCK(encoding=3, text=[str(track_no)])])
            if disc_no:
                audio.tags.setall("TPOS", [TPOS(encoding=3, text=[str(disc_no)])])
            if year:
                audio.tags.setall("TDRC", [TDRC(encoding=3, text=[str(year)])])
            if lyrics_text and not audio.tags.getall("USLT"):
                audio.tags.add(USLT(encoding=3, lang="chi", desc="", text=lyrics_text))
            if cover_bytes and not audio.tags.getall("APIC"):
                audio.tags.add(APIC(encoding=3, mime=cover_mime or "image/jpeg", type=3, desc="", data=cover_bytes))
            audio.save()
    except Exception as error:
        LOGGER.warning("BUTLER-UPGRADE-TAG-WARN %s %s", new_path.name, type(error).__name__)


class ButlerUpgrader:
    """Runs quality-upgrade batches directly on the NAS in a background thread."""

    def __init__(self, files: ButlerFileOps, data, resolver=None, on_done=None, allow_private=False, clock=time.time):
        self._files = files
        self._data = Path(data)
        self._resolver = resolver
        self._on_done = on_done
        self._check = _check_shape if allow_private else check_public
        self._clock = clock
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker = None
        self._state = self._load_state()
        if self._state.get("state") == "running":
            # A previous container restart interrupted an in-flight batch.
            self._state["state"] = "idle"
            self._state["current"] = None
            for item in self._state.get("items", []):
                if item.get("status") in ("queued", "running"):
                    item["status"] = "cancelled"
                    item["reason"] = "容器重启，批次已中止，可重新洗版"
            self._persist_state()

    def _state_path(self) -> Path:
        return self._data / "butler-upgrade-state.json"

    def _empty_state(self):
        return {
            "state": "idle",
            "batchId": "",
            "target": "",
            "total": 0,
            "done": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "current": None,
            "startedAt": None,
            "finishedAt": None,
            "items": [],
        }

    def _load_state(self):
        path = self._state_path()
        if not path.is_file():
            return self._empty_state()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("items"), list):
                return raw
        except (OSError, ValueError):
            pass
        return self._empty_state()

    def _persist_state(self):
        try:
            self._data.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            write_private(self._state_path(), payload)
        except OSError:
            pass

    def snapshot(self):
        with self._lock:
            return {
                "available": self._files.available(),
                "state": self._state["state"],
                "batchId": self._state["batchId"],
                "target": self._state["target"],
                "total": self._state["total"],
                "done": self._state["done"],
                "succeeded": self._state["succeeded"],
                "failed": self._state["failed"],
                "skipped": self._state["skipped"],
                "current": dict(self._state["current"]) if self._state["current"] else None,
                "startedAt": self._state["startedAt"],
                "finishedAt": self._state["finishedAt"],
                "items": [dict(entry) for entry in self._state["items"]],
            }

    def start(self, body):
        if not self._files.available():
            raise ButlerUpgradeError("files_unavailable", 503)
        if not isinstance(body, dict):
            raise ButlerUpgradeError("bad_request")
        target = _normalize_target(body.get("target"))
        skips_sound_checks = bool(body.get("skipsSoundChecks", False))
        keep_old_in_trash = bool(body.get("keepOldInTrash", False))
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not (1 <= len(raw_items) <= MAX_UPGRADE_ITEMS):
            raise ButlerUpgradeError("bad_request")

        plan = []
        state_items = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ButlerUpgradeError("bad_request")
            track_id = raw.get("trackId")
            title = raw.get("title")
            nas_path = raw.get("nasPath")
            candidates = raw.get("candidates")
            if (
                not isinstance(track_id, str) or not track_id.strip()
                or not isinstance(title, str) or not title.strip()
                or not isinstance(nas_path, str) or not nas_path.strip()
                or not isinstance(candidates, list) or not candidates
            ):
                raise ButlerUpgradeError("bad_request")
            item_target = _normalize_target(raw["target"]) if raw.get("target") is not None else target
            clean_candidates = []
            for cand in candidates[:MAX_CANDIDATES_PER_ITEM]:
                if not isinstance(cand, dict):
                    continue
                url = cand.get("url")
                platform = cand.get("platform")
                quality = cand.get("quality")
                music_info = cand.get("musicInfo")
                has_url = isinstance(url, str) and bool(url.strip())
                has_resolve = isinstance(platform, str) and isinstance(quality, str) and isinstance(music_info, dict)
                if not has_url and not has_resolve:
                    continue
                cand_target = _normalize_target(cand["target"]) if cand.get("target") is not None else item_target
                clean_candidates.append({
                    "sourceName": str(cand.get("sourceName") or "").strip(),
                    "target": cand_target,
                    "url": url.strip() if has_url else "",
                    "platform": platform.strip() if isinstance(platform, str) else "",
                    "quality": quality.strip() if isinstance(quality, str) else "",
                    "musicInfo": music_info if has_resolve else None,
                    "available": cand.get("available") if isinstance(cand.get("available"), list) else None,
                    "userAgent": str(cand.get("userAgent") or "").strip(),
                    "referer": str(cand.get("referer") or "").strip(),
                })
            if not clean_candidates:
                raise ButlerUpgradeError("bad_request")
            artists = [a.strip() for a in (raw.get("artists") or []) if isinstance(a, str) and a.strip()]
            plan.append({
                "trackId": track_id.strip(),
                "title": title.strip(),
                "artists": artists,
                "album": str(raw.get("album") or "").strip(),
                "trackNumber": raw.get("trackNumber"),
                "discNumber": raw.get("discNumber"),
                "year": raw.get("year"),
                "nasPath": nas_path.strip(),
                "target": item_target,
                "candidates": clean_candidates,
            })
            state_items.append({
                "trackId": track_id.strip(),
                "title": title.strip(),
                "artist": " / ".join(artists),
                "status": "queued",
                "stage": "",
                "reason": "",
                "oldPath": "",
                "newPath": "",
                "oldSummary": "",
                "newSummary": "",
                "oldSizeBytes": 0,
                "newSizeBytes": 0,
                "sourceName": "",
                "achievedTarget": "",
                "settled": False,
            })

        batch_id = uuid.uuid4().hex[:16]
        with self._lock:
            if self._state["state"] == "running":
                raise ButlerUpgradeError("busy", 409)
            self._cancel_event.clear()
            self._state = {
                "state": "running",
                "batchId": batch_id,
                "target": target,
                "total": len(plan),
                "done": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
                "current": None,
                "startedAt": int(self._clock()),
                "finishedAt": None,
                "items": state_items,
            }
            self._persist_state()
            self._worker = threading.Thread(
                target=self._run_batch,
                args=(batch_id, target, skips_sound_checks, keep_old_in_trash, plan),
                name="butler-upgrade",
                daemon=True,
            )
            self._worker.start()
        LOGGER.info("BUTLER-UPGRADE-START batch=%s target=%s items=%s", batch_id, target, len(plan))
        return {"ok": True, "batchId": batch_id, "accepted": len(plan)}

    def cancel(self):
        with self._lock:
            if self._state["state"] != "running":
                return {"ok": False}
            self._cancel_event.set()
            return {"ok": True}

    def _set_stage(self, index: int, stage: str):
        with self._lock:
            item = self._state["items"][index]
            item["status"] = "running"
            item["stage"] = stage
            self._state["current"] = {
                "trackId": item["trackId"],
                "title": item["title"],
                "stage": stage,
            }

    def _run_batch(self, batch_id: str, target: str, skips_sound_checks: bool, keep_old_in_trash: bool, plan: list):
        staging = self._files.root / STAGING_DIR_NAME
        try:
            for index, item in enumerate(plan):
                if self._cancel_event.is_set():
                    with self._lock:
                        entry = self._state["items"][index]
                        entry["status"] = "cancelled"
                        entry["stage"] = ""
                        entry["reason"] = "已停止"
                    continue
                outcome = self._process_one(index, batch_id, target, skips_sound_checks, keep_old_in_trash, item, staging)
                with self._lock:
                    entry = self._state["items"][index]
                    entry.update(outcome)
                    self._state["done"] += 1
                    if outcome["status"] == "replaced":
                        self._state["succeeded"] += 1
                    elif outcome["status"] == "skipped":
                        self._state["skipped"] += 1
                    elif outcome["status"] == "failed":
                        self._state["failed"] += 1
                    self._persist_state()
                if outcome["status"] == "replaced" and self._on_done is not None:
                    try:
                        self._on_done(outcome["newPath"])
                    except Exception:
                        pass
        finally:
            with self._lock:
                self._state["state"] = "idle"
                self._state["current"] = None
                self._state["finishedAt"] = int(self._clock())
                self._persist_state()
                LOGGER.info(
                    "BUTLER-UPGRADE-END batch=%s done=%s succeeded=%s skipped=%s failed=%s",
                    batch_id, self._state["done"], self._state["succeeded"],
                    self._state["skipped"], self._state["failed"],
                )

    def _process_one(self, index, batch_id, target, skips_sound_checks, keep_old_in_trash, item, staging: Path):
        self._set_stage(index, "inspecting")
        try:
            old_path, old_rel = self._files.resolve_existing_file(item["nasPath"])
        except ButlerFileError:
            return {"status": "failed", "stage": "", "reason": "NAS 上找不到原来的文件", "settled": False}
        if old_path.suffix.lower() == ".cue":
            return {"status": "skipped", "stage": "", "oldPath": old_rel, "reason": "整轨 CUE 文件不替换", "settled": True}

        old_size = int(old_path.lstat().st_size)
        try:
            old_facts = _facts(old_path)
        except Exception:
            old_facts = {"format": old_path.suffix.lower().lstrip("."), "durationMs": 0, "sampleRate": 0, "bitDepth": 0, "bitRate": 0}
        old_summary = _describe_facts(old_facts, old_size)

        item_target = item.get("target") or target
        candidate_targets = [c.get("target") or item_target for c in item["candidates"]]
        if all(_is_target_met(old_facts, t) for t in candidate_targets):
            return {
                "status": "skipped",
                "stage": "",
                "oldPath": old_rel,
                "oldSummary": old_summary,
                "oldSizeBytes": old_size,
                "reason": f"原文件实测已经是 {old_summary}，不用洗",
                "settled": True,
            }

        if staging.is_symlink():
            return {"status": "failed", "stage": "", "oldPath": old_rel, "oldSummary": old_summary, "oldSizeBytes": old_size, "reason": "暂存目录路径异常", "settled": False}
        staging.mkdir(exist_ok=True)
        self._files.own(staging, folder=True)

        temp_files = []
        try:
            old_dur = int(old_facts.get("durationMs") or 0)
            old_sha = None
            old_cut_loaded = False
            old_cut = None

            accepted = None
            last_error = "没有能下载所选音质的来源"
            last_reject = None

            for cand in item["candidates"]:
                if self._cancel_event.is_set():
                    return {"status": "cancelled", "stage": "", "oldPath": old_rel, "oldSummary": old_summary, "oldSizeBytes": old_size, "reason": "已停止", "settled": False}
                cand_target = cand.get("target") or item_target
                if _is_target_met(old_facts, cand_target):
                    continue

                self._set_stage(index, "downloading")
                url = cand["url"]
                source_name = cand["sourceName"] or ""
                if not url and cand["musicInfo"] is not None:
                    if self._resolver is None:
                        last_error = "容器未配置音源解析器"
                        continue
                    try:
                        resolve_body = {
                            "platform": cand["platform"],
                            "quality": cand["quality"],
                            "exact": True,
                            "purpose": "download",
                            "musicInfo": cand["musicInfo"],
                        }
                        if cand["available"] is not None:
                            resolve_body["available"] = cand["available"]
                        resolved = self._resolver.resolve_nas(resolve_body, OWNER)
                        url = resolved["url"]
                        source_name = resolved.get("sourceName") or source_name
                    except ResolveError as err:
                        last_error = f"音源解析失败（{err.code}）"
                        continue
                if not url:
                    continue
                part = staging / f".xiyue-upgrade-{batch_id}-{index}-{uuid.uuid4().hex[:6]}.part"
                temp_files.append(part)
                try:
                    self._download_url(url, part, cand["userAgent"], cand["referer"])
                except ButlerUpgradeError as err:
                    if err.code == "cancelled":
                        return {"status": "cancelled", "stage": "", "oldPath": old_rel, "oldSummary": old_summary, "oldSizeBytes": old_size, "reason": "已停止", "settled": False}
                    last_error = self._download_error_text(err.code)
                    part.unlink(missing_ok=True)
                    continue
                except urllib.error.HTTPError as err:
                    last_error = f"下载失败（HTTP {err.code}）"
                    part.unlink(missing_ok=True)
                    continue
                except (urllib.error.URLError, OSError):
                    last_error = "下载连接中断"
                    part.unlink(missing_ok=True)
                    continue

                self._set_stage(index, "verifying")
                try:
                    new_facts = _facts(part)
                except Exception:
                    last_error = "下载到的文件不是有效音频"
                    part.unlink(missing_ok=True)
                    continue
                new_size = int(part.lstat().st_size)
                new_summary = _describe_facts(new_facts, new_size)

                def _reject(reason_text: str):
                    nonlocal last_reject
                    last_reject = {
                        "status": "skipped",
                        "stage": "",
                        "oldPath": old_rel,
                        "oldSummary": old_summary,
                        "newSummary": new_summary,
                        "oldSizeBytes": old_size,
                        "newSizeBytes": new_size,
                        "sourceName": source_name,
                        "reason": reason_text,
                        "settled": True,
                    }
                    part.unlink(missing_ok=True)

                if not _accepts_new_quality(new_facts, cand_target):
                    _reject(f"下载到的是 {new_summary}，未达到所选音质")
                    continue

                if new_size == old_size:
                    if old_sha is None:
                        old_sha = _sha256_file(old_path)
                    if _sha256_file(part) == old_sha:
                        _reject("下载到的和原来是同一个文件")
                        continue

                new_dur = int(new_facts.get("durationMs") or 0)
                if old_dur > 0 and new_dur > 0 and abs(new_dur - old_dur) > 10_000:
                    _reject(f"时长相差过大（原 {old_dur // 1000}s / 新 {new_dur // 1000}s），不是同一版本")
                    continue

                corr = _same_recording_correlation(old_path, part)
                if corr is not None and corr < 0.65:
                    _reject(f"和原来的不像同一个录音（相似度 {round(corr * 100)}%）")
                    continue

                if not skips_sound_checks:
                    if not old_cut_loaded:
                        old_cut = _estimate_cutoff_hz(old_path, int(old_facts.get("sampleRate") or 0), old_dur)
                        old_cut_loaded = True
                    new_cut = _estimate_cutoff_hz(part, int(new_facts.get("sampleRate") or 0), new_dur)
                    if new_cut is not None:
                        old_is_lossy = (old_facts.get("format") or "").lower() not in LOSSLESS_FORMATS
                        if cand_target == "lossless" and old_is_lossy:
                            floor = max((old_cut or 0.0) + 500.0, 17500.0)
                            if new_cut < floor:
                                _reject(f"高频只到 {new_cut / 1000.0:.1f} kHz，像是有损转的（可开“不核对音质”跳过）")
                                continue
                        elif cand_target in ("hires", "master") and int(new_facts.get("sampleRate") or 0) > 48_000:
                            if new_cut < 22500.0:
                                _reject(f"高频只到 {new_cut / 1000.0:.1f} kHz，像是升频文件（可开“不核对音质”跳过）")
                                continue

                accepted = (part, new_facts, source_name, cand_target)
                break

            if accepted is None:
                if last_reject is not None:
                    return last_reject
                return {
                    "status": "failed",
                    "stage": "",
                    "oldPath": old_rel,
                    "oldSummary": old_summary,
                    "oldSizeBytes": old_size,
                    "reason": last_error,
                    "settled": False,
                }

            downloaded_path, new_facts, source_name, achieved_target = accepted

            if self._cancel_event.is_set():
                return {"status": "cancelled", "stage": "", "oldPath": old_rel, "oldSummary": old_summary, "oldSizeBytes": old_size, "reason": "已停止", "settled": False}

            self._set_stage(index, "tagging")
            new_format = new_facts["format"]
            # Rename staged part with real extension so mutagen / loudness recognize format cleanly.
            typed_part = downloaded_path.with_suffix("." + new_format)
            if typed_part != downloaded_path:
                os.replace(downloaded_path, typed_part)
                temp_files.append(typed_part)
                downloaded_path = typed_part

            _apply_preserved_tags(downloaded_path, old_path, item)
            if new_format in loudness.WRITABLE:
                try:
                    gains = loudness.measure_audio(downloaded_path)
                    loudness.write_tags(downloaded_path, new_format, gains, track_only=True)
                except Exception as err:
                    LOGGER.warning("BUTLER-UPGRADE-LOUD-WARN %s %s", old_path.name, loudness._reason(err))

            self._files.own(downloaded_path, folder=False)
            final_size = int(downloaded_path.lstat().st_size)
            final_summary = _describe_facts(new_facts, final_size)

            self._set_stage(index, "replacing")
            target_path = old_path.with_suffix("." + new_format)
            same_name = target_path == old_path
            if not same_name and target_path.exists():
                return {
                    "status": "failed",
                    "stage": "",
                    "oldPath": old_rel,
                    "oldSummary": old_summary,
                    "newSummary": final_summary,
                    "oldSizeBytes": old_size,
                    "newSizeBytes": final_size,
                    "sourceName": source_name,
                    "reason": f"目标位置已有同名文件 {target_path.name}",
                    "settled": False,
                }

            stamp = int(self._clock())
            trash_path = self._files.unique_trash_path(old_path, stamp)
            if same_name:
                os.replace(old_path, trash_path)
                try:
                    os.replace(downloaded_path, target_path)
                except OSError:
                    try:
                        os.replace(trash_path, old_path)
                    except OSError:
                        pass
                    return {
                        "status": "failed",
                        "stage": "",
                        "oldPath": old_rel,
                        "oldSummary": old_summary,
                        "oldSizeBytes": old_size,
                        "reason": "替换写入磁盘失败，原文件已保留",
                        "settled": False,
                    }
            else:
                try:
                    os.replace(downloaded_path, target_path)
                except OSError:
                    return {
                        "status": "failed",
                        "stage": "",
                        "oldPath": old_rel,
                        "oldSummary": old_summary,
                        "oldSizeBytes": old_size,
                        "reason": "新文件写入磁盘失败，原文件未动",
                        "settled": False,
                    }
                try:
                    os.replace(old_path, trash_path)
                except OSError:
                    try:
                        target_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return {
                        "status": "failed",
                        "stage": "",
                        "oldPath": old_rel,
                        "oldSummary": old_summary,
                        "oldSizeBytes": old_size,
                        "reason": "旧文件移出失败，已回滚",
                        "settled": False,
                    }

            if not keep_old_in_trash:
                try:
                    trash_path.unlink(missing_ok=True)
                    self._files._cleanup_empty_trash_dir(trash_path.parent)
                except OSError:
                    pass

            new_rel = target_path.relative_to(self._files.root).as_posix()
            return {
                "status": "replaced",
                "stage": "",
                "reason": "",
                "oldPath": old_rel,
                "newPath": new_rel,
                "oldSummary": old_summary,
                "newSummary": final_summary,
                "oldSizeBytes": old_size,
                "newSizeBytes": final_size,
                "sourceName": source_name,
                "achievedTarget": achieved_target,
                "settled": True,
            }
        finally:
            for path in temp_files:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _download_url(self, url: str, part: Path, user_agent: str, referer: str):
        self._check(url)
        headers = {"User-Agent": user_agent or USER_AGENT}
        if referer:
            headers["Referer"] = referer
        opener = urllib.request.build_opener(_Guard(self._check))
        with opener.open(urllib.request.Request(url, headers=headers), timeout=30) as response:
            try:
                total = int(response.headers.get("Content-Length") or 0)
            except ValueError:
                total = 0
            if total > MAX_AUDIO_BYTES:
                raise ButlerUpgradeError("too_large")
            if shutil.disk_usage(part.parent).free < total + SPARE_BYTES:
                raise ButlerUpgradeError("no_space")
            received = 0
            with open(part, "xb") as out:
                while True:
                    if self._cancel_event.is_set():
                        raise ButlerUpgradeError("cancelled")
                    chunk = response.read(262144)
                    if not chunk:
                        break
                    if received == 0 and not _is_audio(chunk):
                        raise ButlerUpgradeError("not_audio")
                    received += len(chunk)
                    if received > MAX_AUDIO_BYTES:
                        raise ButlerUpgradeError("too_large")
                    out.write(chunk)
        if received < MIN_AUDIO_BYTES or (total and received != total):
            raise ButlerUpgradeError("incomplete")

    @staticmethod
    def _download_error_text(code: str) -> str:
        mapping = {
            "bad_url": "下载地址无效",
            "too_large": "文件过大，已跳过",
            "no_space": "NAS 剩余空间不足",
            "not_audio": "下载到的不是音频文件",
            "incomplete": "下载未完整完成",
        }
        return mapping.get(code, f"下载失败（{code}）")
