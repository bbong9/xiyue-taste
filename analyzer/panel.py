import gzip
import json
import sqlite3
import threading
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class PanelState:
    """Thread-safe scan state shared by the scanner and HTTP threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self.scanRequested = threading.Event()
        self._values = {
            "state": "idle",
            "done": 0,
            "total": 0,
            "startedAt": None,
            "finishedAt": None,
            "lastAnalyzed": None,
            "lastExported": None,
            "nextScanAt": None,
            "error": None,
        }

    def snapshot(self):
        with self._lock:
            return self._values.copy()

    def begin_scan(self):
        with self._lock:
            self._values.update(
                state="scanning", done=0, total=0,
                startedAt=datetime.now(timezone.utc).isoformat(),
                nextScanAt=None, error=None,
            )

    def set_progress(self, done, total):
        with self._lock:
            self._values.update(done=done, total=total)

    def finish_scan(self, analyzed, exported):
        with self._lock:
            self._values.update(
                state="idle", finishedAt=datetime.now(timezone.utc).isoformat(),
                lastAnalyzed=analyzed, lastExported=exported, error=None,
            )

    def fail_scan(self, exception):
        with self._lock:
            self._values.update(
                state="idle", finishedAt=datetime.now(timezone.utc).isoformat(),
                lastAnalyzed=None, lastExported=None,
                error=type(exception).__name__,
            )

    def set_next(self, timestamp):
        with self._lock:
            self._values["nextScanAt"] = datetime.fromtimestamp(
                timestamp, timezone.utc
            ).isoformat()

    def request_scan(self):
        self.scanRequested.set()

    def wait_for_next(self, seconds):
        self.scanRequested.wait(seconds)
        self.scanRequested.clear()


class OutputIndex:
    """Cache exported tracks by mtime, without retaining their vectors."""

    def __init__(self, out_dir):
        self._path = Path(out_dir) / "xiyue-taste-v1.json.gz"
        self._lock = threading.Lock()
        self._mtime = None
        self._tracks = []

    def load(self):
        with self._lock:
            try:
                mtime = self._path.stat().st_mtime_ns
            except FileNotFoundError:
                self._mtime = None
                self._tracks = []
                return self._tracks
            if mtime != self._mtime:
                with gzip.open(self._path, "rt", encoding="utf-8") as handle:
                    tracks = json.load(handle)["tracks"]
                for track in tracks:
                    del track["vector"]
                self._tracks = tracks
                self._mtime = mtime
            return self._tracks

    @staticmethod
    def _histogram(values, start, stop, width):
        if not values:
            return {"min": None, "max": None, "bins": []}
        bins = [
            {"from": value, "to": value + width, "count": 0}
            for value in range(start, stop, width)
        ]
        for value in values:
            bucket = min(max(int((value - start) // width), 0), len(bins) - 1)
            bins[bucket]["count"] += 1
        return {"min": min(values), "max": max(values), "bins": bins}

    @staticmethod
    def _ranking(values):
        counts = Counter(value for value in values if value)
        return [
            {"name": name, "count": count}
            for name, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:8]
        ]

    def stats(self):
        tracks = self.load()
        return {
            "trackCount": len(tracks),
            "bpm": self._histogram(
                [track["bpm"] for track in tracks if track["bpm"] is not None],
                60, 180, 10,
            ),
            "loudness": self._histogram(
                [track["loudnessLUFS"] for track in tracks
                 if track["loudnessLUFS"] is not None],
                -24, -4, 2,
            ),
            "languages": self._ranking(track["lyricsLanguage"] for track in tracks),
            "genres": self._ranking(track["genre"] for track in tracks),
            "artists": self._ranking(
                artist for track in tracks for artist in (track["artists"] or [])
            ),
        }

    def tracks(self, q, offset, limit):
        query = q.casefold()
        fields = ("title", "artists", "album", "bpm", "loudnessLUFS", "durationSec")
        matches = [
            {"index": index, **{key: track[key] for key in fields}}
            for index, track in enumerate(self.load())
            if query in " ".join((
                track["title"], " ".join(track["artists"] or []), track["album"] or ""
            )).casefold()
        ]
        offset = max(0, offset)
        limit = max(0, min(limit, 100))
        return {"total": len(matches), "items": matches[offset:offset + limit]}

    def track(self, index):
        tracks = self.load()
        if index < 0 or index >= len(tracks):
            return None
        track = tracks[index]
        fields = (
            "title", "artists", "album", "genre", "year", "durationSec",
            "bpm", "loudnessLUFS", "lyricsLanguage", "path",
        )
        return {
            **{key: track[key] for key in fields},
            "neighbors": [
                {
                    "index": neighbor, "title": tracks[neighbor]["title"],
                    "artists": tracks[neighbor]["artists"], "score": score,
                }
                for neighbor, score in track["neighbors"]
            ],
        }

    def failures(self, data_dir):
        path = Path(data_dir) / "cache.sqlite"
        if not path.exists():
            return []
        with closing(sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True
        )) as connection:
            return [
                {"path": path, "error": error}
                for path, error in connection.execute(
                    "SELECT path, error FROM tracks "
                    "WHERE error IS NOT NULL AND error != '' ORDER BY path"
                )
            ]
