import json
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path

from . import ANALYZER_VERSION
from .features import extract_features
from .output import write_output
from .similarity import build_similarity
from .tags import read_tags


AUDIO_EXTENSIONS = {
    ".flac", ".mp3", ".m4a", ".aac", ".wav", ".aiff",
    ".ape", ".ogg", ".opus", ".wma", ".dsf",
}


def _analyze(path):
    return extract_features(path), read_tags(path)


def scan(music, data, out, workers=2):
    music = Path(music)
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(data / "cache.sqlite")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS tracks(
                path TEXT PRIMARY KEY, size INTEGER, mtime REAL, version TEXT,
                features TEXT, tags TEXT, error TEXT
            )"""
        )
        cached = {
            row["path"]: row for row in connection.execute("SELECT * FROM tracks")
        }
        paths = set()
        pending = []
        for directory, directories, filenames in os.walk(music):
            directories[:] = sorted(name for name in directories if not name.startswith("."))
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                relative = path.relative_to(music).as_posix()
                paths.add(relative)
                stat = path.stat()
                previous = cached.get(relative)
                if previous is not None and (
                    previous["size"], previous["mtime"], previous["version"]
                ) == (stat.st_size, stat.st_mtime, ANALYZER_VERSION):
                    continue
                pending.append((path, relative, stat.st_size, stat.st_mtime))

        for removed in cached.keys() - paths:
            connection.execute("DELETE FROM tracks WHERE path = ?", (removed,))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_analyze, str(path)): (relative, size, mtime)
                for path, relative, size, mtime in pending
            }
            for future in as_completed(futures):
                relative, size, mtime = futures[future]
                try:
                    features, tags = future.result()
                    features_json = json.dumps(features, allow_nan=False)
                    tags_json = json.dumps(tags, ensure_ascii=False, allow_nan=False)
                    error = None
                except Exception as exception:
                    features_json = tags_json = None
                    error = type(exception).__name__
                connection.execute(
                    """INSERT INTO tracks(path, size, mtime, version, features, tags, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        size=excluded.size, mtime=excluded.mtime, version=excluded.version,
                        features=excluded.features, tags=excluded.tags, error=excluded.error""",
                    (relative, size, mtime, ANALYZER_VERSION, features_json, tags_json, error),
                )
                connection.commit()
        connection.commit()
        tracks = [
            {
                "path": row["path"],
                "size": row["size"],
                "mtime": row["mtime"],
                "features": json.loads(row["features"]),
                "tags": json.loads(row["tags"]),
            }
            for row in connection.execute(
                "SELECT * FROM tracks WHERE error IS NULL ORDER BY path"
            )
        ]
    vectors, neighbors = build_similarity(tracks)
    write_output(out, tracks, vectors, neighbors)
    return {"analyzed": len(pending), "tracks": len(tracks)}
