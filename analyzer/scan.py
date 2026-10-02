import faulthandler
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, TimeoutError, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import closing
from pathlib import Path

from . import ANALYZER_VERSION
from .features import extract_features
from .log import LOGGER
from .output import write_output
from .similarity import build_similarity
from .tags import read_tags


AUDIO_EXTENSIONS = {
    ".flac", ".mp3", ".m4a", ".aac", ".wav", ".aiff",
    ".ape", ".ogg", ".opus", ".wma", ".dsf",
}


class WorkersKeepDying(Exception):
    pass


def _exit_codes(executor):
    try:
        processes = list((getattr(executor, "_processes", None) or {}).values())
        for process in processes:
            process.join(timeout=2)
        return [process.exitcode for process in processes]
    except Exception:
        return []


def _exit_hint(code):
    return {
        -9: "被系统强制结束，多半是内存不够",
        -4: "非法指令，处理器不支持分析库要用的指令集",
        -11: "段错误",
    }.get(code, "")


def _probe():
    return "ok"


def _import_failure():
    """Why a fresh interpreter cannot load the analysis libraries: the last of
    what it printed before it stopped."""
    try:
        outcome = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", "import analyzer.scan"],
            capture_output=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if outcome.returncode == 0:
        return ""
    return "import exit=%s: %s" % (
        outcome.returncode, " | ".join(outcome.stderr.decode("utf-8", "replace").strip().splitlines()[-6:])[:600],
    )


def probe_worker():
    try:
        executor = ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=faulthandler.enable,
        )
        try:
            try:
                return executor.submit(_probe).result(timeout=120)
            except BrokenProcessPool:
                codes = _exit_codes(executor)
                code = codes[0] if codes else "未知"
                return f"died exit={code} {_exit_hint(code)} {_import_failure()}".rstrip()
            except TimeoutError:
                for process in list((getattr(executor, "_processes", None) or {}).values()):
                    process.terminate()
                return "timeout"
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    except Exception as exception:
        return f"error {type(exception).__name__}"


def _analyze(path):
    return extract_features(path), read_tags(path)


def scan(music, data, out, workers=2, progress=None, analyze=None):
    started = time.monotonic()
    if analyze is None:
        analyze = _analyze
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
                ) == (stat.st_size, stat.st_mtime, ANALYZER_VERSION) and previous["error"] != "BrokenProcessPool":
                    continue
                pending.append((path, relative, stat.st_size, stat.st_mtime))

        for removed in cached.keys() - paths:
            connection.execute("DELETE FROM tracks WHERE path = ?", (removed,))
        LOGGER.info("SCAN-START pending=%s cached=%s workers=%s", len(pending), len(cached), workers)
        if progress is not None:
            progress(0, len(pending))
        done = 0
        deaths = 0
        failed = 0

        def record(item, result_or_exception):
            nonlocal done, deaths, failed
            _, relative, size, mtime = item
            try:
                if isinstance(result_or_exception, Exception):
                    raise result_or_exception
                features, tags = result_or_exception
                features_json = json.dumps(features, allow_nan=False)
                tags_json = json.dumps(tags, ensure_ascii=False, allow_nan=False)
                error = None
                deaths = 0
            except Exception as exception:
                features_json = tags_json = None
                failed += 1
                if isinstance(exception, BrokenProcessPool):
                    error = "WorkerDied"
                    deaths += 1
                    codes = _exit_codes(executor)
                    LOGGER.error(
                        "WORKER-DIED %s exit=%s %s", relative, codes,
                        " ".join(_exit_hint(code) for code in codes if _exit_hint(code)),
                    )
                else:
                    error = type(exception).__name__
                    LOGGER.warning("ANALYZE-FAIL %s %s: %s", relative, error, str(exception)[:200])
            connection.execute(
                """INSERT INTO tracks(path, size, mtime, version, features, tags, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size=excluded.size, mtime=excluded.mtime, version=excluded.version,
                    features=excluded.features, tags=excluded.tags, error=excluded.error""",
                (relative, size, mtime, ANALYZER_VERSION, features_json, tags_json, error),
            )
            connection.commit()
            done += 1
            if progress is not None:
                progress(done, len(pending))
            if deaths >= 5:
                LOGGER.error("SCAN-ABORT WorkersKeepDying")
                raise WorkersKeepDying()

        context = multiprocessing.get_context("spawn")
        next_item = 0
        while next_item < len(pending):
            suspects = []
            futures = {}
            broken = False
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=faulthandler.enable)
            try:
                while next_item < len(pending) or futures:
                    while next_item < len(pending) and len(futures) < workers:
                        item = pending[next_item]
                        try:
                            future = executor.submit(analyze, str(item[0]))
                        except BrokenProcessPool:
                            broken = True
                            break
                        futures[future] = item
                        next_item += 1
                    if broken:
                        break
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        item = futures.pop(future)
                        try:
                            result = future.result()
                        except BrokenProcessPool:
                            suspects.append(item)
                            broken = True
                            continue
                        except Exception as exception:
                            result = exception
                        record(item, result)
                    if broken:
                        break
            finally:
                if broken:
                    codes = _exit_codes(executor)
                    LOGGER.error(
                        "POOL-BROKEN running=%s exit=%s %s",
                        ",".join(item[1] for item in [*suspects, *futures.values()]), codes,
                        " ".join(_exit_hint(code) for code in codes if _exit_hint(code)),
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=True, cancel_futures=True)

            for future, item in futures.items():
                try:
                    result = future.result()
                except BrokenProcessPool:
                    suspects.append(item)
                    continue
                except Exception as exception:
                    result = exception
                record(item, result)

            for item in suspects:
                executor = ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=faulthandler.enable)
                try:
                    try:
                        result = executor.submit(analyze, str(item[0])).result()
                    except Exception as exception:
                        result = exception
                    record(item, result)
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
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
    LOGGER.info(
        "SCAN-END analyzed=%s ok=%s failed=%s seconds=%.2f",
        len(pending), done - failed, failed, time.monotonic() - started,
    )
    return {"analyzed": len(pending), "tracks": len(tracks)}
