"""Loudness measurement, safe tag writes and the owner's asynchronous API."""

import errno
import gzip
import ipaddress
import json
import logging
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TXXX

from analyzer import __main__ as command
from analyzer import loudness
from analyzer.access import AccessSettings
from analyzer.accounts import AccountStore
from analyzer.downloads import Downloads
from analyzer.panel import OutputIndex, PanelState
from analyzer.serve import make_server
from test_analyzer import _AUDIO, _file_server, _wait_for
from test_sources import _call


def _summary(integrated=-27.8, peak=-24.1):
    return f"""
    Integrated loudness:
      I:         {integrated} LUFS
      Threshold: -37.8 LUFS
    Loudness range:
      LRA:        3.0 LU
    True peak:
      Peak:      {peak} dBFS
    """


def _ffmpeg(monkeypatch, outputs=None):
    calls = []

    def run(args, **options):
        path = Path(args[args.index("-i") + 1])
        calls.append(path)
        assert args == [
            "ffmpeg", "-hide_banner", "-nostats", "-threads", "1", "-i", str(path),
            "-map", "0:a:0", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-",
        ]
        assert options["timeout"] == 600
        return subprocess.CompletedProcess(args, 0, stdout="", stderr=(outputs or {}).get(path.name, _summary()))

    monkeypatch.setattr(loudness.subprocess, "run", run)
    return calls


def _flac(path, seconds=1, album=None, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.sin(2 * np.pi * 440 * np.arange(8000 * seconds) / 8000) * 0.2
    sf.write(path, samples, 8000, format="FLAC", subtype="PCM_16")
    audio = FLAC(path)
    audio["title"], audio["artist"], audio["comment"] = ["歌名"], ["歌手"], ["不改的注释"]
    if album is not None:
        audio["album"] = [album]
    for key, value in (tags or {}).items():
        audio[key] = [value]
    audio.save()
    return path


@pytest.fixture
def folders(tmp_path):
    library, data, out = (tmp_path / name for name in ("library", "data", "out"))
    library.mkdir()
    return library, data, out


def _rows(data):
    return {row["path"]: row for row in loudness._rows(data)}


def _exported(out):
    with gzip.open(out / "xiyue-loudness-v1.json.gz", "rt", encoding="utf-8") as file:
        return json.load(file)


def test_measure_parses_last_summary_and_time(monkeypatch):
    _ffmpeg(monkeypatch, {
        "song.part": _summary(-20, -1) + "\ntime=00:01:00.00\ntime=00:05:25.25\n" + _summary(),
    })
    assert loudness.measure_audio(Path("song.part")) == {
        "track_gain": 9.8, "track_peak": pytest.approx(0.062373), "duration": 325.25,
    }


@pytest.mark.parametrize("stderr,code,reason", [
    (_summary(-70, "-inf"), 0, "silent"),
    ("no audio summary", 0, "no_summary"),
    (_summary(), 1, "ffmpeg_exit_1"),
    (_summary("nan", -1), 0, "no_summary"),
])
def test_measure_rejects_bad_summary(monkeypatch, stderr, code, reason):
    monkeypatch.setattr(loudness.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code, "", stderr))
    with pytest.raises(loudness.LoudnessError, match=reason):
        loudness.measure_audio("song.flac")


def test_measure_times_out(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 600)

    monkeypatch.setattr(loudness.subprocess, "run", timeout)
    with pytest.raises(loudness.LoudnessError, match="timeout"):
        loudness.measure_audio("song.flac")


@pytest.mark.parametrize("value,peak,expected", [
    (" +2.5 dB ", False, 2.5), ("-6.2", False, -6.2), (b"1 DB", False, 1),
    ("nan", False, None), ("inf", False, None), ("61", False, None), ("-61", False, None),
    ("-0.1", True, None), ("10.01", True, None), ("0.062373", True, 0.062373),
])
def test_tag_value_validation(value, peak, expected):
    assert loudness.parse_tag(value, peak) == expected


def test_reads_flac_and_id3_tags_case_insensitively(tmp_path):
    flac = _flac(tmp_path / "song.flac", tags={
        "replaygain_track_gain": " +2.50 db ", "ReplayGain_Track_Peak": "0.9",
        "replaygain_album_gain": "-4.25", "replaygain_album_peak": "nan",
    })
    assert loudness.read_gains(flac, "flac") == {
        "track_gain": 2.5, "track_peak": 0.9, "album_gain": -4.25, "album_peak": None,
    }
    mp3 = tmp_path / "song.mp3"
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="RePlAyGaIn_TrAcK_GaIn", text=[" -1.25 dB "]))
    tags.add(TXXX(encoding=3, desc="replaygain_track_peak", text=["12"]))
    tags.save(mp3)
    assert loudness.read_gains(mp3, "mp3") == {
        "track_gain": -1.25, "track_peak": None, "album_gain": None, "album_peak": None,
    }


def test_incremental_measure_skips_unchanged_and_hidden_and_prunes_deleted(folders, monkeypatch):
    library, data, out = folders
    a, b = _flac(library / "a.FLAC"), _flac(library / "sub" / "b.flac")
    original = a.read_bytes()
    for hidden in (".trash/a.flac", ".hidden/a.flac", "@eaDir/a.flac", "#recycle/a.flac", ".song.flac"):
        _flac(library / hidden)
    calls = _ffmpeg(monkeypatch)
    progress = []
    assert loudness.measure(library, data, out, progress=lambda *v: progress.append(v))["measured"] == 2
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert loudness.measure(library, data, out)["measured"] == 0
    assert len(calls) == 2
    old = a.stat()
    os.utime(a, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))
    assert loudness.measure(library, data, out)["measured"] == 1
    assert calls[-1] == a
    b.unlink()
    loudness.measure(library, data, out)
    assert list(_rows(data)) == ["a.FLAC"]
    assert a.read_bytes() == original
    assert loudness.read_gains(a, "flac")["track_gain"] is None


def test_tags_skip_decoding_remeasure_overrides_and_errors_retry(folders, monkeypatch):
    library, data, out = folders
    path = _flac(library / "tag.flac", tags={"replaygain_track_gain": "-2 dB"})
    original = path.read_bytes()
    calls = _ffmpeg(monkeypatch, {"bad.flac": "bad"})
    bad = _flac(library / "bad.flac")
    result = loudness.measure(library, data, out)
    assert (result["fromTags"], result["failed"], result["measured"]) == (1, 1, 0)
    assert calls == [bad]
    result = loudness.measure(library, data, out)
    assert result["failed"] == 1 and calls == [bad, bad]
    result = loudness.measure(library, data, out, remeasure=True)
    assert (result["measured"], result["failed"]) == (1, 1)
    assert _rows(data)["tag.flac"]["source"] == "measured"
    assert _rows(data)["tag.flac"]["track_gain"] == 9.8
    assert path.read_bytes() == original


def test_album_energy_weighting_preserves_existing_album_tags(folders, monkeypatch):
    library, data, out = folders
    _flac(library / "album" / "a.flac", seconds=1, album="  专辑 ")
    _flac(library / "album" / "b.flac", seconds=3, album="专辑")
    _flac(library / "album" / "tag.flac", seconds=2, album="专辑", tags={
        "REPLAYGAIN_TRACK_GAIN": "-2 dB", "REPLAYGAIN_ALBUM_GAIN": "+1 dB", "REPLAYGAIN_ALBUM_PEAK": "0.8",
    })
    _flac(library / "plain.flac")
    _flac(library / "solo.flac", album="另一张")
    _ffmpeg(monkeypatch, {"a.flac": _summary(-28, -20), "b.flac": _summary(-18, -10)})
    loudness.measure(library, data, out)
    rows = _rows(data)
    expected = round(-18 - 10 * math.log10((10 ** -2.8 + 3 * 10 ** -1.8 + 2 * 10 ** -1.6) / 6), 2)
    for name in ("a", "b"):
        row = rows[f"album/{name}.flac"]
        assert row["album_key"] == "album\x1f专辑"
        assert row["album_gain"] == expected and row["album_peak"] == pytest.approx(0.316228)
    assert rows["album/tag.flac"]["album_gain"] == 1
    assert rows["album/tag.flac"]["album_peak"] == 0.8
    assert rows["plain.flac"]["album_key"] is None and rows["plain.flac"]["album_gain"] is None
    assert rows["solo.flac"]["album_gain"] == rows["solo.flac"]["track_gain"]


def test_computed_album_gain_on_tagged_track_is_recomputed_when_group_changes(folders):
    library, data, out = folders
    _flac(library / "a.flac", album="Album", tags={"REPLAYGAIN_TRACK_GAIN": "+10 dB"})
    loudness.measure(library, data, out)
    assert _rows(data)["a.flac"]["album_gain"] == 10
    _flac(library / "b.flac", album=" album ", tags={"REPLAYGAIN_TRACK_GAIN": "0 dB"})
    loudness.measure(library, data, out)
    rows = _rows(data)
    assert rows["a.flac"]["album_key"] == rows["b.flac"]["album_key"]
    assert rows["a.flac"]["album_gain"] == rows["b.flac"]["album_gain"] != 10


def test_write_flac_preserves_audio_metadata_times_and_permissions(folders, monkeypatch):
    library, data, out = folders
    path = _flac(library / "a.flac", album="专辑", tags={"replaygain_track_gain": "-1 dB"})
    path.chmod(0o640)
    samples, rate = sf.read(path)
    md5, total_samples = FLAC(path).info.md5_signature, FLAC(path).info.total_samples
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out, remeasure=True)
    old = path.stat()
    result = loudness.write(library, data, out, library)
    assert result == {"written": ["a.flac"], "failed": [], "skipped": []}
    new = path.stat()
    assert (new.st_mtime_ns, new.st_atime_ns) == (old.st_mtime_ns, old.st_atime_ns)
    assert new.st_mode == old.st_mode
    audio = FLAC(path)
    assert audio.info.md5_signature == md5 and audio.info.total_samples == total_samples
    assert audio["title"] == ["歌名"] and audio["artist"] == ["歌手"] and audio["album"] == ["专辑"]
    assert audio["comment"] == ["不改的注释"]
    assert audio["REPLAYGAIN_TRACK_GAIN"] == ["+9.80 dB"]
    assert audio["REPLAYGAIN_TRACK_PEAK"] == ["0.062373"]
    assert audio["REPLAYGAIN_ALBUM_GAIN"] == ["+9.80 dB"]
    assert np.array_equal(sf.read(path)[0], samples) and sf.info(path).samplerate == rate
    assert not list(library.glob(".xiyue-rg-*"))
    assert _rows(data)["a.flac"]["source"] == "written"
    assert _exported(out)["tracks"][0]["source"] == "written"


def test_write_flac_with_long_utf8_name(folders, monkeypatch):
    library, data, out = folders
    name = "歌あ" * 41 + "abc.flac"
    assert 250 < len(name.encode("utf-8")) <= 255
    path = _flac(library / name)
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    assert loudness.write(library, data, out, library) == {
        "written": [name], "failed": [], "skipped": [],
    }
    assert path.name == name and path.is_file()
    assert FLAC(path)["REPLAYGAIN_TRACK_GAIN"] == ["+9.80 dB"]
    assert not list(library.glob(".xiyue-rg-*"))


@pytest.mark.parametrize(("error", "reason"), [
    (OSError(errno.ENAMETOOLONG, "long"), "name_too_long"),
    (OSError(errno.EACCES, "denied"), "permission_denied"),
    (OSError(errno.EPERM, "denied"), "permission_denied"),
    (OSError(errno.ENOSPC, "full"), "no_space"),
    (OSError(errno.EDQUOT, "quota"), "no_space"),
    (OSError(errno.EROFS, "read only"), "read_only"),
    (OSError(errno.EIO, "other"), "OSError"),
    (OSError(errno.ENOENT, "missing"), "FileNotFoundError"),
    (loudness.LoudnessError("silent"), "silent"),
])
def test_reason_classifies_common_os_errors(error, reason):
    assert loudness._reason(error) == reason


def test_snapshot_lists_problems_without_changing_existing_fields(folders, monkeypatch):
    library, data, out = folders
    _flac(library / "a-measure.flac")
    _flac(library / "b-write.flac")
    _ffmpeg(monkeypatch, {"a-measure.flac": _summary(-70, -24)})
    loudness.measure(library, data, out)
    service = loudness.Loudness(library, data, out, library)
    before = service.snapshot()

    def fail_write(*args, **kwargs):
        raise OSError(errno.ENAMETOOLONG, "long")

    with monkeypatch.context() as patch:
        patch.setattr(loudness, "write_tags", fail_write)
        assert loudness.write(library, data, out, library)["failed"] == [
            {"path": "b-write.flac", "reason": "name_too_long"},
        ]
    with loudness._database(data) as connection:
        connection.execute("UPDATE loudness SET write_error='permission_denied' WHERE path='a-measure.flac'")
        connection.commit()
    state = loudness._read_state(data)
    snapshot = service.snapshot()
    assert snapshot["problems"] == [
        {"path": "a-measure.flac", "stage": "measure", "reason": "silent"},
        {"path": "b-write.flac", "stage": "write", "reason": "name_too_long"},
    ]
    assert snapshot["counts"] == before["counts"] == {
        "fromTags": 0, "measured": 1, "written": 0, "failed": 1, "unsupported": 0,
    }
    assert snapshot["pending"] == before["pending"] == ["b-write.flac"]
    assert {key: snapshot[key] for key in state} == state
    assert loudness.write(library, data, out, library)["written"] == ["b-write.flac"]
    assert service.snapshot()["problems"] == [
        {"path": "a-measure.flac", "stage": "measure", "reason": "silent"},
    ]
    with loudness._database(data) as connection:
        for number in reversed(range(201)):
            loudness._save_row(connection, {"path": f"limit/{number:03}.flac", "error": "silent"})
    problems = service.snapshot()["problems"]
    assert len(problems) == 200
    assert [item["path"] for item in problems] == ["a-measure.flac"] + [
        f"limit/{number:03}.flac" for number in range(199)
    ]


def test_write_changed_file_is_skipped_and_marked_for_remeasure(folders, monkeypatch):
    library, data, out = folders
    path = _flac(library / "a.flac")
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    original = path.read_bytes()
    assert loudness.write(library, data, out, library) == {
        "written": [], "failed": [], "skipped": [{"path": "a.flac", "reason": "changed"}],
    }
    assert path.read_bytes() == original and _rows(data)["a.flac"]["size"] is None
    assert loudness.measure(library, data, out)["measured"] == 1


def test_validation_failure_leaves_original_and_cleans_temporary_file(folders, monkeypatch):
    library, data, out = folders
    path = _flac(library / "a.flac")
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    original = path.read_bytes()
    monkeypatch.setattr(loudness, "_validate", lambda *args: False)
    result = loudness.write(library, data, out, library)
    assert result["failed"] == [{"path": "a.flac", "reason": "validation_failed"}]
    assert path.read_bytes() == original and not list(library.glob(".xiyue-rg-*"))
    row = _rows(data)["a.flac"]
    assert row["write_error"] == "validation_failed" and row["source"] == "measured"
    assert row["track_gain"] == 9.8 and row["error"] is None


def test_write_updates_only_matching_taste_cache_stats(folders, monkeypatch):
    library, data, out = folders
    path = _flac(library / "最近在听" / "白智英" / "x.flac")
    # Avoid FLAC padding masking the size change we need to exercise.
    FLAC(path).save(padding=lambda info: 0)
    old = path.stat()
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    with sqlite3.connect(data / "cache.sqlite") as db:
        db.execute("CREATE TABLE tracks(path TEXT PRIMARY KEY, size INTEGER, mtime REAL, version TEXT, features TEXT, tags TEXT, error TEXT)")
        for name in ("白智英/x.flac", "最近在听/白智英/x.flac", "x.flac", "other.flac", "智英/x.flac"):
            db.execute("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?)", (
                name, old.st_size, old.st_mtime, "unchanged-version", '{"vector":[1,2]}', '{"title":"歌"}', None,
            ))
        db.execute("INSERT INTO tracks VALUES ('bad-stats', 1, 1, 'v', 'features', 'tags', 'error')")
    assert loudness.write(library, data, out, library / "最近在听")["written"] == ["最近在听/白智英/x.flac"]
    new = path.stat()
    assert new.st_size != old.st_size
    with sqlite3.connect(data / "cache.sqlite") as db:
        for name, size, mtime, version, features, tags, error in db.execute("SELECT * FROM tracks WHERE path != 'bad-stats'"):
            assert size == (new.st_size if name in ("白智英/x.flac", "最近在听/白智英/x.flac", "x.flac") else old.st_size)
            assert mtime == old.st_mtime
            assert (version, features, tags, error) == ("unchanged-version", '{"vector":[1,2]}', '{"title":"歌"}', None)
        assert db.execute("SELECT * FROM tracks WHERE path='bad-stats'").fetchone() == (
            "bad-stats", 1, 1, "v", "features", "tags", "error",
        )


def test_export_omits_nulls_and_failures_and_sorts_paths(folders, monkeypatch):
    library, data, out = folders
    for name in ("z", "a"):
        _flac(library / f"{name}.flac", tags={"replaygain_track_gain": "-2"})
    _flac(library / "bad.flac")
    _ffmpeg(monkeypatch, {"bad.flac": "no summary"})
    loudness.measure(library, data, out)
    document = _exported(out)
    assert document["version"] == 1 and document["reference"] == -18 and document["generatedAt"].endswith("Z")
    assert document["tracks"] == [
        {"path": f"{name}.flac", "trackGain": -2, "source": "tag"} for name in ("a", "z")
    ]
    assert not list(out.glob(".xiyue-loud-*"))


def test_write_selection_and_counts(folders, monkeypatch):
    library, data, out = folders
    _flac(library / "measured.flac")
    _flac(library / "tagged.flac", tags={"replaygain_track_gain": "-3"})
    sf.write(library / "unsupported.wav", np.zeros(8000), 8000)
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    service = loudness.Loudness(library, data, out, library)
    snapshot = service.snapshot()
    assert snapshot["counts"] == {"fromTags": 1, "measured": 2, "written": 0, "failed": 0, "unsupported": 1}
    assert snapshot["pending"] == ["measured.flac"]
    result = loudness.write(library, data, out, library, ["tagged.flac", "unsupported.wav", "unknown.flac"])
    assert result == {"written": [], "failed": [], "skipped": [
        {"path": "tagged.flac", "reason": "not_measured"},
        {"path": "unsupported.wav", "reason": "unsupported"},
        {"path": "unknown.flac", "reason": "unknown"},
    ]}
    assert loudness.write(library, data, out, library)["written"] == ["measured.flac"]
    assert service.snapshot()["counts"]["written"] == 1
    assert service.snapshot()["pending"] == []


def test_measure_write_and_other_process_share_lock(folders):
    library, data, out = folders
    with loudness._acquire(data):
        for operation in (
            lambda: loudness.measure(library, data, out),
            lambda: loudness.write(library, data, out, library),
            lambda: loudness.Loudness(library, data, out, library).start("measuring"),
        ):
            with pytest.raises(loudness.LoudnessError, match="busy"):
                operation()
        script = (
            "import fcntl,sys\n"
            "with open(sys.argv[1], 'a+b') as lock:\n"
            " try:\n"
            "  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            " except BlockingIOError:\n"
            "  print('busy')\n"
        )
        result = subprocess.run([sys.executable, "-c", script, str(data / "loudness.lock")], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "busy"
    assert loudness.measure(library, data, out)["measured"] == 0


@contextmanager
def _server(folders, panel=True):
    library, data, out = folders
    data.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    accounts = AccountStore(data)
    accounts.create("测试家人", "loudness-family-test")
    family = accounts.login("测试家人", "loudness-family-test", "测试设备")["token"]
    access = AccessSettings(
        None, token="loudness-owner-test", network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2",
    )
    service = loudness.Loudness(library, data, out, library)
    server = make_server(
        out, 0, data=data, state=PanelState() if panel else None,
        index=OutputIndex(out) if panel else None, accounts=accounts, access=access, loudness=service,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1], family, service
    finally:
        server.shutdown()
        server.server_close()


def _request(port, method, path, body=None, token="loudness-owner-test", status=200):
    headers = {"Authorization": "Bearer " + token} if token else {}
    code, _, payload = _call(port, method, path, body, headers)
    result = json.loads(payload)
    assert code == status, (code, result)
    return result


def _idle(service):
    for _ in range(250):
        snapshot = service.snapshot()
        if snapshot["state"] == "idle":
            return snapshot
        time.sleep(0.01)
    raise AssertionError("loudness job did not finish")


def _finish_followups(service):
    while True:
        with service._followup_lock:
            timer = service._followup_timer
        if timer is None:
            return
        timer.join(3)
        assert not timer.is_alive(), "follow-up check did not finish"


def test_followup_finds_moved_tagged_file_without_ffmpeg(folders, monkeypatch, caplog):
    library, data, out = folders
    old = _flac(library / "downloads/a.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-2 dB"})
    calls = _ffmpeg(monkeypatch)
    service = loudness.Loudness(library, data, out, library)
    release = threading.Event()
    check = service._check_followup

    def gated_check(index, generation):
        assert release.wait(3)
        check(index, generation)

    monkeypatch.setattr(loudness, "_FOLLOWUP_SECONDS", (0.05, 0.1, 0.15))
    monkeypatch.setattr(service, "_check_followup", gated_check)
    caplog.set_level(logging.INFO, logger="xiyue")
    try:
        service.request_measure("downloads/a.flac")
        assert _idle(service)["lastMeasure"]["fromTags"] == 1
        assert [track["path"] for track in _exported(out)["tracks"]] == ["downloads/a.flac"]
        new = library / "最近在听/歌手/专辑/a.flac"
        new.parent.mkdir(parents=True)
        old.rename(new)
        release.set()
        _finish_followups(service)
        assert _idle(service)["lastMeasure"]["fromTags"] == 1
        tracks = _exported(out)["tracks"]
        assert [track["path"] for track in tracks] == ["最近在听/歌手/专辑/a.flac"]
        assert tracks[0]["source"] == "tag" and tracks[0]["trackGain"] == -2
        assert set(_rows(data)) == {"最近在听/歌手/专辑/a.flac"}
        assert calls == []
        messages = [record.getMessage() for record in caplog.records if "LOUD-FOLLOWUP" in record.message]
        assert messages == [
            "LOUD-FOLLOWUP run changed=1 files=1",
            "LOUD-FOLLOWUP skipped files=1",
            "LOUD-FOLLOWUP skipped files=1",
        ]
        assert service._followup_files == {}
    finally:
        release.set()
        _finish_followups(service)


def test_followup_skips_unchanged_files_and_clears_tracking(folders, monkeypatch, caplog):
    library, data, out = folders
    _flac(library / "downloads/a.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-2 dB"})
    calls = _ffmpeg(monkeypatch)
    service = loudness.Loudness(library, data, out, library)
    monkeypatch.setattr(loudness, "_FOLLOWUP_SECONDS", (0.05, 0.1, 0.15))
    caplog.set_level(logging.INFO, logger="xiyue")
    service.request_measure("downloads/a.flac")
    assert service._followup_timer.daemon is True
    _finish_followups(service)
    _idle(service)
    messages = [record.getMessage() for record in caplog.records]
    assert [message for message in messages if message.startswith("LOUD-FOLLOWUP")] == [
        "LOUD-FOLLOWUP skipped files=1",
    ] * 3
    assert sum(message.startswith("LOUD-START") for message in messages) == 1
    assert service._followup_files == {}
    assert service._followup_timer is None
    assert calls == []


def test_followup_new_download_restarts_one_cumulative_timer(folders, monkeypatch, caplog):
    library, data, out = folders
    _flac(library / "downloads/a.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-2 dB"})
    _flac(library / "downloads/b.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-3 dB"})
    service = loudness.Loudness(library, data, out, library)
    timers, clock = [], [100.0]

    class Timer:
        def __init__(self, seconds, function, args):
            self.seconds, self.function, self.args = seconds, function, args
            self.cancelled = self.fired = False

        def start(self):
            timers.append(self)

        def cancel(self):
            self.cancelled = True

        def fire(self):
            self.fired = True
            self.function(*self.args)

    monkeypatch.setattr(loudness, "_FOLLOWUP_SECONDS", (0.05, 0.1, 0.15))
    monkeypatch.setattr(loudness.threading, "Timer", Timer)
    monkeypatch.setattr(loudness.time, "monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="xiyue")
    service.request_measure("downloads/a.flac")
    _idle(service)
    first = service._followup_timer
    clock[0] = 100.02
    service.request_measure("downloads/b.flac")
    _idle(service)
    second = service._followup_timer
    assert first.cancelled and second is not first
    assert [timer for timer in timers if not timer.cancelled and not timer.fired] == [second]
    assert second.seconds == pytest.approx(0.05)
    assert set(service._followup_files) == {"downloads/a.flac", "downloads/b.flac"}
    # Even a callback that had already entered before cancellation cannot
    # clear the new round's files or schedule another timer.
    first.fire()
    assert service._followup_timer is second and len(timers) == 2
    for instant in (100.07, 100.12, 100.17):
        timer = service._followup_timer
        assert timer.daemon is True
        assert timer.seconds == pytest.approx(0.05)
        clock[0] = instant
        timer.fire()
    assert service._followup_timer is None
    assert service._followup_files == {}
    assert [timer for timer in timers if not timer.cancelled and not timer.fired] == []
    messages = [record.getMessage() for record in caplog.records if "LOUD-FOLLOWUP" in record.message]
    assert messages == ["LOUD-FOLLOWUP skipped files=2"] * 3


def test_followup_uses_existing_busy_retry_then_measures(folders, monkeypatch, caplog):
    library, data, out = folders
    old = _flac(library / "downloads/a.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-2 dB"})
    calls = _ffmpeg(monkeypatch)
    service = loudness.Loudness(library, data, out, library)
    entered, release = threading.Event(), threading.Event()
    retry = service._retry_measure

    def gated_retry(retries_left):
        entered.set()
        assert release.wait(3)
        retry(retries_left)

    monkeypatch.setattr(loudness, "_FOLLOWUP_SECONDS", (0.05, 0.1, 0.15))
    monkeypatch.setattr(loudness, "_MEASURE_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(service, "_retry_measure", gated_retry)
    caplog.set_level(logging.INFO, logger="xiyue")
    try:
        service.request_measure("downloads/a.flac")
        _idle(service)
        with loudness._acquire(data):
            old.rename(library / "moved.flac")
            assert entered.wait(3)
            timer = service._measure_retry
            assert timer is not None
            assert any(record.getMessage() == "LOUD-AFTER-DOWNLOAD busy_retry" for record in caplog.records)
        release.set()
        timer.join(3)
        assert not timer.is_alive()
        _finish_followups(service)
        assert _idle(service)["lastMeasure"]["fromTags"] == 1
        assert [track["path"] for track in _exported(out)["tracks"]] == ["moved.flac"]
        assert service._measure_retry is None
        assert calls == []
    finally:
        release.set()
        _finish_followups(service)


def test_request_measure_starts_incrementally_and_exports_new_track(folders, monkeypatch, caplog):
    library, data, out = folders
    _flac(library / "old.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-1 dB"})
    calls = _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    _flac(library / "new.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-2 dB"})
    service = loudness.Loudness(library, data, out, library)
    caplog.set_level(logging.INFO, logger="xiyue")
    service.request_measure()
    status = _idle(service)
    assert status["lastMeasure"]["fromTags"] == 1
    assert status["lastMeasure"]["measured"] == 0
    assert calls == []
    tracks = _exported(out)["tracks"]
    assert [track["path"] for track in tracks] == ["new.flac", "old.flac"]
    assert tracks[0]["trackGain"] == -2
    assert tracks[0]["source"] == "tag"
    messages = [record.getMessage() for record in caplog.records if "LOUD-AFTER-DOWNLOAD" in record.message]
    assert messages == ["LOUD-AFTER-DOWNLOAD started"]


def test_request_measure_coalesces_busy_requests(folders, monkeypatch, caplog):
    library, data, out = folders
    _flac(library / "new.flac", tags={"REPLAYGAIN_TRACK_GAIN": "-2 dB"})
    service = loudness.Loudness(library, data, out, library)
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    retry, start = service._retry_measure, service.start

    def gated_retry(retries_left):
        entered.set()
        assert release.wait(3)
        retry(retries_left)

    def observed_start(operation, **options):
        assert operation == "measuring" and options == {"remeasure": False}
        start(operation, **options)
        started.set()

    monkeypatch.setattr(loudness, "_MEASURE_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(service, "_retry_measure", gated_retry)
    monkeypatch.setattr(service, "start", observed_start)
    caplog.set_level(logging.INFO, logger="xiyue")
    try:
        with loudness._acquire(data):
            service.request_measure()
            timer = service._measure_retry
            service.request_measure()
            service.request_measure()
            assert service._measure_retry is timer
            assert entered.wait(3)
            assert not started.is_set()
        release.set()
        assert started.wait(3)
        timer.join(3)
        assert not timer.is_alive()
        assert service._measure_retry is None
        assert _idle(service)["lastMeasure"]["fromTags"] == 1
        assert [track["path"] for track in _exported(out)["tracks"]] == ["new.flac"]
        messages = [record.getMessage() for record in caplog.records if "LOUD-AFTER-DOWNLOAD" in record.message]
        assert messages == ["LOUD-AFTER-DOWNLOAD busy_retry", "LOUD-AFTER-DOWNLOAD started"]
    finally:
        release.set()


def test_request_measure_gives_up_after_twenty_retries(folders, monkeypatch, caplog):
    library, data, out = folders
    service = loudness.Loudness(library, data, out, library)
    pending = []

    class Timer:
        def __init__(self, seconds, function, args):
            assert seconds == 30
            self.function, self.args = function, args

        def start(self):
            pending.append(self)

    monkeypatch.setattr(loudness.threading, "Timer", Timer)
    caplog.set_level(logging.INFO, logger="xiyue")
    with loudness._acquire(data):
        service.request_measure()
        for _ in range(20):
            assert len(pending) == 1
            timer = pending.pop()
            assert timer.daemon is True
            timer.function(*timer.args)
        assert pending == []
        assert service._measure_retry is None
    messages = [record.getMessage() for record in caplog.records if "LOUD-AFTER-DOWNLOAD" in record.message]
    assert messages == ["LOUD-AFTER-DOWNLOAD busy_retry"] * 20 + ["LOUD-AFTER-DOWNLOAD gave_up"]


def test_http_owner_permissions_busy_bad_paths_and_counts(folders, monkeypatch):
    library, data, out = folders
    _flac(library / "a.flac")
    _ffmpeg(monkeypatch)
    loudness.measure(library, data, out)
    with _server(folders) as (port, family, service):
        for method, path in (
            ("GET", "/api/loudness"), ("POST", "/api/loudness/scan"), ("POST", "/api/loudness/write"),
        ):
            assert _request(port, method, path, token=family, status=403) == {"error": "owner_only"}
            assert _request(port, method, path, token=None, status=401) == {"error": "unauthorized"}
        for paths in ("a.flac", {}, [1], None):
            assert _request(port, "POST", "/api/loudness/write", {"paths": paths}, status=400) == {"error": "bad_request"}
        assert _request(port, "POST", "/api/loudness/scan", {"remeasure": "yes"}, status=400) == {"error": "bad_request"}
        with loudness._acquire(data):
            for path in ("/api/loudness/scan", "/api/loudness/write"):
                assert _request(port, "POST", path, status=409) == {"error": "busy"}
        status = _request(port, "GET", "/api/loudness")
        assert status["pending"] == ["a.flac"] and status["counts"]["measured"] == 1
        assert status["lastMeasure"]["measured"] == 1
        code, headers, body = _call(port, "GET", "/xiyue-loudness-v1.json.gz", headers={"Authorization": "Bearer " + family})
        assert code == 200 and headers["Content-Encoding"] == "gzip"
        assert json.loads(gzip.decompress(body)) == _exported(out)


def test_http_background_progress_and_history_survive_restart(folders, monkeypatch):
    library, data, out = folders
    _flac(library / "a.flac")
    entered, release = threading.Event(), threading.Event()
    original = loudness.measure_audio
    _ffmpeg(monkeypatch)

    def measured(path):
        entered.set()
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(loudness, "measure_audio", measured)
    with _server(folders) as (port, _, service):
        try:
            assert _request(port, "POST", "/api/loudness/scan", status=202) == {"ok": True}
            assert entered.wait(3)
            snapshot = _request(port, "GET", "/api/loudness")
            assert (snapshot["state"], snapshot["done"], snapshot["total"]) == ("measuring", 0, 1)
            assert _request(port, "POST", "/api/loudness/write", status=409) == {"error": "busy"}
        finally:
            release.set()
            _idle(service)
        assert _request(port, "POST", "/api/loudness/write", status=202) == {"ok": True}
        final = _idle(service)
        assert (final["done"], final["total"], final["lastWrite"]["written"]) == (1, 1, 1)
        assert final["lastMeasure"]["measured"] == 1
        assert final["pending"] == [] and final["counts"]["written"] == 1
        restarted = loudness.Loudness(library, data, out, library).snapshot()
        assert restarted == final


def test_http_unconfigured_panel_returns_503(folders):
    with _server(folders, panel=False) as (port, _, service):
        for method, path in (
            ("GET", "/api/loudness"), ("POST", "/api/loudness/scan"), ("POST", "/api/loudness/write"),
        ):
            assert _request(port, method, path, status=503) == {"error": "loudness_unconfigured"}


@pytest.mark.parametrize("failure", [False, True])
def test_download_tags_part_or_finishes_when_measurement_fails(tmp_path, monkeypatch, caplog, failure):
    seen = []
    real_write = loudness.write_tags

    def measure(path):
        assert path.name.startswith(".xiyue-part-")
        seen.append(path)
        if failure:
            raise loudness.LoudnessError("no_summary")
        return {"track_gain": 2.5, "track_peak": 0.9}

    def write(path, format, gains, track_only=False):
        assert path == seen[0] and format == "flac" and track_only
        assert not list(tmp_path.glob("歌.flac"))
        real_write(path, format, gains, track_only=track_only)

    monkeypatch.setattr(loudness, "measure_audio", measure)
    monkeypatch.setattr(loudness, "write_tags", write)
    caplog.set_level(logging.WARNING, logger="xiyue")
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        job = _wait_for(downloads, downloads.submit({"url": base + "/a?secret=private", "filename": "歌.flac"}))
        assert job["state"] == "done" and len(seen) == 1
        result = tmp_path / job["path"]
        if failure:
            assert result.read_bytes() == _AUDIO
            assert "LOUD-DOWNLOAD-FAIL 歌.flac no_summary" in caplog.text
            assert "secret=private" not in caplog.text
        else:
            assert loudness.read_gains(result, "flac") == {
                "track_gain": 2.5, "track_peak": 0.9, "album_gain": None, "album_peak": None,
            }
        assert not list(tmp_path.glob(".xiyue-part-*"))
    finally:
        server.shutdown()
        server.server_close()


def test_cli_measure_write_paths_and_write_all(folders, monkeypatch, capsys):
    library, data, out = folders
    for name in ("a", "b"):
        _flac(library / f"{name}.flac")
    _ffmpeg(monkeypatch)
    base = ["analyzer", "loudness", "--library", str(library), "--data", str(data), "--out", str(out)]
    for options, expected in (
        ([], {"measured": 2}),
        (["--write", "a.flac"], {"written": ["a.flac"]}),
        (["--write-all"], {"written": ["b.flac"]}),
    ):
        monkeypatch.setattr(sys, "argv", base + options)
        command.main()
        result = json.loads(capsys.readouterr().out)
        for key, value in expected.items():
            assert result[key] == value


def test_run_cycle_handles_busy_missing_library_and_unexpected_failure(folders, monkeypatch, caplog):
    library, data, out = folders
    args = SimpleNamespace(downloads=library, data=data, out=out)
    caplog.set_level(logging.WARNING, logger="xiyue")
    with loudness._acquire(data):
        command._measure_loudness(args)
    assert "LOUD-SKIP busy" in caplog.text
    args.downloads = library / "missing"
    command._measure_loudness(args)
    assert "LOUD-SKIP no_library" in caplog.text

    def failure(*args, **kwargs):
        raise RuntimeError("not for logging: private URL")

    monkeypatch.setattr(loudness, "measure", failure)
    command._measure_loudness(args)
    assert "LOUD-FAIL measuring RuntimeError" in caplog.text and "private URL" not in caplog.text


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("format,codec", [("mp3", "libmp3lame"), ("m4a", "aac")])
def test_real_mp3_and_m4a_safe_write(folders, format, codec):
    library, data, out = folders
    source = _flac(library / "input.flac", album="专辑")
    path = library / ("song." + format)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", codec, str(path)],
        check=True, capture_output=True, timeout=30,
    )
    source.unlink()
    old = path.stat()
    result = loudness.measure(library, data, out)
    assert result["measured"] == 1 and result["failed"] == 0
    assert loudness.write(library, data, out, library)["written"] == [path.name]
    assert path.stat().st_mtime_ns == old.st_mtime_ns
    assert loudness.read_gains(path, format)["track_gain"] == _rows(data)[path.name]["track_gain"]
    assert not list(library.glob(".xiyue-rg-*"))
