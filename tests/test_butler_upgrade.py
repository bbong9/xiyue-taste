"""Tests for NAS-side background ButlerUpgrader & /api/butler/upgrade/* endpoints."""

import contextlib
import http.server
import ipaddress
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from mutagen.flac import FLAC
from mutagen.id3 import APIC, TALB, TIT2, TPE1, USLT
from mutagen.mp3 import MP3

from analyzer.access import AccessSettings
from analyzer.accounts import AccountStore
from analyzer.butler_files import ButlerFileOps
from analyzer.butler_upgrade import ButlerUpgrader
from analyzer.panel import OutputIndex, PanelState
from analyzer.personal import PersonalStore
from analyzer.serve import make_server

OWNER_TOKEN = "test-owner-upgrade-token"
PASSWORD = "Family-upgrade-password"
SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _make_audio(path: Path, fmt: str, *, freq=440, duration=6, bitrate="128k"):
    """Synthesizes a deterministic musical tone (sum of harmonics + envelope) using ffmpeg."""
    expr = (
        f"(0.4+0.6*abs(sin(2*PI*0.35*t)))"
        f"*(0.5*sin(2*PI*{freq}*t)+0.25*sin(2*PI*2200*t)+0.15*sin(2*PI*8000*t)+0.1*sin(2*PI*19500*t))"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"aevalsrc={expr}:s=44100:d={duration}",
        "-ac", "2",
    ]
    if fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", bitrate]
    else:
        cmd += ["-c:a", "flac", "-sample_fmt", "s16"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)


def _http(port, method, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req_headers = {"Content-Type": "application/json"} if data is not None else {}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


@contextlib.contextmanager
def _file_server(directory: Path):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def _api_server(tmp_path, files, upgrader, accounts=None, personal=None):
    access = AccessSettings(
        None, token=OWNER_TOKEN, network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2",
    )
    server = make_server(
        tmp_path, 0, data=tmp_path, state=PanelState(), index=OutputIndex(tmp_path),
        access=access, accounts=accounts, personal=personal,
        butler_files=files, butler_upgrade=upgrader,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _wait_idle(port, headers, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, snap = _http(port, "GET", "/api/butler/upgrade/status", headers=headers)
        assert status == 200
        if snap["state"] == "idle":
            return snap
        time.sleep(0.1)
    raise AssertionError("upgrade batch did not finish in time")


def test_butler_upgrade_replaces_mp3_with_flac_preserving_tags_and_rejecting_different_song(tmp_path):
    library = tmp_path / "downloads"
    album = library / "周杰伦" / "叶惠美"
    album.mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    # Original low-bitrate MP3 with embedded lyrics and cover art.
    old_mp3 = album / "01 晴天.mp3"
    _make_audio(old_mp3, "mp3", freq=440, duration=6, bitrate="128k")
    mp3_audio = MP3(old_mp3)
    if mp3_audio.tags is None:
        mp3_audio.add_tags()
    mp3_audio.tags.add(TIT2(encoding=3, text=["晴天"]))
    mp3_audio.tags.add(TPE1(encoding=3, text=["周杰伦"]))
    mp3_audio.tags.add(TALB(encoding=3, text=["叶惠美"]))
    mp3_audio.tags.add(USLT(encoding=3, lang="chi", desc="", text="[00:01.00]故事的小黄花"))
    mp3_audio.tags.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="", data=SAMPLE_JPEG))
    mp3_audio.save()

    # Matching FLAC (same recording, high frequencies up to 19.5 kHz).
    good_flac = upstream / "good.flac"
    _make_audio(good_flac, "flac", freq=440, duration=6)

    # A second MP3 whose upstream candidate is a totally different recording (different envelope rhythm).
    second_mp3 = album / "02 稻香.mp3"
    _make_audio(second_mp3, "mp3", freq=440, duration=6, bitrate="128k")
    different_flac = upstream / "different.flac"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "aevalsrc=(0.2+0.8*abs(sin(2*PI*1.9*t)))*sin(2*PI*660*t):s=44100:d=6",
        "-ac", "2", "-c:a", "flac", "-sample_fmt", "s16", str(different_flac),
    ], check=True)

    # Already-FLAC track should be skipped cleanly with settled=True.
    already_flac = album / "03 七里香.flac"
    _make_audio(already_flac, "flac", freq=440, duration=6)

    done_callbacks = []
    files = ButlerFileOps(library)
    upgrader = ButlerUpgrader(
        files, tmp_path / "data", on_done=done_callbacks.append, allow_private=True,
    )
    owner_headers = {"Authorization": f"Bearer {OWNER_TOKEN}"}

    with _file_server(upstream) as media_port, _api_server(tmp_path, files, upgrader) as api_port:
        status, started = _http(api_port, "POST", "/api/butler/upgrade/start", {
            "target": "lossless",
            "skipsSoundChecks": False,
            "keepOldInTrash": False,
            "items": [
                {
                    "trackId": "trk-1",
                    "title": "晴天",
                    "artists": ["周杰伦"],
                    "album": "叶惠美",
                    "trackNumber": 3,
                    "year": 2003,
                    "nasPath": "/vol1/1000/Music/周杰伦/叶惠美/01 晴天.mp3",
                    "candidates": [{"sourceName": "本地测试源", "url": f"http://127.0.0.1:{media_port}/good.flac"}],
                },
                {
                    "trackId": "trk-2",
                    "title": "稻香",
                    "artists": ["周杰伦"],
                    "nasPath": "/vol1/1000/Music/周杰伦/叶惠美/02 稻香.mp3",
                    "candidates": [{"sourceName": "本地测试源", "url": f"http://127.0.0.1:{media_port}/different.flac"}],
                },
                {
                    "trackId": "trk-3",
                    "title": "七里香",
                    "artists": ["周杰伦"],
                    "nasPath": "/vol1/1000/Music/周杰伦/叶惠美/03 七里香.flac",
                    "candidates": [{"sourceName": "本地测试源", "url": f"http://127.0.0.1:{media_port}/good.flac"}],
                },
            ],
        }, headers=owner_headers)
        assert status == 202
        assert started["accepted"] == 3

        snap = _wait_idle(api_port, owner_headers)
        assert snap["done"] == 3
        assert snap["succeeded"] == 1
        assert snap["skipped"] == 2
        assert snap["failed"] == 0

        by_id = {entry["trackId"]: entry for entry in snap["items"]}

        # Track 1: replaced on disk, old MP3 deleted, FLAC carries preserved tags/lyrics/cover + ReplayGain!
        t1 = by_id["trk-1"]
        assert t1["status"] == "replaced"
        assert t1["settled"] is True
        assert t1["oldPath"] == "周杰伦/叶惠美/01 晴天.mp3"
        assert t1["newPath"] == "周杰伦/叶惠美/01 晴天.flac"
        assert "MP3" in t1["oldSummary"]
        assert "FLAC" in t1["newSummary"]
        assert not old_mp3.exists()
        new_flac = album / "01 晴天.flac"
        assert new_flac.is_file()
        flac_tags = FLAC(new_flac)
        assert flac_tags["TITLE"] == ["晴天"]
        assert flac_tags["ARTIST"] == ["周杰伦"]
        assert flac_tags["ALBUM"] == ["叶惠美"]
        assert flac_tags["TRACKNUMBER"] == ["3"]
        assert flac_tags["DATE"] == ["2003"]
        assert flac_tags["LYRICS"] == ["[00:01.00]故事的小黄花"]
        assert len(flac_tags.pictures) == 1
        assert bytes(flac_tags.pictures[0].data) == SAMPLE_JPEG
        assert done_callbacks == ["周杰伦/叶惠美/01 晴天.flac"]

        # Track 2: rejected as different recording; original MP3 stays untouched!
        t2 = by_id["trk-2"]
        assert t2["status"] == "skipped"
        assert "不是" in t2["reason"] or "不像同一个录音" in t2["reason"]
        assert second_mp3.is_file()
        assert not (album / "02 稻香.flac").exists()

        # Track 3: already lossless -> skipped cleanly.
        t3 = by_id["trk-3"]
        assert t3["status"] == "skipped"
        assert t3["settled"] is True

    # Reload from disk: persisted latest batch state survives container restart.
    reloaded = ButlerUpgrader(files, tmp_path / "data", allow_private=True)
    persisted = reloaded.snapshot()
    assert persisted["succeeded"] == 1
    assert len(persisted["items"]) == 3


def test_butler_upgrade_http_owner_only(tmp_path):
    library = tmp_path / "downloads"
    library.mkdir()
    accounts = AccountStore(tmp_path)
    personal = PersonalStore(tmp_path)
    accounts.create("家人", PASSWORD)
    family_token = accounts.login("家人", PASSWORD, "iPhone")["token"]
    family_headers = {"Authorization": f"Bearer {family_token}"}

    files = ButlerFileOps(library)
    upgrader = ButlerUpgrader(files, tmp_path / "data", allow_private=True)
    with _api_server(tmp_path, files, upgrader, accounts=accounts, personal=personal) as port:
        status, body = _http(port, "GET", "/api/butler/upgrade/status", headers=family_headers)
        assert status == 403 and body["error"] == "owner_only"
        status, _ = _http(port, "POST", "/api/butler/upgrade/start", {"target": "lossless", "items": []}, headers=family_headers)
        assert status == 403


def test_butler_upgrade_per_item_and_candidate_fallback_targets(tmp_path):
    library = tmp_path / "downloads"
    album = library / "刀郎" / "精选"
    album.mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    mp3_fallback = album / "01 手心里的温柔.mp3"
    mp3_direct_lower = album / "02 爱上一个不回家的人.mp3"
    already_flac = album / "03 冲动的惩罚.flac"
    _make_audio(mp3_fallback, "mp3", freq=440, duration=6, bitrate="128k")
    _make_audio(mp3_direct_lower, "mp3", freq=440, duration=6, bitrate="128k")
    _make_audio(already_flac, "flac", freq=440, duration=6)

    good_flac = upstream / "good.flac"
    _make_audio(good_flac, "flac", freq=440, duration=6)

    files = ButlerFileOps(library)
    upgrader = ButlerUpgrader(files, tmp_path / "data", allow_private=True)
    owner_headers = {"Authorization": f"Bearer {OWNER_TOKEN}"}

    with _file_server(upstream) as media_port, _api_server(tmp_path, files, upgrader) as api_port:
        status, started = _http(api_port, "POST", "/api/butler/upgrade/start", {
            "target": "master",
            "skipsSoundChecks": False,
            "keepOldInTrash": False,
            "items": [
                {
                    # Batch target is master: master candidate fails (404), falls back cleanly to lossless!
                    "trackId": "trk-fb",
                    "title": "手心里的温柔",
                    "artists": ["刀郎"],
                    "nasPath": "/vol1/1000/Music/刀郎/精选/01 手心里的温柔.mp3",
                    "target": "master",
                    "candidates": [
                        {"sourceName": "源A", "target": "master", "url": f"http://127.0.0.1:{media_port}/missing-master.flac"},
                        {"sourceName": "源A", "target": "lossless", "url": f"http://127.0.0.1:{media_port}/good.flac"},
                    ],
                },
                {
                    # Explicit per-item effectiveTarget = lossless inside a master batch.
                    "trackId": "trk-lower",
                    "title": "爱上一个不回家的人",
                    "artists": ["花花兔"],
                    "nasPath": "/vol1/1000/Music/刀郎/精选/02 爱上一个不回家的人.mp3",
                    "target": "lossless",
                    "candidates": [
                        {"sourceName": "源B", "target": "lossless", "url": f"http://127.0.0.1:{media_port}/good.flac"},
                    ],
                },
                {
                    # Already 16bit FLAC: master fails (404) and lossless fallback must NOT replace 16bit FLAC with 16bit FLAC!
                    "trackId": "trk-already-flac",
                    "title": "冲动的惩罚",
                    "artists": ["刀郎"],
                    "nasPath": "/vol1/1000/Music/刀郎/精选/03 冲动的惩罚.flac",
                    "target": "master",
                    "candidates": [
                        {"sourceName": "源A", "target": "master", "url": f"http://127.0.0.1:{media_port}/missing-master.flac"},
                        {"sourceName": "源A", "target": "lossless", "url": f"http://127.0.0.1:{media_port}/good.flac"},
                    ],
                },
            ],
        }, headers=owner_headers)
        assert status == 202 and started["accepted"] == 3

        snap = _wait_idle(api_port, owner_headers)
        assert snap["succeeded"] == 2
        assert snap["failed"] == 1
        by_id = {entry["trackId"]: entry for entry in snap["items"]}

        assert by_id["trk-fb"]["status"] == "replaced"
        assert by_id["trk-fb"]["achievedTarget"] == "lossless"
        assert not mp3_fallback.exists() and (album / "01 手心里的温柔.flac").is_file()

        assert by_id["trk-lower"]["status"] == "replaced"
        assert by_id["trk-lower"]["achievedTarget"] == "lossless"
        assert not mp3_direct_lower.exists() and (album / "02 爱上一个不回家的人.flac").is_file()

        assert by_id["trk-already-flac"]["status"] == "failed"
        assert by_id["trk-already-flac"]["settled"] is False
        assert already_flac.is_file()
