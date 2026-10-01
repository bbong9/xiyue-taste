import gzip
import json
import os
import sqlite3
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import mutagen.flac
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import ANALYZER_VERSION
from analyzer.features import extract_features
from analyzer.output import write_output
from analyzer.scan import scan
from analyzer.serve import make_server
from analyzer.similarity import build_similarity
from analyzer.tags import lyrics_language, read_tags


SAMPLE_RATE = 22050


def write_sine(path, frequency=440, seconds=6):
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    audio = 0.3 * np.sin(2 * np.pi * frequency * time)
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    return audio


def test_click_track_detects_120_bpm(tmp_path):
    audio = np.zeros(SAMPLE_RATE * 16)
    click = 0.7 * np.hanning(256)
    for offset in range(SAMPLE_RATE // 4, len(audio) - len(click), SAMPLE_RATE // 2):
        audio[offset:offset + len(click)] = click
    path = tmp_path / "clicks.wav"
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    features = extract_features(path)
    assert 117 <= features["bpm"] <= 123 or 58.5 <= features["bpm"] <= 61.5
    assert len(features["vector"]) == 53


def test_cache_skips_unchanged_and_reanalyzes_mtime_change(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    path = music / "song.WAV"
    write_sine(path)
    hidden = music / ".hidden"
    hidden.mkdir()
    write_sine(hidden / "ignored.wav")
    data, out = tmp_path / "data", tmp_path / "out"
    assert scan(music, data, out, workers=2) == {"analyzed": 1, "tracks": 1}
    assert scan(music, data, out, workers=2) == {"analyzed": 0, "tracks": 1}
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 2))
    assert scan(music, data, out, workers=2) == {"analyzed": 1, "tracks": 1}
    with sqlite3.connect(data / "cache.sqlite") as connection:
        row = connection.execute(
            "SELECT path, size, mtime, version, error FROM tracks"
        ).fetchone()
    assert row == ("song.WAV", path.stat().st_size, path.stat().st_mtime, ANALYZER_VERSION, None)
    path.unlink()
    assert scan(music, data, out, workers=2) == {"analyzed": 0, "tracks": 0}


def test_output_schema_vectors_and_neighbors(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    for index, frequency in enumerate((220, 440, 880)):
        write_sine(music / f"song-{index}.wav", frequency)
    out = tmp_path / "out"
    assert scan(music, tmp_path / "data", out, workers=2)["tracks"] == 3
    with gzip.open(out / "xiyue-taste-v1.json.gz", "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    assert set(document) == {"schema", "analyzerVersion", "generatedAt", "tracks"}
    assert document["schema"] == 1
    assert document["analyzerVersion"] == ANALYZER_VERSION
    assert datetime.fromisoformat(document["generatedAt"]).utcoffset().total_seconds() == 0
    fields = {
        "path", "size", "mtime", "title", "artists", "album", "genre", "year",
        "durationSec", "bpm", "loudnessLUFS", "lyricsLanguage", "vector", "neighbors",
    }
    for index, track in enumerate(document["tracks"]):
        assert set(track) == fields
        assert len(track["vector"]) == 53
        assert all(value == round(value, 4) for value in track["vector"])
        assert len(track["neighbors"]) == min(20, len(document["tracks"]) - 1)
        assert all(neighbor[0] != index for neighbor in track["neighbors"])
        assert all(score == round(score, 4) for _, score in track["neighbors"])
    tracks = [
        {
            "path": f"song-{index}.wav", "size": 1, "mtime": 1,
            "features": {"vector": [float(index)] * 53, "durationSec": 6,
                         "bpm": 120, "loudnessLUFS": -15},
            "tags": {"title": f"Song {index}", "artists": None, "album": None,
                     "genre": None, "year": None, "lyricsLanguage": None},
        }
        for index in range(22)
    ]
    vectors, neighbors = build_similarity(tracks)
    write_output(out, tracks, vectors, neighbors)
    with gzip.open(out / "xiyue-taste-v1.json.gz", "rt", encoding="utf-8") as handle:
        larger = json.load(handle)
    assert all(len(track["neighbors"]) == 20 for track in larger["tracks"])
    assert all(
        neighbor[0] != index
        for index, track in enumerate(larger["tracks"])
        for neighbor in track["neighbors"]
    )


def test_lyrics_language_table():
    cases = [
        ("[00:01.20]\u4f60\u597d\u4e16\u754c", "zh"),
        ("[01:02.34]\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", "ja"),
        ("[00:03.45]\uc548\ub155\ud558\uc138\uc694", "ko"),
        ("[00:04.56]Hello wonderful world", "en"),
        ("[00:05.67]123456789", "unknown"),
    ]
    for lyrics, expected in cases:
        assert lyrics_language(lyrics) == expected


def test_gbk_lrc_does_not_fail_track(tmp_path):
    path = tmp_path / "song.wav"
    write_sine(path)
    path.with_suffix(".lrc").write_bytes(
        "[00:01.00]\u4f60\u597d\u4e16\u754c".encode("gbk")
    )
    tags = read_tags(path)
    assert "title" in tags


def test_similar_sine_tracks_are_mutual_nearest_neighbors(tmp_path):
    first = tmp_path / "a.wav"
    audio = write_sine(first, seconds=8)
    rng = np.random.default_rng(42)
    second, noise = tmp_path / "a-prime.wav", tmp_path / "noise.wav"
    sf.write(second, audio + rng.normal(0, 0.0001, len(audio)), SAMPLE_RATE, subtype="FLOAT")
    sf.write(noise, rng.normal(0, 0.15, len(audio)), SAMPLE_RATE, subtype="FLOAT")
    tracks = [{"features": extract_features(path)} for path in (first, second, noise)]
    _, neighbors = build_similarity(tracks)
    assert [index for index, _ in neighbors[0]] == [1, 2]
    assert [index for index, _ in neighbors[1]] == [0, 2]
    assert neighbors[0][0][1] > neighbors[0][1][1]
    assert neighbors[1][0][1] > neighbors[1][1][1]


def test_flac_without_genre_or_year_reads_tags(tmp_path):
    path = tmp_path / "song.flac"
    sf.write(path, np.zeros(SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")
    flac = mutagen.flac.FLAC(path)
    flac["title"] = "Only Title"
    flac["artist"] = "Someone"
    flac.save()
    tags = read_tags(path)
    assert tags["title"] == "Only Title"
    assert tags["artists"] == ["Someone"]
    assert tags["genre"] is None
    assert tags["year"] is None


def test_m4a_decodes_through_ffmpeg(tmp_path):
    wav, m4a = tmp_path / "song.wav", tmp_path / "song.m4a"
    write_sine(wav, seconds=6)
    subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-c:a", "aac", m4a], check=True)
    features = extract_features(m4a)
    assert len(features["vector"]) == 53
    assert 5.5 <= features["durationSec"] <= 6.5


def test_serve_sends_gzip_json_and_honours_if_modified_since(tmp_path):
    payload = b'{"schema":1,"tracks":[]}'
    (tmp_path / "xiyue-taste-v1.json.gz").write_bytes(gzip.compress(payload))
    server = make_server(tmp_path, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/xiyue-taste-v1.json.gz"
        with urllib.request.urlopen(url) as response:
            assert response.headers["Content-Encoding"] == "gzip"
            assert response.headers["Content-Type"] == "application/json"
            assert gzip.decompress(response.read()) == payload
            modified = response.headers["Last-Modified"]
        with pytest.raises(urllib.error.HTTPError) as unchanged:
            urllib.request.urlopen(urllib.request.Request(url, headers={"If-Modified-Since": modified}))
        assert unchanged.value.code == 304
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(url.replace("v1", "v0"))
        assert missing.value.code == 404
        assert missing.value.headers["Content-Encoding"] is None
    finally:
        server.shutdown()
        server.server_close()


from analyzer.panel import OutputIndex, PanelState


def _write_panel_output(out, count=3):
    tracks = [
        {
            "path": f"song-{index}.wav", "size": 1, "mtime": 1,
            "features": {
                "vector": [float(index)] * 53, "durationSec": 6,
                "bpm": (50, 120, 190)[index % 3],
                "loudnessLUFS": (-30, -15, 0)[index % 3],
            },
            "tags": {
                "title": f"Moonlight {index}", "artists": ["Artist"],
                "album": "Album", "genre": "Ambient", "year": "2026",
                "lyricsLanguage": ("zh", None, "en")[index % 3],
            },
        }
        for index in range(count)
    ]
    vectors, neighbors = build_similarity(tracks)
    write_output(out, tracks, vectors, neighbors)
    return tracks


def test_panel_state_progress_finish_and_failure():
    state = PanelState()
    state.begin_scan()
    state.set_progress(3, 10)
    snapshot = state.snapshot()
    assert snapshot["state"] == "scanning"
    assert (snapshot["done"], snapshot["total"]) == (3, 10)
    snapshot["done"] = 99
    assert state.snapshot()["done"] == 3
    state.finish_scan(5, 120)
    snapshot = state.snapshot()
    assert snapshot["state"] == "idle"
    assert snapshot["lastAnalyzed"] == 5
    assert snapshot["lastExported"] == 120
    state.fail_scan(ValueError())
    assert state.snapshot()["error"] == "ValueError"
    state.request_scan()
    state.wait_for_next(0)
    assert not state.scanRequested.is_set()


def test_output_index_stats_histograms_and_languages(tmp_path):
    index = OutputIndex(tmp_path)
    assert index.stats()["trackCount"] == 0
    assert index.stats()["bpm"] == {"min": None, "max": None, "bins": []}
    tracks = _write_panel_output(tmp_path)
    stats = index.stats()
    assert stats["trackCount"] == 3
    assert len(stats["bpm"]["bins"]) == 12
    assert sum(bucket["count"] for bucket in stats["bpm"]["bins"]) == 3
    assert stats["bpm"]["bins"][0]["count"] == 1
    assert stats["bpm"]["bins"][-1]["count"] == 1
    assert len(stats["loudness"]["bins"]) == 10
    assert sum(bucket["count"] for bucket in stats["loudness"]["bins"]) == 3
    assert sum(item["count"] for item in stats["languages"]) == sum(
        bool(track["tags"]["lyricsLanguage"]) for track in tracks
    )
    assert all("vector" not in track for track in index.load())


def test_output_index_search_limit_and_neighbor_details(tmp_path):
    _write_panel_output(tmp_path)
    index = OutputIndex(tmp_path)
    matches = index.tracks("MOONLIGHT 1", 0, 50)
    assert matches["total"] == 1
    assert matches["items"][0]["index"] == 1
    neighbor = index.track(0)["neighbors"][0]
    assert "title" in neighbor and "score" in neighbor
    assert index.track(99) is None
    _write_panel_output(tmp_path, count=105)
    assert len(index.tracks("", 0, 1000)["items"]) == 100


def test_panel_http_routes_and_scan_request(tmp_path):
    _write_panel_output(tmp_path)
    state = PanelState()
    server = make_server(
        tmp_path, 0, data=tmp_path / "data", state=state, index=OutputIndex(tmp_path)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/html")
            assert response.headers["Cache-Control"] == "no-cache"
        with urllib.request.urlopen(base + "/api/status") as response:
            assert "state" in json.load(response)
            assert response.headers["Content-Type"] == "application/json; charset=utf-8"
            assert response.headers["Cache-Control"] == "no-store"
        request = urllib.request.Request(base + "/api/scan", data=b"", method="POST")
        with urllib.request.urlopen(request) as response:
            assert json.load(response) == {"ok": True}
        assert state.scanRequested.is_set()
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(base + "/api/track?i=99")
        assert missing.value.code == 404
        assert json.load(missing.value) == {"error": "not_found"}
        with urllib.request.urlopen(base + "/xiyue-taste-v1.json.gz") as response:
            assert response.headers["Content-Encoding"] == "gzip"
    finally:
        server.shutdown()
        server.server_close()


def test_scan_reports_initial_and_completed_progress(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    for index in range(2):
        write_sine(music / f"song-{index}.wav")
    updates = []
    result = scan(
        music, tmp_path / "data", tmp_path / "out", workers=2,
        progress=lambda done, total: updates.append((done, total)),
    )
    assert result == {"analyzed": 2, "tracks": 2}
    assert updates[0] == (0, 2)
    assert updates[-1] == (2, 2)
    assert updates == [(0, 2), (1, 2), (2, 2)]
