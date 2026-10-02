import base64
import gzip
import http.client
import http.server
import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
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
from analyzer.ask import Asker, AskError
from analyzer.features import extract_features
from analyzer.output import write_output
from analyzer.scan import scan
from analyzer.serve import _Handler, make_server
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


class _AskIndex:
    def load(self):
        return [
            {"title": "First", "artists": ["A", "B"], "path": "first.wav",
             "genre": "Pop", "lyricsLanguage": "en", "bpm": 120.6, "loudnessLUFS": -15.26},
            {"title": "Second", "artists": None, "path": "second.wav",
             "genre": None, "lyricsLanguage": None, "bpm": None, "loudnessLUFS": None},
            {"title": "第三首", "artists": ["歌手"], "path": "third.wav",
             "genre": "Ambient", "lyricsLanguage": "zh", "bpm": 60, "loudnessLUFS": -25},
        ]


def _fake_llm(content):
    calls = []

    def urlopen(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps({
            "choices": [{"message": {"content": content}}],
        }).encode("utf-8"))

    return urlopen, calls


def test_ask_without_key_does_not_call_model():
    urlopen, calls = _fake_llm('{"picks":[]}')
    asker = Asker(_AskIndex(), "", "https://example.com/v1", "test-model", urlopen=urlopen)
    with pytest.raises(AskError) as failure:
        asker.ask("安静")
    assert (failure.value.code, failure.value.status) == ("ask_unconfigured", 503)
    assert calls == []


def test_ask_selects_valid_unique_indices_and_sends_catalog():
    urlopen, calls = _fake_llm('```json\n{"reason":"安静","picks":[2,0,2,9,"x"]}\n```')
    asker = Asker(
        _AskIndex(), "test-key", "https://example.com/v1/", "test-model",
        timeout=42, urlopen=urlopen,
    )
    result = asker.ask("下雨天安静一点的")
    assert result == {
        "reason": "安静",
        "songs": [],
        "playlists": [],
        "items": [
            {"index": 2, "title": "第三首", "artists": ["歌手"], "path": "third.wav"},
            {"index": 0, "title": "First", "artists": ["A", "B"], "path": "first.wav"},
        ],
    }
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "https://example.com/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 42
    body = json.loads(request.data)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 1200
    content = body["messages"][1]["content"]
    assert "下雨天安静一点的" in content
    assert "2|第三首|歌手|Ambient|zh|60|-25.0" in content
    assert "0|First|A/B|Pop|en|121|-15.3" in content
    assert "1|Second|||||" in content


@pytest.mark.parametrize("error", [
    urllib.error.URLError("offline"),
    urllib.error.HTTPError("https://example.com/v1", 500, "failed", {}, None),
    TimeoutError(),
])
def test_ask_network_errors_are_generic(error):
    def urlopen(request, timeout):
        raise error

    asker = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen)
    with pytest.raises(AskError) as failure:
        asker.ask("安静")
    assert (failure.value.code, failure.value.status) == ("ask_failed", 502)


@pytest.mark.parametrize("content", ["not JSON", '{"picks":{}}', '{"reason":"安静"}', None])
def test_ask_invalid_model_content_fails(content):
    urlopen, _ = _fake_llm(content)
    asker = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen)
    with pytest.raises(AskError) as failure:
        asker.ask("安静")
    assert (failure.value.code, failure.value.status) == ("ask_failed", 502)


@pytest.mark.parametrize(("reason", "picks", "expected_reason", "expected_indices"), [
    (None, [], "", []),
    ("静" * 61, [True, False, -1, 1.0, 2], "静" * 60, [2]),
])
def test_ask_empty_results_reason_and_integer_filter(reason, picks, expected_reason, expected_indices):
    urlopen, _ = _fake_llm(json.dumps({"reason": reason, "picks": picks}))
    asker = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen)
    result = asker.ask("安静")
    assert result["reason"] == expected_reason
    assert [item["index"] for item in result["items"]] == expected_indices


@pytest.mark.parametrize(("body", "configured", "ask_error", "status", "expected"), [
    ({"q": ""}, True, None, 400, {"error": "bad_query"}),
    ({"q": " "}, True, None, 400, {"error": "bad_query"}),
    ({"q": "静" * 101}, True, None, 400, {"error": "bad_query"}),
    ({"q": 1}, True, None, 400, {"error": "bad_query"}),
    ([], True, None, 400, {"error": "bad_query"}),
    ({"q": "安静"}, False, None, 503, {"error": "ask_unconfigured"}),
    ({"q": "安静"}, True, AskError("ask_failed", 502), 502, {"error": "ask_failed"}),
    ({"q": "安静"}, True, AskError("ask_unconfigured", 503), 503, {"error": "ask_unconfigured"}),
    ({"q": " 安静 "}, True, None, 200, {"reason": "安静", "items": [{"index": 2}]}),
])
def test_ask_http_queries(tmp_path, body, configured, ask_error, status, expected):
    queries = []

    class FakeAsker:
        def ask(self, query, taste=(), part="all"):
            queries.append(query)
            if ask_error is not None:
                raise ask_error
            return {"reason": "安静", "items": [{"index": 2}]}

    server = make_server(
        tmp_path, 0, data=tmp_path / "data", state=PanelState(), index=_AskIndex(),
        asker=FakeAsker() if configured else None,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    try:
        connection.request("POST", "/api/ask", json.dumps(body).encode("utf-8"),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == status
        assert json.loads(response.read()) == expected
        assert queries == (["安静"] if status == 200 or ask_error is not None else [])
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(("length", "body", "status", "error"), [
    (None, b"", 413, "too_large"),
    ("invalid", b"", 413, "too_large"),
    ("4097", b"", 413, "too_large"),
    ("1", b"{", 400, "bad_query"),
])
def test_ask_http_body_validation(tmp_path, length, body, status, error):
    server = make_server(
        tmp_path, 0, data=tmp_path / "data", state=PanelState(), index=_AskIndex(),
        asker=object(),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    try:
        connection.putrequest("POST", "/api/ask")
        if length is not None:
            connection.putheader("Content-Length", length)
        connection.endheaders(body)
        response = connection.getresponse()
        assert response.status == status
        assert json.loads(response.read()) == {"error": error}
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


from analyzer.butler import Butler


def test_butler_artists_filters_groups_and_uses_original_names():
    content = '```json\n{"groups":[{"names":[0,1,9],"to":1},{"names":[1,2],"to":2},{"names":[3],"to":3},{"names":[4,5],"to":7}]}\n```'
    urlopen, calls = _fake_llm(content)
    artists = [{"name": name, "songs": i + 1} for i, name in enumerate(
        ["周杰倫", "周杰伦", "Jay Chou", "A", "B", "C"]
    )]
    result = Butler("test-key", "https://example.com/v1", "test-model", urlopen=urlopen).artists(artists)
    assert result == {"groups": [{"names": ["周杰倫", "周杰伦"], "to": "周杰伦"}]}
    body = json.loads(calls[0][0].data)
    assert "0|周杰倫|1" in body["messages"][1]["content"]
    assert body["max_tokens"] == 2000
    assert calls[0][1] == 120


def test_butler_songs_keeps_first_change_and_original_id():
    urlopen, calls = _fake_llm('{"songs":[{"i":0,"title":"夜曲"},{"i":0,"title":"x"},{"i":1,"title":"原歌名"},{"i":2,"album":""},{"i":5,"title":"y"}]}')
    songs = [
        {"id": "a", "title": "夜曲 无损", "album": "", "artists": ["周杰伦"], "path": "/音乐/夜曲.flac"},
        {"id": "b", "title": "原歌名", "album": "", "artists": [], "path": "b.flac"},
        {"id": "c", "title": "C", "album": "", "artists": [], "path": "c.flac"},
    ]
    result = Butler("test-key", "https://example.com/v1", "test-model", urlopen=urlopen).songs(songs)
    assert result == {"songs": [{"id": "a", "title": "夜曲"}]}
    body = json.loads(calls[0][0].data)
    assert "/音乐/夜曲.flac" in body["messages"][1]["content"]
    assert body["max_tokens"] == 3000


@pytest.mark.parametrize("method", ["artists", "songs"])
def test_butler_missing_key_never_calls_model(method):
    urlopen, calls = _fake_llm('{}')
    with pytest.raises(AskError) as failure:
        getattr(Butler("", "https://example.com/v1", "test-model", urlopen=urlopen), method)([])
    assert (failure.value.code, failure.value.status) == ("ask_unconfigured", 503)
    assert calls == []


@pytest.mark.parametrize("method", ["artists", "songs"])
@pytest.mark.parametrize("content", [None, "not JSON", "{}"])
def test_butler_network_and_bad_response_errors(method, content):
    def offline(request, timeout):
        raise urllib.error.URLError("offline")
    urlopen = offline if content is None else _fake_llm(content)[0]
    with pytest.raises(AskError) as failure:
        getattr(Butler("test-key", "https://example.com/v1", "test-model", urlopen=urlopen), method)([])
    assert (failure.value.code, failure.value.status) == ("ask_failed", 502)


@pytest.mark.parametrize(("kind", "count", "configured", "status"), [
    ("artists", 0, True, 400), ("artists", 601, True, 400),
    ("artists", 1, False, 503), ("artists", 1, True, 200),
    ("songs", 61, True, 400), ("songs", 1, True, 200),
])
def test_butler_http_limits_and_results(tmp_path, kind, count, configured, status):
    calls = []
    class FakeButler:
        def artists(self, items):
            calls.append(items)
            return {"groups": []}
        def songs(self, items):
            calls.append(items)
            return {"songs": []}
    item = {"name": "周杰伦", "songs": 3} if kind == "artists" else {
        "id": "a", "title": "夜曲", "album": "", "artists": ["周杰伦"], "path": "a.flac",
    }
    server = make_server(tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex(),
                         butler=FakeButler() if configured else None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    try:
        items = [item] * count
        connection.request("POST", "/api/butler/" + kind, json.dumps({kind: items}).encode("utf-8"),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == status
        expected = {"groups" if kind == "artists" else "songs": []} if status == 200 else {
            "error": "ask_unconfigured" if status == 503 else "bad_request",
        }
        assert json.loads(response.read()) == expected
        assert calls == ([items] if status == 200 else [])
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_ask_platform_recommendations_filter_and_deduplicate():
    urlopen, _ = _fake_llm(json.dumps({
        "reason": "r", "picks": [0],
        "songs": [{"title": " 晴天 ", "artist": "周杰伦"}, {"title": "晴天", "artist": "周杰伦"},
                  {"title": "", "artist": "x"}, "bad", {"title": "稻香", "artist": "周杰伦"}],
        "playlists": ["雨天 华语", "雨天 华语", "", "一二三四五六七八九十一二三四五六七八九十一"],
    }))
    result = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen).ask("q")
    assert result["songs"] == [{"title": "晴天", "artist": "周杰伦"}, {"title": "稻香", "artist": "周杰伦"}]
    assert result["playlists"] == ["雨天 华语"]


def test_ask_old_model_content_keeps_library_and_empty_platform_fields():
    urlopen, _ = _fake_llm('{"reason":"r","picks":[0]}')
    result = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen).ask("q")
    assert result["songs"] == []
    assert result["playlists"] == []
    assert [item["index"] for item in result["items"]] == [0]


def test_ask_includes_taste_only_when_given():
    urlopen, calls = _fake_llm('{"picks":[]}')
    asker = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen)
    asker.ask("q", ["陈奕迅", "林俊杰"])
    asker.ask("q")
    first, second = [json.loads(request.data) for request, _ in calls]
    assert "常听歌手：陈奕迅、林俊杰\n要求：q" in first["messages"][1]["content"]
    assert "常听歌手" not in second["messages"][1]["content"]
    assert first["max_tokens"] == 1200


@pytest.mark.parametrize("taste", [["A"] * 21, "A", [""], ["A" * 41], [1]])
def test_ask_http_rejects_invalid_taste(tmp_path, taste):
    urlopen, calls = _fake_llm('{"picks":[]}')
    asker = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen)
    server = make_server(tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex(), asker=asker)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    try:
        connection.request("POST", "/api/ask", json.dumps({"q": "q", "taste": taste}).encode("utf-8"),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {"error": "bad_query"}
        assert calls == []
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("configured", [True, False])
def test_ask_http_status_does_not_expose_key_or_call_model(tmp_path, configured):
    urlopen, calls = _fake_llm('{"picks":[]}')
    asker = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen)
    server = make_server(tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex(),
                         asker=asker if configured else None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/ask/status") as response:
            body = response.read()
            assert response.status == 200
            assert b"test-key" not in body
            assert json.loads(body) == {"configured": configured, "model": "test-model" if configured else ""}
        assert calls == []
    finally:
        server.shutdown()
        server.server_close()



@pytest.fixture(autouse=True)
def _trust_local_client(request, monkeypatch):
    """Tests that are not about access control talk to the server directly."""
    if not request.node.name.startswith("test_access_"):
        monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)


@pytest.mark.parametrize(("address", "forwarded_header", "trusted"), [
    ("192.168.50.23", None, True),
    ("192.168.50.1", None, False),
    ("192.168.96.1", None, False),
    ("192.168.50.2", None, False),
    ("10.0.0.5", None, False),
    ("192.168.50.23", "X-Forwarded-For", False),
    ("192.168.50.23", "X-Real-IP", False),
    ("192.168.50.23", "Forwarded", False),
])
def test_access_trust_requires_direct_home_client(address, forwarded_header, trusted):
    import ipaddress
    from email.message import Message
    from analyzer.serve import _Handler

    handler = object.__new__(_Handler)
    handler._trusted_network = ipaddress.ip_network("192.168.50.0/24")
    handler._host_ip = "192.168.50.2"
    handler.client_address = (address, 12345)
    handler.headers = Message()
    if forwarded_header:
        handler.headers[forwarded_header] = "1.2.3.4"
    assert handler._is_trusted() is trusted


@pytest.mark.parametrize(("method", "path", "token", "authorization", "status"), [
    ("GET", "/api/ask/status", "test-token", None, 401),
    ("GET", "/api/ask/status", "test-token", "Bearer test-token", 200),
    ("GET", "/api/ask/status", "test-token", "Bearer wrong-token", 401),
    ("GET", "/", "test-token", None, 200),
    ("GET", "/xiyue-taste-v1.json.gz", "test-token", None, 401),
    ("GET", "/api/ask/status", "", None, 401),
    ("POST", "/api/scan", "test-token", None, 401),
    ("POST", "/api/scan", "test-token", "Bearer test-token", 200),
    ("GET", "/xiyue-taste-v1%2ejson.gz", "test-token", None, 401),
])
def test_access_http_requires_token_outside_trusted_network(
    tmp_path, method, path, token, authorization, status,
):
    import ipaddress

    state = PanelState()
    server = make_server(
        tmp_path, 0, data=tmp_path, state=state, index=_AskIndex(),
        access_token=token, trusted_network=ipaddress.ip_network("192.168.50.0/24"),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    headers = {"X-Forwarded-For": "1.2.3.4"}
    if authorization is not None:
        headers["Authorization"] = authorization
    try:
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        body = response.read()
        assert response.status == status
        assert b"test-token" not in body
        if status == 401:
            assert response.getheader("WWW-Authenticate") == "Bearer"
            assert response.getheader("Content-Type") == "application/json"
            assert json.loads(body) == {"error": "unauthorized"}
            assert not state.scanRequested.is_set()
        elif path == "/api/ask/status":
            assert json.loads(body) == {"configured": False, "model": ""}
        elif path == "/api/scan":
            assert json.loads(body) == {"ok": True}
            assert state.scanRequested.is_set()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_ask_library_part_sends_catalog_and_small_budget():
    from analyzer.ask import LIBRARY_PROMPT

    urlopen, calls = _fake_llm(json.dumps({
        "reason": "r", "picks": [2, 0, 2, -1],
        "songs": [{"title": "ignored", "artist": "A"}], "playlists": ["ignored"],
    }))
    result = Asker(_AskIndex(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen).ask(
        "q", ["A"], part="library"
    )
    body = json.loads(calls[0][0].data)
    assert body["messages"][0]["content"] == LIBRARY_PROMPT
    assert body["max_tokens"] == 300
    assert "曲库（编号" in body["messages"][1]["content"]
    assert "常听歌手：A\n要求：q" in body["messages"][1]["content"]
    assert result == {
        "reason": "r", "songs": [], "playlists": [],
        "items": [
            {"index": 2, "title": "第三首", "artists": ["歌手"], "path": "third.wav"},
            {"index": 0, "title": "First", "artists": ["A", "B"], "path": "first.wav"},
        ],
    }


def test_ask_online_part_skips_catalog():
    from analyzer.ask import ONLINE_PROMPT

    class NoCatalog:
        def load(self):
            raise AssertionError("Online requests must not load the catalog")

    songs = [{"title": f"song{i}", "artist": "A"} for i in range(12)]
    urlopen, calls = _fake_llm(json.dumps({"songs": songs, "playlists": ["雨天"]}))
    result = Asker(NoCatalog(), "test-key", "https://example.com/v1", "test-model", urlopen=urlopen).ask(
        "q", ["A"], part="online"
    )
    body = json.loads(calls[0][0].data)
    assert body["messages"][0]["content"] == ONLINE_PROMPT
    assert "来自常听歌手的歌最多 3 首" in body["messages"][0]["content"]
    assert body["max_tokens"] == 700
    assert "曲库" not in body["messages"][1]["content"]
    assert body["messages"][1]["content"] == "常听歌手：A\n要求：q"
    assert result == {"reason": "", "songs": songs[:10], "playlists": ["雨天"], "items": []}


@pytest.mark.parametrize(("body", "status", "expected_part"), [
    ({"q": "q", "part": "bogus"}, 400, None),
    ({"q": "q", "part": "online"}, 200, "online"),
    ({"q": "q"}, 200, "all"),
])
def test_ask_http_part(tmp_path, body, status, expected_part):
    parts = []

    class FakeAsker:
        def ask(self, query, taste=(), part="all"):
            parts.append(part)
            return {"reason": "", "songs": [], "playlists": [], "items": []}

    server = make_server(
        tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex(), asker=FakeAsker()
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    try:
        connection.request("POST", "/api/ask", json.dumps(body).encode("utf-8"),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == status
        result = json.loads(response.read())
        if status == 400:
            assert result == {"error": "bad_query"}
            assert parts == []
        else:
            assert parts == [expected_part]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


class _Configurable:
    def __init__(self):
        self.seen = None

    def configure(self, api_key, base_url, model):
        self.seen = (api_key, base_url, model)


def test_llm_settings_save_keeps_the_key_unless_a_new_one_is_given(tmp_path):
    from analyzer.settings import LLMSettings, SettingsError

    path = tmp_path / "llm-settings.json"
    target = _Configurable()
    settings = LLMSettings(path, "env-key", "https://env.example/v1", "env-model", targets=(target,))
    assert target.seen == ("env-key", "https://env.example/v1", "env-model")

    settings.update("https://new.example/v1/", "new-model")
    assert target.seen == ("env-key", "https://new.example/v1", "new-model")
    settings.update("https://new.example/v1", "new-model", "panel-key")
    assert target.seen[0] == "panel-key"
    assert settings.status() == {"configured": True, "baseURL": "https://new.example/v1", "model": "new-model"}
    assert path.stat().st_mode & 0o777 == 0o600

    again = _Configurable()
    LLMSettings(path, "env-key", "https://env.example/v1", "env-model", targets=(again,))
    assert again.seen == ("panel-key", "https://new.example/v1", "new-model")

    for base_url, model, key in [("ftp://x", "m", None), ("https://x", "", None), ("https://x", "m", "a b")]:
        with pytest.raises(SettingsError):
            settings.update(base_url, model, key)


def test_llm_settings_test_reports_the_answer_and_hides_the_key(tmp_path):
    from analyzer.settings import LLMSettings

    urlopen, calls = _fake_llm('{"ok": true}')
    settings = LLMSettings(tmp_path / "s.json", "secret-key", "https://example.com/v1", "m", urlopen=urlopen)
    result = settings.test()
    assert result["ok"] is True and result["model"] == "m" and len(calls) == 1

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b'{"message":"bad secret-key"}'))

    refused = LLMSettings(tmp_path / "s.json", "secret-key", "https://example.com/v1", "m", urlopen=refuse).test()
    assert refused["ok"] is False and refused["error"] == "http_401"
    assert "secret-key" not in refused["detail"]
    assert LLMSettings(tmp_path / "t.json", "", "https://example.com/v1", "m").test()["error"] == "unconfigured"


def test_llm_http_reads_saves_and_never_returns_the_key(tmp_path):
    from analyzer.settings import LLMSettings

    urlopen, _ = _fake_llm('{"picks":[]}')
    asker = Asker(_AskIndex(), "", "https://example.com/v1", "old-model", urlopen=urlopen)
    llm = LLMSettings(tmp_path / "llm-settings.json", "", "https://example.com/v1", "old-model", targets=(asker,))
    server = make_server(tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex(), asker=asker, llm=llm)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        def post(body):
            request = urllib.request.Request(
                base + "/api/llm", data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            return urllib.request.urlopen(request)

        with post({"baseURL": "https://example.com/v1", "model": "new-model", "apiKey": "panel-key"}) as response:
            body = response.read()
            assert b"panel-key" not in body
            assert json.loads(body) == {"configured": True, "baseURL": "https://example.com/v1", "model": "new-model"}
        with urllib.request.urlopen(base + "/api/llm") as response:
            assert b"panel-key" not in response.read()
        assert asker.status() == {"configured": True, "model": "new-model"}
        with pytest.raises(urllib.error.HTTPError) as failure:
            post({"baseURL": "nope", "model": "m"})
        assert failure.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


from analyzer.downloads import Downloads, DownloadError

def _real_flac(seconds=3):
    buffer = io.BytesIO()
    noise = np.random.default_rng(7).uniform(-0.5, 0.5, 44100 * seconds).astype("float32")
    sf.write(buffer, noise, 44100, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


_AUDIO = _real_flac()


def _file_server(files):
    """Serves {path: (status, bytes)} on 127.0.0.1; returns (server, base)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            status, payload = files.get(self.path.split("?", 1)[0], (404, b""))
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _wait_for(downloads, job_id):
    for _ in range(100):
        job = next(item for item in downloads.snapshot() if item["id"] == job_id)
        if job["finishedAt"] is not None:
            return job
        time.sleep(0.05)
    raise AssertionError("download did not finish")


def test_download_saves_the_song_with_lyrics_and_cover_next_to_it(tmp_path):
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        cover = b"\xff\xd8\xff" + bytes(10)
        job_id = downloads.submit({
            "url": base + "/a?secret=1",
            "filename": "歌.flac",
            "directory": "歌手/专辑",
            "lyrics": "[00:01.00]词",
            "cover": base64.b64encode(cover).decode("ascii"),
        })
        job = _wait_for(downloads, job_id)
        assert job["state"] == "done"
        assert job["path"] == "歌手/专辑/歌.flac"
        directory = tmp_path / "歌手" / "专辑"
        assert (directory / "歌.flac").read_bytes() == _AUDIO
        assert (directory / "歌.lrc").read_text(encoding="utf-8") == "[00:01.00]词"
        assert (directory / "歌.jpg").read_bytes() == cover
        assert not any(path.name.startswith(".xiyue-part-") for path in tmp_path.rglob("*"))
    finally:
        server.shutdown()
        server.server_close()


def test_download_keeps_both_when_the_name_is_taken(tmp_path):
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        body = {"url": base + "/a", "filename": "歌.flac"}
        first = _wait_for(downloads, downloads.submit(body))
        second = _wait_for(downloads, downloads.submit(body))
        assert first["state"] == "done"
        assert second["state"] == "done"
        assert second["path"] == "歌 (2).flac"
        assert (tmp_path / "歌.flac").is_file()
        assert (tmp_path / "歌 (2).flac").is_file()
    finally:
        server.shutdown()
        server.server_close()


def test_download_refuses_bad_names_and_addresses(tmp_path):
    downloads = Downloads(tmp_path, allow_private=True)
    body = {"url": "http://127.0.0.1/a", "filename": "歌.flac"}
    for filename in ("../a.flac", ".a.flac", "a.txt", "a", "a/b.flac"):
        with pytest.raises(DownloadError):
            downloads.submit({**body, "filename": filename})
    for directory in ("../x", "a/../b", ".x", "/x"):
        with pytest.raises(DownloadError):
            downloads.submit({**body, "directory": directory})
    for url in ("ftp://x/a.flac", "file:///etc/passwd"):
        with pytest.raises(DownloadError):
            downloads.submit({**body, "url": url})
    with pytest.raises(DownloadError):
        downloads.submit({**body, "cover": "不是base64"})


def test_download_fails_for_a_web_page_and_leaves_nothing(tmp_path):
    server, base = _file_server({"/a": (200, b"<!doctype html>" + bytes(70_000))})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        job = _wait_for(downloads, downloads.submit({"url": base + "/a", "filename": "歌.flac"}))
        assert job["state"] == "failed"
        assert job["error"] == "not_audio"
        assert not any(path.is_file() for path in tmp_path.rglob("*"))
    finally:
        server.shutdown()
        server.server_close()


def test_download_refuses_home_network_addresses_by_default(tmp_path):
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path)
        job = _wait_for(downloads, downloads.submit({"url": base + "/a", "filename": "歌.flac"}))
        assert job["state"] == "failed"
        assert job["error"] == "bad_url"
    finally:
        server.shutdown()
        server.server_close()


def test_download_reports_what_the_server_said(tmp_path):
    server, base = _file_server({"/a": (403, b"")})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        job = _wait_for(downloads, downloads.submit({"url": base + "/a", "filename": "歌.flac"}))
        assert job["state"] == "failed"
        assert job["error"] == "http_403"
    finally:
        server.shutdown()
        server.server_close()


def test_download_http_lists_jobs_without_the_address(tmp_path):
    file_server, file_base = _file_server({"/a": (200, _AUDIO)})
    (tmp_path / "dl").mkdir()
    server = make_server(
        tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex(),
        downloads=Downloads(tmp_path / "dl", allow_private=True),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(
            base + "/api/downloads",
            data=json.dumps({"url": file_base + "/a?secret=1", "filename": "歌.flac"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request) as response:
            job_id = json.load(response)["id"]
        for _ in range(100):
            with urllib.request.urlopen(base + "/api/downloads") as response:
                text = response.read().decode("utf-8")
            result = json.loads(text)
            job = next(item for item in result["jobs"] if item["id"] == job_id)
            if job["state"] == "done":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("download did not finish")
        assert "secret=1" not in text
        assert result["available"] is True
    finally:
        server.shutdown()
        server.server_close()
        file_server.shutdown()
        file_server.server_close()

    server = make_server(tmp_path, 0, data=tmp_path, state=PanelState(), index=_AskIndex())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/downloads",
            data=b"{}", headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request)
        assert failure.value.code == 503
    finally:
        server.shutdown()
        server.server_close()


def test_download_locates_the_folder_the_phone_marked(tmp_path):
    directory = tmp_path / "歌手" / "专辑"
    directory.mkdir(parents=True)
    name = ".xiyue-probe-" + "a" * 32
    marker = directory / name
    marker.write_text("")
    downloads = Downloads(tmp_path)
    assert downloads.locate(name) == "歌手/专辑"
    marker.rename(tmp_path / name)
    assert downloads.locate(name) == ""
    with pytest.raises(DownloadError) as missing:
        downloads.locate(".xiyue-probe-" + "b" * 32)
    assert missing.value.status == 404
    for invalid in ("../x", "歌.flac"):
        with pytest.raises(DownloadError) as failure:
            downloads.locate(invalid)
        assert failure.value.status == 400


def test_download_does_not_follow_a_link_out_of_the_folder(tmp_path):
    root, outside = tmp_path / "dl", tmp_path / "out"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(root, allow_private=True)
        for directory in ("linked", "linked/子"):
            job = _wait_for(downloads, downloads.submit({
                "url": base + "/a", "filename": "歌.flac", "directory": directory,
            }))
            assert job["state"] == "failed"
            assert job["error"] == "outside"
            assert list(outside.rglob("*")) == []
        (root / "歌.lrc").symlink_to(outside / "x.lrc")
        job = _wait_for(downloads, downloads.submit({
            "url": base + "/a", "filename": "歌.flac", "lyrics": "[00:01.00]词",
        }))
        assert job["state"] == "done"
        assert list(outside.rglob("*")) == []
    finally:
        server.shutdown()
        server.server_close()


def test_download_reports_what_the_file_really_is(tmp_path):
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        job = _wait_for(downloads, downloads.submit({"url": base + "/a", "filename": "歌.flac"}))
        assert job["state"] == "done"
        assert job["format"] == "flac"
        assert job["sampleRate"] == 44100
        assert job["bitDepth"] == 16
        assert 2900 <= job["durationMs"] <= 3100
        assert job["bitRate"] > 0
    finally:
        server.shutdown()
        server.server_close()


def test_download_names_the_file_by_its_real_format(tmp_path):
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        job = _wait_for(downloads, downloads.submit({
            "url": base + "/a", "filename": "歌.mp3", "lyrics": "词",
        }))
        assert job["state"] == "done"
        assert job["path"] == "歌.flac"
        assert (tmp_path / "歌.lrc").exists()
        assert not (tmp_path / "歌.mp3").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_download_refuses_a_song_cut_short_and_leaves_nothing(tmp_path):
    server, base = _file_server({"/a": (200, _AUDIO)})
    try:
        downloads = Downloads(tmp_path, allow_private=True)
        body = {"url": base + "/a", "filename": "歌.flac", "minDurationMs": 60000}
        job = _wait_for(downloads, downloads.submit(body))
        assert job["state"] == "failed"
        assert job["error"] == "too_short"
        assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
        for minimum in (-1, "3", True):
            with pytest.raises(DownloadError):
                downloads.submit({**body, "minDurationMs": minimum})
    finally:
        server.shutdown()
        server.server_close()
