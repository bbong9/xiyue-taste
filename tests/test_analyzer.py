import gzip
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mutagen.flac
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import ANALYZER_VERSION
from analyzer.features import extract_features
from analyzer.output import write_output
from analyzer.scan import scan
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
