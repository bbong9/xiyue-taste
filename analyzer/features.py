import subprocess

import librosa
import numpy as np
import pyloudnorm


SAMPLE_RATE = 22050


def _duration(path):
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(output.strip())


def _decode(path, offset):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{offset:.3f}", "-t", "60", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _tempo(onset_envelope, sample_rate):
    """The tempo librosa.beat.beat_track reports, without the beat tracker
    that follows it: that part is compiled code which crashes on some machines
    and we never used its result."""
    if not onset_envelope.any():
        return 0.0
    return float(np.asarray(librosa.feature.tempo(onset_envelope=onset_envelope, sr=sample_rate)).item())


def _onset_count(onset_envelope, sample_rate, hop_length=512):
    """How many onsets librosa.onset.onset_detect finds with its defaults,
    picked in plain Python for the same reason as _tempo."""
    envelope = onset_envelope - np.min(onset_envelope)
    envelope = envelope / (np.max(envelope) + librosa.util.tiny(envelope))
    if not envelope.any() or not np.all(np.isfinite(envelope)):
        return 0
    pre_max = int(np.ceil(0.03 * sample_rate // hop_length))
    post_max = int(np.ceil(0.00 * sample_rate // hop_length + 1))
    pre_avg = int(np.ceil(0.10 * sample_rate // hop_length))
    post_avg = int(np.ceil(0.10 * sample_rate // hop_length + 1))
    wait = int(np.ceil(0.03 * sample_rate // hop_length))
    delta = 0.07
    size = envelope.shape[0]
    count = 0
    first = (
        envelope[0] >= np.max(envelope[: min(post_max, size)])
        and envelope[0] >= np.mean(envelope[: min(post_avg, size)]) + delta
    )
    if first:
        count += 1
    index = wait + 1 if first else 1
    while index < size:
        value = envelope[index]
        if value != np.max(envelope[max(0, index - pre_max) : min(index + post_max, size)]):
            index += 1
            continue
        if value < np.mean(envelope[max(0, index - pre_avg) : min(index + post_avg, size)]) + delta:
            index += 1
            continue
        count += 1
        index += wait + 1
    return count


def analyze_audio(audio):
    sample_rate = SAMPLE_RATE
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    vector = [*mfcc.mean(axis=1), *mfcc.std(axis=1)]
    for feature in (
        librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
        librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate),
        librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
        librosa.feature.spectral_flatness(y=audio),
    ):
        vector.extend((feature.mean(), feature.std()))
    vector.extend(librosa.feature.chroma_stft(y=audio, sr=sample_rate).mean(axis=1))
    for feature in (
        librosa.feature.zero_crossing_rate(y=audio),
        librosa.feature.rms(y=audio),
    ):
        vector.extend((feature.mean(), feature.std()))
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    bpm = _tempo(onset_envelope, sample_rate)
    onset_rate = _onset_count(onset_envelope, sample_rate) / (len(audio) / sample_rate)
    loudness = float(pyloudnorm.Meter(SAMPLE_RATE).integrated_loudness(audio))
    vector.extend((bpm, onset_rate, loudness))
    return [float(value) for value in vector], bpm, loudness


def extract_features(path):
    duration = _duration(path)
    offset = duration * 0.3 if duration >= 60 else 0
    audio = _decode(path, offset)
    vector, bpm, loudness = analyze_audio(audio)
    return {
        "vector": vector,
        "bpm": bpm,
        "loudnessLUFS": loudness,
        "durationSec": duration,
    }
