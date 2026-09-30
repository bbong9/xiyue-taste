import librosa
import numpy as np
import pyloudnorm


def extract_features(path):
    duration = float(librosa.get_duration(path=path))
    offset = duration * 0.3 if duration >= 60 else 0
    audio, sample_rate = librosa.load(
        path, sr=22050, mono=True, offset=offset, duration=60
    )
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
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sample_rate)
    bpm = float(np.asarray(tempo).item())
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sample_rate)
    onset_rate = len(onsets) / (len(audio) / sample_rate)
    loudness = float(pyloudnorm.Meter(22050).integrated_loudness(audio))
    vector.extend((bpm, onset_rate, loudness))
    return {
        "vector": [float(value) for value in vector],
        "bpm": bpm,
        "loudnessLUFS": loudness,
        "durationSec": duration,
    }
