"""Frame-aligned music beat extraction."""

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/edge-numba-cache")

import librosa
import numpy as np


def _get_tempo(audio_name):
    fields = audio_name.split("_")
    music = next((field for field in fields if field.startswith("m") and len(field) == 4), None)
    if music is None:
        raise ValueError("cannot infer AIST++ tempo from {}".format(audio_name))
    if music[:3] in ("mBR", "mPO", "mLO", "mMH", "mLH", "mWA", "mKR", "mJS", "mJB"):
        return int(music[3]) * 10 + 80
    if music[:3] == "mHO":
        return int(music[3]) * 5 + 110
    raise ValueError("unknown AIST++ music id: {}".format(music))


def extract_music_beat_features(audio_path, fps=30):
    sample_rate = 44100
    hop_length = int(round(sample_rate / float(fps)))
    audio, _ = librosa.load(str(audio_path), sr=sample_rate)
    envelope = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length
    )
    try:
        start_bpm = _get_tempo(Path(audio_path).stem)
    except ValueError:
        start_bpm = float(librosa.beat.tempo(y=audio, sr=sample_rate)[0])
    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,
    )
    beats = np.zeros_like(envelope, dtype=bool)
    beats[beat_indices] = True
    return beats
