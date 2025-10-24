"""Audio loading, resampling, and utility helpers."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import logging
from typing import List, Tuple

log = logging.getLogger(__name__)


def load_audio(file_path: Path) -> Tuple[np.ndarray, int]:
    """Load an audio file (float32)."""
    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    return data, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Resample if needed."""
    if orig_sr == target_sr:
        return audio, orig_sr
    log.info("Resampling %d → %d Hz", orig_sr, target_sr)
    mono = audio.mean(axis=1)
    resampled = librosa.resample(y=mono, orig_sr=orig_sr, target_sr=target_sr)
    return resampled[:, None].astype("float32"), target_sr


def split_audio(audio: np.ndarray, sr: int, chunk_sec: int) -> Tuple[List[np.ndarray], List[float]]:
    """Split audio into fixed-size chunks."""
    step = int(chunk_sec * sr)
    if step <= 0 or step >= len(audio):
        return [audio], [len(audio) / sr]
    chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
    durations = [len(c) / sr for c in chunks]
    return chunks, durations


def srt_time(seconds: float) -> str:
    """Convert seconds → SRT timestamp."""
    hr = int(seconds // 3600)
    seconds %= 3600
    mn = int(seconds // 60)
    seconds %= 60
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hr:02}:{mn:02}:{int(seconds):02},{ms:03}"
