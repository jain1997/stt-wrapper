#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speaker Diarization (local, CUDA-enabled)
✅ Uses librosa for resampling (no ffmpeg)
✅ Automatically switches between CUDA and CPU
✅ Works with long recordings (minutes)
"""

import os
import time
import numpy as np
import librosa
import sherpa_onnx
from datetime import datetime

# ===============================================================
# 🔧 CONFIGURATION
# ===============================================================

AUDIO_FILE = "/Users/gauravjain/Documents/work/stt-wrapper/test_data/speaker_diarization.wav"
SEGMENTATION_MODEL = "/Users/gauravjain/Downloads/sherpa-onnx-reverb-diarization-v1/model.onnx"
EMBEDDING_MODEL = "/Users/gauravjain/Downloads/wespeaker_en_voxceleb_resnet34.onnx"

NUM_SPEAKERS = 2      # 0 = auto-detect
THRESHOLD = 0.0       # clustering threshold (if NUM_SPEAKERS == 0)
MIN_ON = 0.3
MIN_OFF = 0.5
SAVE_SRT = True
USE_CUDA = True        # ✅ enable GPU (set False to force CPU)

# ===============================================================


def load_audio(path: str, target_sr: int = 16000):
    """Load an audio file with librosa and resample to 16kHz mono."""
    print(f"Loading {path} ...")
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = audio.astype(np.float32)
    print(f"Loaded {len(audio)/sr:.2f}s @ {sr}Hz (mono)")
    return audio, sr


def build_diarizer(
    segmentation_model_path: str,
    embedding_model_path: str,
    num_clusters: int,
    threshold: float,
    min_on: float,
    min_off: float,
    use_cuda: bool = False,
):
    """Create sherpa_onnx OfflineSpeakerDiarization instance with GPU/CPU support."""
    # Configure device settings for both segmentation and embedding
    provider = "cuda" if use_cuda else "cpu"
    print(f"🚀 Using {'CUDA GPU' if use_cuda else 'CPU'} for inference")

    segmentation_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=segmentation_model_path,
        ),
        provider=provider,
        debug=False,
    )

    embedding_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=embedding_model_path,
        provider=provider,
        debug=False,
    )

    clustering_cfg = sherpa_onnx.FastClusteringConfig(
        num_clusters=num_clusters,
        threshold=threshold,
    )

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation_cfg,
        embedding=embedding_cfg,
        clustering=clustering_cfg,
        min_duration_on=min_on,
        min_duration_off=min_off,
    )

    if not cfg.validate():
        raise RuntimeError("Invalid diarization config. Check model paths or ONNXRuntime GPU install.")

    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def to_srt_time(sec: float) -> str:
    """Convert seconds to SRT time string (HH:MM:SS,mmm)"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def save_as_srt(segments, output_path):
    """Save diarization segments to .srt"""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{to_srt_time(seg.start)} --> {to_srt_time(seg.end)}\n")
            f.write(f"Speaker_{seg.speaker:02d}\n\n")
    print(f"SRT saved to: {output_path}")


def main():
    # === Validate Inputs ===
    for p in [AUDIO_FILE, SEGMENTATION_MODEL, EMBEDDING_MODEL]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    print("\n=== Speaker Diarization (Local - CUDA) ===")
    print(f"Audio file        : {AUDIO_FILE}")
    print(f"Segmentation model: {SEGMENTATION_MODEL}")
    print(f"Embedding model   : {EMBEDDING_MODEL}")
    print(f"Num speakers      : {NUM_SPEAKERS}")
    print(f"Threshold         : {THRESHOLD}")
    print("=============================================\n")

    # === Step 1: Load Audio ===
    audio, sample_rate = load_audio(AUDIO_FILE, target_sr=16000)

    # === Step 2: Build Diarizer (CUDA/CPU) ===
    diarizer = build_diarizer(
        segmentation_model_path=SEGMENTATION_MODEL,
        embedding_model_path=EMBEDDING_MODEL,
        num_clusters=NUM_SPEAKERS,
        threshold=THRESHOLD,
        min_on=MIN_ON,
        min_off=MIN_OFF,
        use_cuda=USE_CUDA,
    )

    # === Step 3: Run Diarization ===
    print("Running diarization...")
    t0 = time.time()
    segments = diarizer.process(audio).sort_by_start_time()
    t1 = time.time()

    # === Step 4: Print Results ===
    print("\n--- Diarization Segments ---")
    for seg in segments:
        print(f"{seg.start:.3f} -- {seg.end:.3f} speaker_{seg.speaker:02d}")
    print("-----------------------------")

    duration = len(audio) / diarizer.sample_rate
    elapsed = t1 - t0
    rtf = elapsed / duration
    print(f"\nAudio duration  : {duration:.3f}s")
    print(f"Processing time : {elapsed:.3f}s")
    print(f"RTF             : {rtf:.3f}\n")

    # === Step 5: Save SRT ===
    if SAVE_SRT:
        out_srt = os.path.splitext(AUDIO_FILE)[0] + "_diarization.srt"
        save_as_srt(segments, out_srt)


if __name__ == "__main__":
    main()
