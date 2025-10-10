#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speaker Diarization (CUDA, auto→2-speaker merge + overlap detection)
✅ Uses librosa (no ffmpeg)
✅ GPU-accelerated via sherpa-onnx
✅ Handles fast turns (<0.1 s)
✅ Detects overlap zones between speakers
"""

import os
import time
import numpy as np
import librosa
import sherpa_onnx
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering

# ===============================================================
# 🔧 CONFIGURATION
# ===============================================================

AUDIO_FILE = ""
SEGMENTATION_MODEL = ""
EMBEDDING_MODEL = ""

NUM_SPEAKERS = 0          # auto-detect
TARGET_SPEAKERS = 2       # merge down to 2
THRESHOLD = 0.25          # used for auto clustering
MIN_ON = 0.05             # accept very short bursts
MIN_OFF = 0.05            # detect short pauses
OVERLAP_WINDOW = 0.10     # seconds to treat as overlap
USE_CUDA = True
SAVE_SRT = True

# ===============================================================


def load_audio(path: str, target_sr: int = 16000):
    """Load audio and resample."""
    print(f"Loading {path} ...")
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = audio.astype(np.float32)
    print(f"Loaded {len(audio)/sr:.2f}s @ {sr}Hz (mono)")
    return audio, sr


def build_diarizer(segmentation_model_path, embedding_model_path,
                   num_clusters, threshold, min_on, min_off, use_cuda):
    """Create sherpa_onnx diarizer config."""
    provider = "cuda" if use_cuda else "cpu"
    print(f"🚀 Using {'CUDA GPU' if use_cuda else 'CPU'} for inference")

    segmentation_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=segmentation_model_path
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
        raise RuntimeError("Invalid diarization config. Check model paths.")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def to_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def merge_to_target_speakers(segments, target_speakers):
    """Merge dynamic clusters to exactly target_speakers."""
    if not segments:
        return segments

    speakers = sorted(set(seg.speaker for seg in segments))
    if len(speakers) <= target_speakers:
        return segments  # already fine

    # Simple heuristic: merge by temporal proximity
    centers = np.array([[np.mean([seg.start, seg.end])] for seg in segments])
    clustering = AgglomerativeClustering(
        n_clusters=target_speakers, affinity="euclidean", linkage="average"
    )
    new_labels = clustering.fit_predict(centers)
    mapping = {}
    for old, new in zip([seg.speaker for seg in segments], new_labels):
        mapping[old] = new

    for seg in segments:
        seg.speaker = mapping.get(seg.speaker, seg.speaker)

    print(f"🔁 Merged from {len(speakers)} → {target_speakers} speakers")
    return segments


def detect_overlaps(segments, sr, audio, overlap_window=0.1):
    """
    Detect likely overlap zones when speaker turns are within 'overlap_window' seconds.
    Returns list of (start, end, spk1, spk2).
    """
    overlaps = []
    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]
        gap = nxt.start - cur.end
        if gap < overlap_window and cur.speaker != nxt.speaker:
            start_olap = max(cur.end - overlap_window / 2, 0)
            end_olap = min(nxt.start + overlap_window / 2, len(audio)/sr)
            overlaps.append((start_olap, end_olap, cur.speaker, nxt.speaker))
    return overlaps


def save_srt_with_overlap(segments, overlaps, output_path):
    """Save speaker and overlap segments to SRT."""
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            f.write(f"{idx}\n")
            f.write(f"{to_srt_time(seg.start)} --> {to_srt_time(seg.end)}\n")
            f.write(f"Speaker_{seg.speaker:02d}\n\n")
            idx += 1

        for (st, et, spk1, spk2) in overlaps:
            f.write(f"{idx}\n")
            f.write(f"{to_srt_time(st)} --> {to_srt_time(et)}\n")
            f.write(f"Overlap: Speaker_{spk1:02d} + Speaker_{spk2:02d}\n\n")
            idx += 1

    print(f"SRT with overlaps saved: {output_path}")


def main():
    for p in [AUDIO_FILE, SEGMENTATION_MODEL, EMBEDDING_MODEL]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    print("\n=== Speaker Diarization (Local - CUDA, 2-Speaker + Overlap) ===")
    print(f"Audio file        : {AUDIO_FILE}")
    print(f"Segmentation model: {SEGMENTATION_MODEL}")
    print(f"Embedding model   : {EMBEDDING_MODEL}")
    print(f"Num speakers      : {NUM_SPEAKERS} (auto)")
    print(f"Threshold         : {THRESHOLD}")
    print(f"Target speakers   : {TARGET_SPEAKERS}")
    print(f"Overlap window    : {OVERLAP_WINDOW}s\n")

    # Step 1: Load Audio
    audio, sample_rate = load_audio(AUDIO_FILE, target_sr=16000)

    # Step 2: Build Diarizer
    diarizer = build_diarizer(
        SEGMENTATION_MODEL, EMBEDDING_MODEL,
        NUM_SPEAKERS, THRESHOLD, MIN_ON, MIN_OFF, USE_CUDA
    )

    # Step 3: Run Diarization
    print("Running diarization...")
    t0 = time.time()
    segments = diarizer.process(audio).sort_by_start_time()
    t1 = time.time()

    # Step 4: Merge to exactly 2 speakers
    segments = merge_to_target_speakers(segments, TARGET_SPEAKERS)

    # Step 5: Detect overlaps
    overlaps = detect_overlaps(segments, sample_rate, audio, overlap_window=OVERLAP_WINDOW)

    # Step 6: Print results
    print("\n--- Final Diarization Segments ---")
    for seg in segments:
        print(f"{seg.start:.3f} -- {seg.end:.3f} speaker_{seg.speaker:02d}")
    print("----------------------------------")

    if overlaps:
        print("\n--- Detected Overlaps ---")
        for st, et, s1, s2 in overlaps:
            print(f"{st:.3f}-{et:.3f}s : speaker_{s1:02d} ↔ speaker_{s2:02d}")
        print("----------------------------------")

    duration = len(audio) / diarizer.sample_rate
    elapsed = t1 - t0
    rtf = elapsed / duration
    print(f"\nAudio duration  : {duration:.3f}s")
    print(f"Processing time : {elapsed:.3f}s")
    print(f"RTF             : {rtf:.3f}\n")

    # Step 7: Save
    if SAVE_SRT:
        out_srt = os.path.splitext(AUDIO_FILE)[0] + "_2speaker_overlap.srt"
        save_srt_with_overlap(segments, overlaps, out_srt)


if __name__ == "__main__":
    main()
