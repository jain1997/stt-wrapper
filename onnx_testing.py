#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speaker Diarization (CUDA, auto→2-speaker merge + overlap detection)
✅ Merges consecutive segments from the same speaker
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

# --- IMPORTANT: Update these paths to match your local files ---
AUDIO_FILE = "/Users/gauravjain/Documents/work/stt-wrapper/test_data/speaker_diarization.wav"  # Path to your audio file (must be mono 16-bit WAV)
SEGMENTATION_MODEL = "/Users/gauravjain/Downloads/sherpa-onnx-reverb-diarization-v1/model.onnx"  # Path to local segmentation model
EMBEDDING_MODEL = "/Users/gauravjain/Downloads/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx" # Path to local embedding model

NUM_SPEAKERS = 0          # Set to 0 to auto-detect number of speakers
TARGET_SPEAKERS = 2       # Merge clusters down to this many speakers
THRESHOLD = 0.25          # Clustering threshold for initial diarization
MIN_ON = 0.05             # Minimum duration for a speaker segment
MIN_OFF = 0.05            # Minimum duration for a pause between segments
OVERLAP_WINDOW = 0.10     # Seconds around a speaker turn to check for overlap
USE_CUDA = True           # Set to True to use GPU, False for CPU
SAVE_SRT = True           # Set to True to save results as an SRT file
MERGE_GAP_SECONDS = 0.15  # Max gap between segments to merge, if speaker is the same

# ===============================================================


def load_audio(path: str, target_sr: int = 16000):
    """Load audio and resample to the target sample rate."""
    print(f"Loading {path} ...")
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = audio.astype(np.float32)
    print(f"Loaded {len(audio)/sr:.2f}s @ {sr}Hz (mono)")
    return audio, sr


def build_diarizer(segmentation_model_path, embedding_model_path,
                   num_clusters, threshold, min_on, min_off, use_cuda):
    """Create and configure the sherpa-onnx diarizer."""
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
        raise RuntimeError("Invalid diarization config. Please check model paths.")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def to_srt_time(sec: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,ms)."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def merge_to_target_speakers(segments, target_speakers, embedding_model_path, audio, sample_rate):
    """
    Merge diarization output to exactly target_speakers clusters using embeddings.
    """
    if not segments:
        return segments

    speakers = sorted(set(seg.speaker for seg in segments))
    if len(speakers) <= target_speakers:
        print(f"Found {len(speakers)} speakers, which is not more than the target of {target_speakers}. No merge needed.")
        return segments

    print(f"🔍 Found {len(speakers)} initial clusters. Extracting embeddings to merge into {target_speakers}...")

    provider = "cuda" if USE_CUDA else "cpu"
    embedding_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=embedding_model_path,
        provider=provider,
        debug=False,
    )
    embedder = sherpa_onnx.SpeakerEmbeddingExtractor(embedding_cfg)

    embeddings, segment_refs = [], []

    for seg in segments:
        start_frame = int(seg.start * sample_rate)
        end_frame = int(seg.end * sample_rate)
        segment_audio = audio[start_frame:end_frame]

        if len(segment_audio) < sample_rate * 0.2:
            continue

        stream = embedder.create_stream()
        stream.accept_waveform(sample_rate, segment_audio)
        stream.input_finished()
        emb = embedder.compute(stream)

        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        segment_refs.append(seg)

    if not embeddings:
        print("⚠️ No valid segments were found for embedding extraction — skipping merge.")
        return segments

    embeddings = np.vstack(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=target_speakers, metric="cosine", linkage="average"
    )
    new_labels = clustering.fit_predict(embeddings)

    merged_segments = []
    for seg, label in zip(segment_refs, new_labels):
        merged_segments.append(
            type("MergedSegment", (object,), {
                "start": seg.start,
                "end": seg.end,
                "speaker": int(label)
            })()
        )

    print(f"🔁 Merged {len(speakers)} → {target_speakers} speakers using embedding-based clustering.")
    return merged_segments


def merge_consecutive_segments(segments, max_gap_seconds=0.1):
    """
    Merge consecutive segments from the same speaker if the gap between them is minimal.
    """
    if len(segments) < 2:
        return segments

    print("Attempting to merge consecutive segments from the same speaker...")
    
    # Ensure segments are sorted by start time
    segments.sort(key=lambda s: s.start)

    merged = []
    if not segments:
        return []

    # Start with the first segment
    current_merge = segments[0]

    for next_seg in segments[1:]:
        # Check if speaker is the same and the gap is within the threshold
        if (next_seg.speaker == current_merge.speaker and
            (next_seg.start - current_merge.end) <= max_gap_seconds):
            # Extend the end time of the current merged segment
            current_merge.end = next_seg.end
        else:
            # Finalize the previous merge and start a new one
            merged.append(current_merge)
            current_merge = next_seg

    # Append the very last segment
    merged.append(current_merge)

    print(f"🔁 Merged consecutive utterances: {len(segments)} segments → {len(merged)} segments.")
    return merged


def detect_overlaps(segments, sr, audio, overlap_window=0.1):
    """
    Detect likely overlap zones when speaker turns are within 'overlap_window' seconds.
    """
    overlaps = []
    if not segments:
        return overlaps
        
    segments.sort(key=lambda seg: seg.start)

    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]
        gap = nxt.start - cur.end
        
        if gap < overlap_window and cur.speaker != nxt.speaker:
            start_olap = max(cur.end - overlap_window / 2, 0)
            end_olap = min(nxt.start + overlap_window / 2, len(audio) / sr)
            overlaps.append((start_olap, end_olap, cur.speaker, nxt.speaker))
    return overlaps


def save_srt_with_overlap(segments, overlaps, output_path):
    """Save speaker segments and detected overlaps to an SRT file."""
    all_events = []
    for seg in segments:
        all_events.append((seg.start, seg.end, f"Speaker_{seg.speaker:02d}"))
    for (st, et, spk1, spk2) in overlaps:
        all_events.append((st, et, f"Overlap: Speaker_{spk1:02d} + Speaker_{spk2:02d}"))
    
    all_events.sort(key=lambda x: x[0])

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(all_events, 1):
            f.write(f"{idx}\n")
            f.write(f"{to_srt_time(start)} --> {to_srt_time(end)}\n")
            f.write(f"{text}\n\n")

    print(f"✅ SRT file with overlaps saved to: {output_path}")


def main():
    """Main function to run the diarization pipeline."""
    for p in [AUDIO_FILE, SEGMENTATION_MODEL, EMBEDDING_MODEL]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print("\n=== Speaker Diarization (Local - CUDA, 2-Speaker + Overlap) ===")
    print(f"Audio file        : {AUDIO_FILE}")
    print(f"Segmentation model: {SEGMENTATION_MODEL}")
    print(f"Embedding model   : {EMBEDDING_MODEL}")
    print(f"Num speakers      : {NUM_SPEAKERS} (auto-detect)")
    print(f"Target speakers   : {TARGET_SPEAKERS}")
    print(f"Merge gap         : {MERGE_GAP_SECONDS}s\n")

    # Step 1: Load Audio
    audio, sample_rate = load_audio(AUDIO_FILE, target_sr=16000)

    # Step 2: Build Diarizer
    diarizer = build_diarizer(
        SEGMENTATION_MODEL, EMBEDDING_MODEL,
        NUM_SPEAKERS, THRESHOLD, MIN_ON, MIN_OFF, USE_CUDA
    )

    # Step 3: Run Initial Diarization
    print("Running initial diarization...")
    t0 = time.time()
    segments = diarizer.process(audio).sort_by_start_time()
    t1 = time.time()
    print(f"Initial diarization found {len(set(s.speaker for s in segments))} speakers from {len(segments)} segments.")

    # Step 4: Merge clusters to the target number of speakers
    if TARGET_SPEAKERS > 0:
        segments = merge_to_target_speakers(segments, TARGET_SPEAKERS, EMBEDDING_MODEL, audio, sample_rate)

    # Step 5: Merge consecutive segments from the same speaker
    segments = merge_consecutive_segments(segments, max_gap_seconds=MERGE_GAP_SECONDS)

    # Step 6: Detect overlaps based on the final, merged segments
    overlaps = detect_overlaps(segments, sample_rate, audio, overlap_window=OVERLAP_WINDOW)

    # Step 7: Print results to console
    print("\n--- Final Diarization Segments ---")
    if segments:
        for seg in segments:
            print(f"[{to_srt_time(seg.start)} --> {to_srt_time(seg.end)}] Speaker_{seg.speaker:02d}")
    else:
        print("No segments found.")
    print("----------------------------------")

    if overlaps:
        print("\n--- Detected Overlaps ---")
        for st, et, s1, s2 in overlaps:
            print(f"[{to_srt_time(st)} --> {to_srt_time(et)}] Overlap: Speaker_{s1:02d} & Speaker_{s2:02d}")
        print("----------------------------------")

    duration = len(audio) / diarizer.sample_rate
    elapsed = t1 - t0
    rtf = elapsed / duration
    print(f"\nAudio duration  : {duration:.3f}s")
    print(f"Processing time : {elapsed:.3f}s (for initial diarization)")
    print(f"Real-Time Factor: {rtf:.3f}\n")

    # Step 8: Save results to SRT file
    if SAVE_SRT:
        out_srt = os.path.splitext(AUDIO_FILE)[0] + "_diarization.srt"
        save_srt_with_overlap(segments, overlaps, out_srt)


if __name__ == "__main__":
    main()