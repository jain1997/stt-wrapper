#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speaker Diarization + Local Oriserve Whisper-Hindi2Hinglish-Prime ASR
→ Full .srt with speaker labels and text
"""

import os
import time
import librosa
import numpy as np
import sherpa_onnx
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# ===============================================================
# 🔧 CONFIGURATION
# ===============================================================

AUDIO_FILE = "/Users/gauravjain/Documents/work/stt-wrapper/test_data/speaker_diarization.wav"  # Path to your audio file (must be mono 16-bit WAV)
SEGMENTATION_MODEL = "/Users/gauravjain/Downloads/sherpa-onnx-reverb-diarization-v1/model.onnx"  # Path to local segmentation model
EMBEDDING_MODEL = "/Users/gauravjain/Downloads/wespeaker_en_voxceleb_resnet34.onnx"    
LOCAL_WHISPER_MODEL_DIR = "/Users/gauravjain/Downloads/prime"

NUM_SPEAKERS = 2
THRESHOLD = 0
MIN_ON = 0.3
MIN_OFF = 0.5
CHUNK_DURATION = 30.0
SAVE_SRT = True

# ===============================================================


def load_audio_16k(path: str, target_sr: int = 16000):
    """Load and resample to 16 kHz mono."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = y.astype(np.float32)
    print(f"Loaded {len(y)/sr:.2f}s @ {sr}Hz")
    return y, sr


def build_diarizer(seg_path: str, emb_path: str):
    """Build sherpa-onnx diarizer."""
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_path),
            debug=False,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_path, debug=False),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=NUM_SPEAKERS, threshold=THRESHOLD),
        min_duration_on=MIN_ON,
        min_duration_off=MIN_OFF,
    )
    if not cfg.validate():
        raise RuntimeError("Invalid diarization config.")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def to_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def save_srt(entries, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (st, et, spk, text) in enumerate(entries, start=1):
            f.write(f"{i}\n")
            f.write(f"{to_srt_time(st)} --> {to_srt_time(et)}\n")
            f.write(f"{spk}: {text.strip()}\n\n")
    print(f"SRT saved: {out_path}")


def load_local_whisper_pipeline(local_dir: str, chunk_dur: float = 30.0):
    """Load Whisper Prime safely (disable timestamps completely)."""
    print(f"Loading Whisper Prime from: {local_dir}")
    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(local_dir)

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_dur,
        framework="pt",
        device=0 if model.device.type == "cuda" else -1,
    )

    print("✅ Whisper Prime loaded successfully (timestamps disabled).")
    return pipe


def transcribe_clip(asr_pipe, audio, sr, start, end):
    """Run ASR on a diarized segment (no timestamp args)."""
    s0, s1 = int(start * sr), int(end * sr)
    segment = audio[s0:s1]
    if len(segment) == 0:
        return ""
    try:
        # Ensure no timestamp options are passed internally
        result = asr_pipe(segment)
        text = result.get("text", "").strip()
        return text
    except Exception as e:
        print(f"⚠️ ASR failed {start:.2f}-{end:.2f}s → {e}")
        return ""


def main():
    audio, sr = load_audio_16k(AUDIO_FILE)
    diarizer = build_diarizer(SEGMENTATION_MODEL, EMBEDDING_MODEL)
    asr_pipe = load_local_whisper_pipeline(LOCAL_WHISPER_MODEL_DIR, CHUNK_DURATION)

    print("\nRunning diarization...")
    t0 = time.time()
    segments = diarizer.process(audio).sort_by_start_time()
    t1 = time.time()
    print(f"✔️ Diarization done in {t1 - t0:.2f}s")

    print("\nTranscribing segments (Whisper Prime, no timestamps)...")
    entries = []
    for seg in segments:
        st, et = seg.start, seg.end
        spk = f"speaker_{seg.speaker:02d}"
        txt = transcribe_clip(asr_pipe, audio, sr, st, et)
        entries.append((st, et, spk, txt))
        print(f"{st:.3f}-{et:.3f}s [{spk}] → {txt}")

    if SAVE_SRT:
        out_srt = os.path.splitext(AUDIO_FILE)[0] + "_final_diarization.srt"
        save_srt(entries, out_srt)


if __name__ == "__main__":
    main()