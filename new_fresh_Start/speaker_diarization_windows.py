import os
from pathlib import Path
import soundfile as sf
import librosa
import torch
import numpy as np
from tqdm import tqdm
import sherpa_onnx
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)


# ------------------------------
# 🧠 Thread safety for Sherpa
# ------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


# ------------------------------
# 🔄 Resample Audio
# ------------------------------
def resample_audio(audio, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        print(f"\n🔄 Resampling {sample_rate} Hz → {target_sample_rate} Hz...")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        print(f"✅ Resampled. New shape: {audio.shape}")
    return audio, target_sample_rate


# ------------------------------
# 🎛️ Split Audio into Chunks
# ------------------------------
def split_audio(audio, sr, chunk_duration_sec=300):
    """Split audio into fixed-duration chunks (default: 5 min)."""
    samples_per_chunk = int(chunk_duration_sec * sr)
    return [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]


# ------------------------------
# 🧩 Init Sherpa Diarization
# ------------------------------
def init_speaker_diarization(num_speakers=-1, cluster_threshold=0.5):
    segmentation_model = "/Users/gauravjain/Documents/work/exported-assets/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_extractor_model = "/Users/gauravjain/Downloads/nemo_en_speakerverification_speakernet.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_extractor_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=cluster_threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        raise RuntimeError("Invalid diarization configuration — check model paths.")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


# ------------------------------
# 🎙️ Whisper Prime ASR
# ------------------------------
def load_whisper_pipeline():
    print("\n📦 Loading OriServe Whisper-Hindi2Hinglish-Prime...")
    model_id = "/Users/gauravjain/Downloads/prime"
    device = 0 if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={
            "task": "transcribe",
            "language": "en"
        }
    )

    print(f"✅ Whisper loaded on device: {device}")
    return pipe


# ------------------------------
# ⏱️ Utility: Format Time
# ------------------------------
def sec_to_srt_time(sec):
    hr = int(sec // 3600)
    sec %= 3600
    mn = int(sec // 60)
    sec %= 60
    sec_i = int(sec)
    ms = int((sec - sec_i) * 1000)
    return f"{hr:02}:{mn:02}:{sec_i:02},{ms:03}"


# ------------------------------
# 🧩 Merge Chunk Diarization Results
# ------------------------------
def merge_diarization_results(results_per_chunk, durations):
    merged = []
    offset = 0.0

    for res, dur in zip(results_per_chunk, durations):
        for seg in res:
            merged.append({
                "start": seg.start + offset,
                "end": seg.end + offset,
                "speaker": seg.speaker
            })
        offset += dur

    merged.sort(key=lambda x: x["start"])
    print(f"✅ Merged {len(results_per_chunk)} chunks → {len(merged)} total segments")
    return merged


# ------------------------------
# ✍️ Generate SRT
# ------------------------------
def generate_srt(diarization_segments, audio, sr, pipe, output_srt="output.srt"):
    srt_lines, idx = [], 1
    print("\n🗣️ Starting transcription for diarized segments...")

    for i, seg in enumerate(tqdm(diarization_segments, desc="Transcribing", unit="segment")):
        start, end, speaker = seg["start"], seg["end"], f"speaker_{seg['speaker']:02}"
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]

        try:
            text = pipe(segment_audio)["text"].strip()
        except Exception as e:
            text = f"[ERROR: {e}]"

        srt_lines.append(f"{idx}\n{sec_to_srt_time(start)} --> {sec_to_srt_time(end)}\n{speaker}: {text}\n")
        idx += 1

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"\n✅ SRT saved → {output_srt}")


# ------------------------------
# 🚀 Main Pipeline
# ------------------------------
def main():
    wave_filename = "/Users/gauravjain/Downloads/testing-rec.wav"
    if not Path(wave_filename).is_file():
        raise FileNotFoundError(f"{wave_filename} not found")

    print("🎧 Loading audio...")
    audio, sr = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    # Diarization setup
    print("\n🔊 Initializing speaker diarization...")
    sd = init_speaker_diarization(num_speakers=2)
    audio, sr = resample_audio(audio, sr, sd.sample_rate)

    # Split long audio
    chunks = split_audio(audio, sr, chunk_duration_sec=300)  # 5 min chunks
    durations = [len(c) / sr for c in chunks]
    print(f"🧩 Audio split into {len(chunks)} chunks")

    # Process each chunk
    all_results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\n🕵️ Diarizing chunk {i}/{len(chunks)} ({len(chunk)/sr:.1f}s)...")
        res = sd.process(chunk).sort_by_start_time()
        all_results.append(res)

    # Merge results
    merged_segments = merge_diarization_results(all_results, durations)

    # Load Whisper
    pipe = load_whisper_pipeline()

    # Transcribe & save
    generate_srt(merged_segments, audio, sr, pipe, output_srt="testing-rec-1.srt")


if __name__ == "__main__":
    main()
