import os
from pathlib import Path
import soundfile as sf
import librosa
import torch
import numpy as np
from tqdm import tqdm
import sherpa_onnx
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# -------------------------------------------------------------------
# 🧠 THREAD SAFETY (Prevents Sherpa from hanging on Windows)
# -------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


# -------------------------------------------------------------------
# 🧩 Utility Functions
# -------------------------------------------------------------------
def sec_to_srt_time(sec: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hr = int(sec // 3600)
    sec %= 3600
    mn = int(sec // 60)
    sec %= 60
    sec_i = int(sec)
    ms = int((sec - sec_i) * 1000)
    return f"{hr:02}:{mn:02}:{sec_i:02},{ms:03}"


def resample_audio(audio, sr, target_sr):
    """Ensure correct sample rate"""
    if sr != target_sr:
        print(f"🔄 Resampling {sr}Hz → {target_sr}Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"✅ Resampled. Shape: {audio.shape}")
    return audio.astype("float32", copy=False), target_sr


def split_audio(audio, sr, max_duration_sec=600):
    """Split long audio into smaller chunks (default 10 min each)"""
    samples_per_chunk = int(max_duration_sec * sr)
    return [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]


# -------------------------------------------------------------------
# 🧩 Sherpa Diarization Setup
# -------------------------------------------------------------------
def init_diarization(num_speakers=2, threshold=0.5):
    segmentation_model = "/Users/gauravjain/Documents/work/exported-assets/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_extractor_model = "/Users/gauravjain/Downloads/nemo_en_speakerverification_speakernet.onnx"

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_extractor_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.1,
        min_duration_off=0.1,
    )

    if not cfg.validate():
        raise RuntimeError("❌ Invalid diarization configuration or missing model file.")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


# -------------------------------------------------------------------
# 🧩 Whisper Pipeline Setup
# -------------------------------------------------------------------
def load_whisper_pipeline():
    print("\n📦 Loading OriServe Whisper-Hindi2Hinglish-Prime...")
    model_id = "/Users/gauravjain/Downloads/prime"
    device = 0 if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        generate_kwargs={"task": "transcribe", "language": "en"},
    )
    print(f"✅ Whisper loaded on device: {device}")
    return pipe


# -------------------------------------------------------------------
# 🧩 Merge Multi-Chunk Diarization Results (Python Safe)
# -------------------------------------------------------------------
def merge_results(all_results, sr, chunk_durations):
    """
    Merge diarization segments from multiple chunks into one continuous timeline.
    Returns a list of dicts: [{'start': float, 'end': float, 'speaker': int}, ...]
    """
    merged = []
    offset = 0.0

    for res, dur in zip(all_results, chunk_durations):
        for seg in res:
            merged.append({
                "start": seg.start + offset,
                "end": seg.end + offset,
                "speaker": seg.speaker
            })
        offset += dur

    merged.sort(key=lambda x: x["start"])
    return merged


# -------------------------------------------------------------------
# 🧩 Generate SRT Subtitles
# -------------------------------------------------------------------
def generate_srt(segments, audio, sr, pipe, output_path="output.srt"):
    srt_lines = []
    idx = 1
    print("\n🗣️ Transcribing diarized segments...")

    for seg in tqdm(segments, desc="Transcribing", unit="segment"):
        start, end, spk = seg["start"], seg["end"], seg["speaker"]
        speaker = f"speaker_{spk:02}"

        start_samp, end_samp = int(start * sr), int(end * sr)
        clip = audio[start_samp:end_samp]

        # Run Whisper transcription
        text = pipe(clip)["text"].strip()
        srt_lines.append(f"{idx}\n{sec_to_srt_time(start)} --> {sec_to_srt_time(end)}\n{speaker}: {text}\n")
        idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"\n✅ Final SRT saved → {output_path}")


# -------------------------------------------------------------------
# 🧩 Main Pipeline
# -------------------------------------------------------------------
def main():
    wav_file = "/Users/gauravjain/Downloads/testing-rec.wav"
    if not Path(wav_file).exists():
        raise FileNotFoundError(f"{wav_file} not found!")

    print("🎧 Loading audio...")
    audio, sr = sf.read(wav_file, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    # Initialize Sherpa
    print("\n🔊 Initializing speaker diarization...")
    sd = init_diarization(num_speakers=2)

    # Resample
    audio, sr = resample_audio(audio, sr, sd.sample_rate)

    # Split if long
    chunks = split_audio(audio, sr, max_duration_sec=600)
    print(f"🧩 Total chunks: {len(chunks)}")

    all_results = []
    durations = [len(c) / sr for c in chunks]

    # Process each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"\n🕵️ Processing chunk {i}/{len(chunks)} ({len(chunk)/sr:.1f}s)...")
        result = sd.process(chunk)
        all_results.append(result.sort_by_start_time())

    # Merge all results
    full_result = merge_results(all_results, sr, durations)
    print(f"✅ Diarization complete → {len(full_result)} total segments")

    # Load Whisper pipeline
    pipe = load_whisper_pipeline()

    # Transcribe and save SRT
    generate_srt(full_result, audio, sr, pipe, output_path="wind_testing-rec.srt")


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
