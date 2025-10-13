import os
from pathlib import Path
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
import sherpa_onnx
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

# ------------------------------
# Step 1: Resample audio
# ------------------------------
def resample_audio(audio, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        print(f"\n🔄 Resampling {sample_rate} Hz → {target_sample_rate} Hz...")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        print(f"✅ Resampled. New shape: {audio.shape}")
    return audio, target_sample_rate


# ------------------------------
# Step 2: Init diarization
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
# Step 3: Load Whisper Prime via pipeline
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
# Step 4: Utilities
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
# Step 5: Generate continuous SRT
# ------------------------------
def generate_srt(diarization_segments, audio, sr, pipe, output_srt="output.srt"):
    srt_lines, idx = [], 1
    print("\n🗣️ Starting transcription for diarized segments...")

    diarization_segments = sorted(diarization_segments, key=lambda x: x.start)

    for i, seg in enumerate(tqdm(diarization_segments, desc="Transcribing", unit="segment")):
        start, end = seg.start, seg.end
        speaker = f"speaker_{seg.speaker:02}"

        # Make timestamps continuous (end of previous = start of current)
        if i > 0:
            prev_end = diarization_segments[i - 1].end
            if start > prev_end:
                start = prev_end

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]

        # Transcribe with pipeline
        text = pipe(segment_audio, batch_size=1)["text"].strip()

        # Format SRT entry
        start_ts = sec_to_srt_time(start)
        end_ts = sec_to_srt_time(end)
        srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{speaker}: {text}\n")
        idx += 1

    # Save SRT
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"\n✅ SRT saved → {output_srt}")


# ------------------------------
# Step 6: Main pipeline
# ------------------------------
def main():
    wave_filename = "/Users/gauravjain/Downloads/testing-rec.wav"
    if not Path(wave_filename).is_file():
        raise FileNotFoundError(f"{wave_filename} not found")

    print("🎧 Loading audio...")
    audio, sr = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    # Diarization
    print("\n🔊 Initializing speaker diarization...")
    sd = init_speaker_diarization(num_speakers=2)
    audio, sr = resample_audio(audio, sr, sd.sample_rate)

    print("\n🕵️ Running speaker diarization...")
    num_chunks = 100
    progress_bar = tqdm(total=num_chunks, desc="Diarization", unit="chunk")

    def progress_callback(n, total):
        progress_bar.n = int((n / total) * num_chunks)
        progress_bar.refresh()
        return 0

    result = sd.process(audio, callback=progress_callback).sort_by_start_time()
    progress_bar.close()
    print(f"✅ Diarization complete ({len(result)} segments).")

    # Whisper pipeline
    pipe = load_whisper_pipeline()

    # Transcribe + SRT
    generate_srt(result, audio, sd.sample_rate, pipe, output_srt="testing-rec.srt")


if __name__ == "__main__":
    main()
