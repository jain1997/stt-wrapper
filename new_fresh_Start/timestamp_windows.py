import os
import csv
import logging
from pathlib import Path

import torch
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

import sherpa_onnx
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GenerationConfig


# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ASR_Diarization_Pipeline")


# ------------------------------
# Environment configuration (important for Windows CUDA)
# ------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:64"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ------------------------------
# Audio utilities
# ------------------------------
def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int):
    if sample_rate != target_sample_rate:
        logger.info(f"Resampling audio from {sample_rate} Hz → {target_sample_rate} Hz")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    return audio, sample_rate


def split_audio(audio: np.ndarray, sr: int, chunk_duration_sec: int = 300):
    samples_per_chunk = int(chunk_duration_sec * sr)
    chunks = [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]
    logger.info(f"Split audio into {len(chunks)} chunks of up to {chunk_duration_sec}s each")
    return chunks


# ------------------------------
# Diarization initialization
# ------------------------------
def init_speaker_diarization(num_speakers: int = -1, cluster_threshold: float = 0.5):
    logger.info("Initializing speaker diarization model")
    segmentation_model = "/path/to/segmentation_model.onnx"
    embedding_model = "/path/to/embedding_model.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=cluster_threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        logger.error("Invalid diarization configuration – check model paths")
        raise RuntimeError("Invalid diarization configuration")

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    logger.info("Speaker diarization model initialized")
    return sd


# ------------------------------
# ASR pipeline (with safe alignment_heads)
# ------------------------------
def load_asr_pipeline(model_id: str):
    logger.info(f"Loading ASR model {model_id}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    # Add alignment_heads
    cfg = model.generation_config.to_dict()
    cfg["alignment_heads"] = [
        [13, 10], [13, 11], [14, 10], [14, 11],
        [15, 10], [15, 11], [16, 10], [16, 11],
        [17, 10], [17, 11]
    ]
    model.generation_config = GenerationConfig.from_dict(cfg)
    logger.info("Patched generation_config with alignment_heads for word-level timestamps")

    # Safe fallback for Windows
    import platform
    if platform.system() == "Windows" and torch.cuda.is_available():
        logger.warning("⚠️  Windows detected: switching timestamp extraction to CPU to avoid CUDA crash")
        model = model.to("cpu")
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if torch.cuda.is_available() and model.device.type == "cuda" else -1,
        generate_kwargs={
            "task": "transcribe",
            "language": "en"
        }
    )

    logger.info(f"ASR pipeline ready on device: {model.device}")
    return pipe


# ------------------------------
# Format time helper
# ------------------------------
def sec_to_srt_time(sec: float):
    hr = int(sec // 3600)
    sec %= 3600
    mn = int(sec // 60)
    sec %= 60
    sec_i = int(sec)
    ms = int((sec - sec_i) * 1000)
    return f"{hr:02}:{mn:02}:{sec_i:02},{ms:03}"


# ------------------------------
# Group consecutive words → utterances
# ------------------------------
def group_words_to_utterances(merged_words: list, max_pause: float = 1.0):
    utterances = []
    if not merged_words:
        return utterances

    current = {
        "speaker": merged_words[0]["speaker"],
        "start": merged_words[0]["start"],
        "end": merged_words[0]["end"],
        "text": merged_words[0]["text"]
    }

    for w in merged_words[1:]:
        if w["speaker"] == current["speaker"] and (w["start"] - current["end"] <= max_pause):
            current["end"] = w["end"]
            current["text"] += " " + w["text"]
        else:
            utterances.append(current)
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["text"]
            }

    utterances.append(current)
    return utterances


# ------------------------------
# Write SRT and TXT
# ------------------------------
def write_srt(utterances, filename="output_aligned.srt"):
    logger.info(f"Writing SRT to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        for i, utt in enumerate(utterances, 1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_time(utt['start'])} --> {sec_to_srt_time(utt['end'])}\n")
            f.write(f"{utt['speaker']}: {utt['text']}\n\n")
    logger.info("SRT writing complete")


def write_txt(utterances, filename="output_aligned.txt"):
    logger.info(f"Writing TXT to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        for utt in utterances:
            f.write(f"[{utt['speaker']}] {sec_to_srt_time(utt['start'])}-{sec_to_srt_time(utt['end'])} {utt['text']}\n")
    logger.info("TXT writing complete")


# ------------------------------
# Main pipeline
# ------------------------------
def main():
    audio_file = "/path/to/your_audio.wav"
    if not Path(audio_file).is_file():
        raise FileNotFoundError(f"{audio_file} not found")

    logger.info(f"Loading audio file {audio_file}")
    audio_all, sr = sf.read(audio_file, dtype="float32", always_2d=True)
    audio_all = audio_all[:, 0]

    # Diarization
    sd = init_speaker_diarization(num_speakers=2)
    audio_resampled, sr = resample_audio(audio_all, sr, sd.sample_rate)
    chunks = split_audio(audio_resampled, sr, chunk_duration_sec=300)
    durations = [len(c) / sr for c in chunks]

    diarization_segments = []
    offset = 0.0
    for idx, chunk in enumerate(tqdm(chunks, desc="🔊 Diarization", unit="chunk")):
        result = sd.process(chunk).sort_by_start_time()
        for seg in result:
            diarization_segments.append({
                "start": seg.start + offset,
                "end": seg.end + offset,
                "speaker": f"speaker_{seg.speaker:02}"
            })
        offset += durations[idx]

    logger.info(f"Diarization complete – total segments: {len(diarization_segments)}")

    # ASR
    model_id = "Oriserve/Whisper-Hindi2Hinglish-Prime"
    asr_pipe = load_asr_pipeline(model_id)

    asr_chunks = split_audio(audio_all, sr, chunk_duration_sec=30)
    logger.info(f"Transcribing {len(asr_chunks)} chunks with progress bar...")

    all_asr_words = []
    for i, chunk in enumerate(tqdm(asr_chunks, desc="🗣️ ASR Progress", unit="chunk")):
        temp_wav = f"temp_chunk_{i}.wav"
        sf.write(temp_wav, chunk, sr)
        result = asr_pipe(temp_wav, return_timestamps="word", chunk_length_s=10, stride_length_s=2)
        if "words" in result:
            for w in result["words"]:
                all_asr_words.append({"text": w["text"], "start": w["start"], "end": w["end"]})
        elif "chunks" in result:
            for c in result["chunks"]:
                all_asr_words.append({"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]})
        os.remove(temp_wav)

    logger.info("ASR complete")

    # Align words to speakers
    merged = []
    for w in all_asr_words:
        assigned = "unknown"
        for seg in diarization_segments:
            if w["start"] >= seg["start"] and w["end"] <= seg["end"]:
                assigned = seg["speaker"]
                break
            if (w["start"] < seg["end"] and w["end"] > seg["start"]):
                assigned = seg["speaker"]
                break
        merged.append({
            "speaker": assigned,
            "start": w["start"],
            "end": w["end"],
            "text": w["text"]
        })

    # CSV
    csv_path = "aligned_words_speakers.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "start", "end", "text"])
        for w in merged:
            writer.writerow([w["speaker"], f"{w['start']:.2f}", f"{w['end']:.2f}", w["text"]])
    logger.info(f"CSV written → {csv_path}")

    # Group → SRT + TXT
    utterances = group_words_to_utterances(merged, max_pause=1.0)
    write_srt(utterances, "output_aligned.srt")
    write_txt(utterances, "output_aligned.txt")

    logger.info("✅ Pipeline complete → SRT + TXT + CSV generated")


if __name__ == "__main__":
    main()
