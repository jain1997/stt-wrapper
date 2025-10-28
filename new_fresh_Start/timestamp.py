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
# Thread-safety / environment for Sherpa
# ------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

# ------------------------------
# Audio utilities
# ------------------------------
def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int):
    if sample_rate != target_sample_rate:
        logger.info(f"Resampling audio from {sample_rate} Hz → {target_sample_rate} Hz")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
        logger.info(f"Resampled: new shape {audio.shape}, sr {sample_rate}")
    return audio, sample_rate

def split_audio(audio: np.ndarray, sr: int, chunk_duration_sec: int = 300):
    samples_per_chunk = int(chunk_duration_sec * sr)
    chunks = [ audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk) ]
    logger.info(f"Split audio into {len(chunks)} chunks of up to {chunk_duration_sec} seconds each")
    return chunks

# ------------------------------
# Diarization initialization
# ------------------------------
def init_speaker_diarization(num_speakers: int = -1, cluster_threshold: float = 0.5):
    logger.info("Initializing speaker diarization model")
    segmentation_model = "/Users/gauravjain/Documents/work/exported-assets/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_model    = "/Users/gauravjain/Downloads/nemo_en_speakerverification_speakernet.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model),
        ),
        embedding    = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_model),
        clustering   = sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=cluster_threshold),
        min_duration_on  = 0.3,
        min_duration_off = 0.5,
    )

    if not config.validate():
        logger.error("Invalid diarization configuration – please check model paths")
        raise RuntimeError("Invalid diarization configuration")

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    logger.info("Speaker diarization model initialized")
    return sd

# ------------------------------
# ASR pipeline (with word-level timestamps support)
# ------------------------------
def load_asr_pipeline(model_id: str, device: str = None):
    device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading ASR model {model_id} on device {device}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    # Patch alignment_heads for word-level timestamps
    cfg = model.generation_config.to_dict()
    cfg["alignment_heads"] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 16], [3, 17], [3, 18], [3, 19], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 16], [4, 17], [4, 18], [4, 19], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [9, 16], [9, 17], [9, 18], [9, 19], [10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 17], [10, 18], [10, 19], [11, 0], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19], [12, 0], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [12, 16], [12, 17], [12, 18], [12, 19], [13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [13, 16], [13, 17], [13, 18], [13, 19], [14, 0], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [14, 16], [14, 17], [14, 18], [14, 19], [15, 0], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [15, 14], [15, 15], [15, 16], [15, 17], [15, 18], [15, 19], [16, 0], [16, 1], [16, 2], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13], [16, 14], [16, 15], [16, 16], [16, 17], [16, 18], [16, 19], [17, 0], [17, 1], [17, 2], [17, 3], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8], [17, 9], [17, 10], [17, 11], [17, 12], [17, 13], [17, 14], [17, 15], [17, 16], [17, 17], [17, 18], [17, 19], [18, 0], [18, 1], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [18, 9], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [18, 15], [18, 16], [18, 17], [18, 18], [18, 19], [19, 0], [19, 1], [19, 2], [19, 3], [19, 4], [19, 5], [19, 6], [19, 7], [19, 8], [19, 9], [19, 10], [19, 11], [19, 12], [19, 13], [19, 14], [19, 15], [19, 16], [19, 17], [19, 18], [19, 19], [20, 0], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 18], [20, 19], [21, 0], [21, 1], [21, 2], [21, 3], [21, 4], [21, 5], [21, 6], [21, 7], [21, 8], [21, 9], [21, 10], [21, 11], [21, 12], [21, 13], [21, 14], [21, 15], [21, 16], [21, 17], [21, 18], [21, 19], [22, 0], [22, 1], [22, 2], [22, 3], [22, 4], [22, 5], [22, 6], [22, 7], [22, 8], [22, 9], [22, 10], [22, 11], [22, 12], [22, 13], [22, 14], [22, 15], [22, 16], [22, 17], [22, 18], [22, 19], [23, 0], [23, 1], [23, 2], [23, 3], [23, 4], [23, 5], [23, 6], [23, 7], [23, 8], [23, 9], [23, 10], [23, 11], [23, 12], [23, 13], [23, 14], [23, 15], [23, 16], [23, 17], [23, 18], [23, 19], [24, 0], [24, 1], [24, 2], [24, 3], [24, 4], [24, 5], [24, 6], [24, 7], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [24, 13], [24, 14], [24, 15], [24, 16], [24, 17], [24, 18], [24, 19], [25, 0], [25, 1], [25, 2], [25, 3], [25, 4], [25, 5], [25, 6], [25, 7], [25, 8], [25, 9], [25, 10], [25, 11], [25, 12], [25, 13], [25, 14], [25, 15], [25, 16], [25, 17], [25, 18], [25, 19], [26, 0], [26, 1], [26, 2], [26, 3], [26, 4], [26, 5], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [26, 15], [26, 16], [26, 17], [26, 18], [26, 19], [27, 0], [27, 1], [27, 2], [27, 3], [27, 4], [27, 5], [27, 6], [27, 7], [27, 8], [27, 9], [27, 10], [27, 11], [27, 12], [27, 13], [27, 14], [27, 15], [27, 16], [27, 17], [27, 18], [27, 19], [28, 0], [28, 1], [28, 2], [28, 3], [28, 4], [28, 5], [28, 6], [28, 7], [28, 8], [28, 9], [28, 10], [28, 11], [28, 12], [28, 13], [28, 14], [28, 15], [28, 16], [28, 17], [28, 18], [28, 19], [29, 0], [29, 1], [29, 2], [29, 3], [29, 4], [29, 5], [29, 6], [29, 7], [29, 8], [29, 9], [29, 10], [29, 11], [29, 12], [29, 13], [29, 14], [29, 15], [29, 16], [29, 17], [29, 18], [29, 19], [30, 0], [30, 1], [30, 2], [30, 3], [30, 4], [30, 5], [30, 6], [30, 7], [30, 8], [30, 9], [30, 10], [30, 11], [30, 12], [30, 13], [30, 14], [30, 15], [30, 16], [30, 17], [30, 18], [30, 19], [31, 0], [31, 1], [31, 2], [31, 3], [31, 4], [31, 5], [31, 6], [31, 7], [31, 8], [31, 9], [31, 10], [31, 11], [31, 12], [31, 13], [31, 14], [31, 15], [31, 16], [31, 17], [31, 18], [31, 19]]
    model.generation_config = GenerationConfig.from_dict(cfg)
    logger.info("Patched generation_config with alignment_heads for word-level timestamps")

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    logger.info("ASR pipeline ready")
    return pipe

# ------------------------------
# Align ASR word timestamps with speaker segments
# ------------------------------
def align_asr_words_to_speakers(asr_words: list, diarization_segments: list):
    logger.info(f"Aligning {len(asr_words)} ASR words to {len(diarization_segments)} speaker segments")
    merged = []
    for w in asr_words:
        assigned = "unknown"
        for seg in diarization_segments:
            if w["start"] >= seg["start"] and w["end"] <= seg["end"]:
                assigned = seg["speaker"]
                break
            if (w["start"] < seg["end"] and w["end"] > seg["start"]):
                assigned = seg["speaker"]
                break
        merged.append({
            "text":    w["text"],
            "start":   w["start"],
            "end":     w["end"],
            "speaker": assigned
        })
    logger.info("Alignment complete")
    return merged

# ------------------------------
# Group consecutive words by same speaker → utterances
# ------------------------------
def group_words_to_utterances(merged_words: list, max_pause: float = 1.0):
    """
    Group words into utterances: consecutive words from same speaker, and if pause > max_pause seconds then break utterance.
    Returns a list of utterances: { speaker, start, end, text }
    """
    logger.info("Grouping words into speaker utterances")
    utterances = []
    if not merged_words:
        return utterances

    current = {
        "speaker": merged_words[0]["speaker"],
        "start":   merged_words[0]["start"],
        "end":     merged_words[0]["end"],
        "text":    merged_words[0]["text"]
    }

    for w in merged_words[1:]:
        if w["speaker"] == current["speaker"] and (w["start"] - current["end"] <= max_pause):
            # continue the utterance
            current["end"]   = w["end"]
            current["text"] += " " + w["text"]
        else:
            # finalize current utterance
            utterances.append(current)
            # start new
            current = {
                "speaker": w["speaker"],
                "start":   w["start"],
                "end":     w["end"],
                "text":    w["text"]
            }
    # append the final one
    utterances.append(current)
    logger.info(f"Grouped into {len(utterances)} utterances")
    return utterances

# ------------------------------
# Format time to SRT time string
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
# Write SRT and TXT
# ------------------------------
def write_utterances_to_srt(utterances: list, srt_path: str):
    logger.info(f"Writing utterances to SRT file: {srt_path}")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, u in enumerate(utterances, start=1):
            f.write(f"{idx}\n")
            f.write(f"{sec_to_srt_time(u['start'])} --> {sec_to_srt_time(u['end'])}\n")
            f.write(f"{u['speaker']}: {u['text']}\n\n")
    logger.info("SRT writing complete")

def write_utterances_to_txt(utterances: list, txt_path: str):
    logger.info(f"Writing utterances to TXT file: {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for u in utterances:
            f.write(f"[{u['speaker']}] {sec_to_srt_time(u['start'])}-{sec_to_srt_time(u['end'])} {u['text']}\n")
    logger.info("TXT writing complete")

# ------------------------------
# Main pipeline
# ------------------------------
def main():
    audio_file = "/Users/gauravjain/Documents/work/stt-wrapper/test_audio/speaker_diarization.wav"
    if not Path(audio_file).is_file():
        logger.error(f"Audio file not found: {audio_file}")
        raise FileNotFoundError(f"{audio_file} not found")

    logger.info(f"Loading audio file {audio_file}")
    audio_all, sr = sf.read(audio_file, dtype="float32", always_2d=True)
    audio_all = audio_all[:, 0]  # convert to mono if stereo

    # Diarization
    sd = init_speaker_diarization(num_speakers=2)
    audio_resampled, sr = resample_audio(audio_all, sr, sd.sample_rate)
    chunks = split_audio(audio_resampled, sr, chunk_duration_sec=300)
    durations = [len(c)/sr for c in chunks]

    logger.info("Running diarization on chunks")
    diarization_segments = []
    offset = 0.0
    for idx, chunk in enumerate(tqdm(chunks, desc="Diarization chunks", unit="chunk")):
        result = sd.process(chunk).sort_by_start_time()
        for seg in result:
            diarization_segments.append({
                "start":   seg.start + offset,
                "end":     seg.end   + offset,
                "speaker": f"speaker_{seg.speaker:02}"
            })
        offset += durations[idx]
    logger.info(f"Completed diarization: total segments {len(diarization_segments)}")

    # ASR
    model_id = "/Users/gauravjain/Downloads/prime/"
    asr_pipe = load_asr_pipeline(model_id)
    logger.info("Running ASR with word-level timestamps")
    asr_result = asr_pipe(
        audio_file,
        return_timestamps="word",
        chunk_length_s=30,
        stride_length_s=5
    )
    logger.info("ASR complete")

    # Extract words with timestamps
    asr_words = []
    if "words" in asr_result:
        for w in asr_result["words"]:
            asr_words.append({"text": w["text"], "start": w["start"], "end": w["end"]})
    elif "chunks" in asr_result:
        for c in asr_result["chunks"]:
            asr_words.append({"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]})
    else:
        logger.error("No word-level timestamps found in ASR result")
        raise RuntimeError("ASR result lacks word timestamps")

    # Align to speakers
    merged_words = align_asr_words_to_speakers(asr_words, diarization_segments)

    # Group into utterances
    utterances = group_words_to_utterances(merged_words, max_pause=1.0)

    # Write outputs
    write_utterances_to_srt(utterances, "output_aligned.srt")
    write_utterances_to_txt(utterances, "output_aligned.txt")

    logger.info("Pipeline complete. Outputs: output_aligned.srt, output_aligned.txt")

if __name__ == "__main__":
    main()
