import os
import logging
from pathlib import Path
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
import re
import sherpa_onnx
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from difflib import SequenceMatcher


# ==============================================
# ⚙️ LOGGER CONFIGURATION
# ==============================================
logging.basicConfig(
    level=logging.DEBUG,  # use DEBUG to see every step
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 🧠 ENV + THREAD CONTROL
# ==============================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


# ==============================================
# 🔊 AUDIO UTILITIES
# ==============================================
def resample_audio(audio, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        logger.info(f"🔄 Resampling {sample_rate} Hz → {target_sample_rate} Hz...")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    return audio, target_sample_rate


def split_audio(audio, sr, chunk_duration_sec=300):
    samples_per_chunk = int(chunk_duration_sec * sr)
    logger.info(f"🔪 Splitting audio into {chunk_duration_sec}s chunks...")
    return [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]


# ==============================================
# 🧩 SHERPA DIARIZATION INITIALIZATION
# ==============================================
def init_speaker_diarization(num_speakers=-1, cluster_threshold=0.5):
    logger.info("🧩 Initializing speaker diarization models...")
    segmentation_model = "/Users/gauravjain/Documents/work/exported-assets/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_extractor_model = "/Users/gauravjain/Downloads/nemo_en_speakerverification_speakernet.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model)
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_extractor_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=cluster_threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        logger.error("❌ Invalid diarization configuration — check model paths.")
        raise RuntimeError("Invalid diarization configuration — check model paths.")
    logger.info("✅ Diarization config validated successfully.")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


# ==============================================
# 🗣️ WHISPER MODEL LOADING
# ==============================================
def load_whisper_pipeline():
    logger.info("📦 Loading OriServe Whisper-Hindi2Hinglish-Prime...")
    model_id = "/Users/gauravjain/Downloads/prime"
    device = 0 if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
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
    logger.info(f"✅ Whisper model loaded successfully on device: {device}")
    return pipe


# ==============================================
# ⏱️ TIME FORMATTER
# ==============================================
def sec_to_srt_time(sec):
    hr = int(sec // 3600)
    sec %= 3600
    mn = int(sec // 60)
    sec %= 60
    sec_i = int(sec)
    ms = int((sec - sec_i) * 1000)
    return f"{hr:02}:{mn:02}:{sec_i:02},{ms:03}"


# ==============================================
# 🧩 MERGE DIARIZATION RESULTS
# ==============================================
def merge_diarization_results(results_per_chunk, durations):
    logger.info("🔗 Merging diarization results from chunks...")
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
    logger.info(f"✅ Merged {len(results_per_chunk)} chunks → {len(merged)} total segments")
    return merged


# ==============================================
# 🧩 FINALIZE DIARIZATION SEGMENTS
# ==============================================
def finalize_diarization_segments(segments):
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x["start"])
    merged = [segments[0]]

    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"]:
            merged[-1]["end"] = max(last["end"], seg["end"])
        else:
            if seg["start"] > last["end"]:
                merged[-1]["end"] = seg["start"]
            merged.append(seg)
    logger.info(f"✅ Finalized {len(segments)} → {len(merged)} diarization segments (gapless merge)")
    return merged


# ==============================================
# 🧠 TEXT NORMALIZATION + DEDUPE (updated)
# ==============================================
_WORD_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]+")

def _tokenize_with_spans(s: str):
    """
    Tokenize into word-like tokens (Hindi + Latin + digits), returning spans:
    [{"text": original, "norm": lowercased, "start": i, "end": j}, ...]
    """
    out = []
    for m in _WORD_RE.finditer(s):
        tok = m.group(0)
        out.append({
            "text": tok,
            "norm": tok.lower(),
            "start": m.start(),
            "end": m.end()
        })
    return out

def _skip_trailing_punct_and_space(s: str, i: int) -> int:
    """Advance index i over spaces/punctuation so the cut starts cleanly."""
    while i < len(s) and not _WORD_RE.match(s[i]):
        i += 1
    return i

def dedupe_with_previous(prev_text: str, curr_text: str, window: int = 4) -> str:
    """
    Exact-window dedupe as requested:
    - Take last `window` words from prev_text (normalized).
    - Find same contiguous sequence in curr_text (normalized).
    - If found, cut everything up to (and including) that sequence in curr_text.
    - Else, use a light fuzzy overlap fallback; otherwise return curr_text.
    """
    if not prev_text or not curr_text:
        return (curr_text or "").strip()

    prev_tokens = _tokenize_with_spans(prev_text)
    curr_tokens = _tokenize_with_spans(curr_text)

    if len(prev_tokens) < window or len(curr_tokens) < window:
        return curr_text.strip()

    needle = [t["norm"] for t in prev_tokens[-window:]]
    hay = [t["norm"] for t in curr_tokens]

    cut_index_chars = None
    for i in range(0, len(hay) - window + 1):
        if hay[i:i + window] == needle:
            cut_index_chars = curr_tokens[i + window - 1]["end"]
            break

    if cut_index_chars is not None:
        cut_index_chars = _skip_trailing_punct_and_space(curr_text, cut_index_chars)
        trimmed = curr_text[cut_index_chars:].strip()
        logger.debug(
            f"🧩 Exact {window}-word overlap found; trimming. "
            f"Needle={' '.join(needle)} | Cut@{cut_index_chars} → {trimmed[:80]!r}"
        )
        return trimmed

    # Optional fallback: light fuzzy overlap for broader repetition
    prev_norm_tail = " ".join([t["norm"] for t in prev_tokens[-10:]])
    curr_norm_head = " ".join([t["norm"] for t in curr_tokens[:10]])
    sim = SequenceMatcher(None, prev_norm_tail, curr_norm_head).ratio()
    if sim > 0.65:
        logger.debug(f"🧩 Fuzzy overlap (sim={sim:.2f}) → trimming first 5 words as fallback")
        if len(curr_tokens) > 5:
            return curr_text[curr_tokens[5]["start"]:].strip()
        return curr_text.strip()

    return curr_text.strip()


# ==============================================
# ✍️ TRANSCRIPTION + EXPORT
# ==============================================
def generate_srt_and_txt(
    diarization_segments,
    audio,
    sr,
    pipe,
    output_srt="output.srt",
    output_txt="output.txt",
    overlap_sec=5.0,
    preroll_sec=0.5,
):
    logger.info(f"🗣️ Starting transcription with {overlap_sec}s context + {preroll_sec}s preroll...")

    srt_lines, txt_lines = [], []
    prev_text = ""

    for i, seg in enumerate(tqdm(diarization_segments, desc="Transcribing", unit="segment")):
        # Timestamp safety buffer
        if i > 0 and seg["start"] <= diarization_segments[i - 1]["end"]:
            seg["start"] = diarization_segments[i - 1]["end"] + 0.05

        seg_start = max(0.0, seg["start"] - preroll_sec)
        seg_end = seg["end"]
        speaker = f"SPEAKER_{seg['speaker']:02}"

        # Compute context overlap
        remaining_overlap = overlap_sec
        context_start = seg_start
        j = i - 1
        while remaining_overlap > 0 and j >= 0:
            prev_seg = diarization_segments[j]
            prev_start = prev_seg["start"]
            available = context_start - prev_start
            take = min(available, remaining_overlap)
            context_start -= take
            remaining_overlap -= take
            j -= 1
            if context_start <= 0:
                context_start = 0.0
                break

        start_sample = int(context_start * sr)
        end_sample = int(seg_end * sr)
        segment_audio = audio[start_sample:end_sample]

        try:
            result = pipe(segment_audio)
            raw_text = result["text"].strip()
            logger.debug(f"📝 Raw text before dedupe [{speaker}] → {raw_text}")
        except Exception as e:
            logger.error(f"[ERROR processing {speaker}] {e}")
            raw_text = f"[ERROR: {e}]"

        cleaned_text = dedupe_with_previous(prev_text, raw_text)
        prev_text = cleaned_text

        srt_lines.append(
            f"{i+1}\n{sec_to_srt_time(seg['start'])} --> {sec_to_srt_time(seg_end)}\n{speaker}: {cleaned_text}\n"
        )
        txt_lines.append(f"[{speaker}] {cleaned_text}")

        logger.debug(f"✅ Final text [{speaker}] → {cleaned_text}")

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    logger.info(f"✅ SRT saved → {output_srt}")
    logger.info(f"✅ TXT saved → {output_txt}")


# ==============================================
# 🚀 MAIN PIPELINE
# ==============================================
def main():
    wave_filename = "/Users/gauravjain/Documents/work/stt-wrapper/test_audio/speaker_diarization.wav"
    if not Path(wave_filename).is_file():
        logger.error(f"{wave_filename} not found")
        raise FileNotFoundError(f"{wave_filename} not found")

    logger.info("🎧 Loading audio...")
    audio, sr = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    logger.info("🔊 Initializing speaker diarization...")
    sd = init_speaker_diarization(num_speakers=2)
    audio, sr = resample_audio(audio, sr, sd.sample_rate)

    chunks = split_audio(audio, sr, chunk_duration_sec=300)
    durations = [len(c) / sr for c in chunks]
    logger.info(f"🧩 Audio split into {len(chunks)} chunks")

    all_results = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"🕵️ Diarizing chunk {i}/{len(chunks)} ({len(chunk)/sr:.1f}s)...")
        res = sd.process(chunk).sort_by_start_time()
        all_results.append(res)

    merged_segments = merge_diarization_results(all_results, durations)
    final_segments = finalize_diarization_segments(merged_segments)

    pipe = load_whisper_pipeline()
    generate_srt_and_txt(
        final_segments,
        audio,
        sr,
        pipe,
        output_srt="testing-rec-final.srt",
        output_txt="testing-rec-final.txt",
        overlap_sec=5.0,
        preroll_sec=0.5,
    )


if __name__ == "__main__":
    main()
