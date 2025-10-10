import os
import re
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import argparse


# ---------- TIMESTAMP UTILS ----------
def timestamp_to_sec(ts):
    """Convert 'HH:MM:SS,mmm' → seconds (float)."""
    h, m, s_ms = ts.strip().split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


# ---------- SRT PARSER ----------
def parse_srt(srt_path):
    """Robust SRT parser — handles variable spacing and multiple lines."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            times = lines[1].strip()
            match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', times)
            if not match:
                continue
            start, end = match.groups()
            # Speaker and text may be on one or more lines
            text_lines = lines[2:]
            joined_text = ' '.join(text_lines).strip()
            speaker, text = joined_text.split(':', 1) if ':' in joined_text else ('SPEAKER_00', joined_text)
            segments.append((index, start, end, speaker.strip(), text.strip()))

    return segments


# ---------- LOAD LOCAL MODEL (with fallback) ----------
def load_local_model(local_model_dir, device):
    """Load Whisper Prime model from local path with fallback to CPU if GPU fails."""
    try:
        processor = AutoProcessor.from_pretrained(local_model_dir)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(local_model_dir).to(device)
        print(f"✅ Loaded model on {device}")
        return processor, model
    except RuntimeError as e:
        print(f"⚠️ Model load failed on {device}: {e}")
        print("➡️ Falling back to CPU...")
        processor = AutoProcessor.from_pretrained(local_model_dir)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(local_model_dir).to("cpu")
        return processor, model


# ---------- TRANSCRIBE SEGMENT ----------
def transcribe_segment(processor, model, y, sr, start_s, end_s, device):
    """Transcribe one segment using Whisper Prime with GPU→CPU fallback."""
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    segment = y[start_sample:end_sample]

    if len(segment) == 0:
        return ""

    inputs = processor(segment, sampling_rate=sr, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e):
            print("⚠️ GPU OOM during segment — retrying on CPU...")
            torch.cuda.empty_cache()
            model = model.to("cpu")
            inputs = inputs.to("cpu")
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
        else:
            raise e

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ---------- GENERATE TRANSLATED SRT ----------
def generate_new_srt(parsed_srt, y, sr, output_path, processor, model, device):
    new_lines = []
    for idx, start, end, speaker, text in tqdm(parsed_srt, desc="🎧 Translating segments", colour="cyan", leave=False):
        start_s = timestamp_to_sec(start)
        end_s = timestamp_to_sec(end)
        try:
            new_text = transcribe_segment(processor, model, y, sr, start_s, end_s, device)
        except Exception as e:
            print(f"⚠️ Error segment {idx}: {e}")
            new_text = text
        new_lines.append(f"{idx}\n{start} --> {end}\n{speaker}: {new_text}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    print(f"✅ Translated SRT saved at: {output_path}")


# ---------- MAIN PIPELINE ----------
def process_file(audio_path, srt_path, output_dir, local_model_dir, device):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.srt")

    print(f"\n🎬 Processing: {base_name}")
    print(f"📄 Audio: {audio_path}")
    print(f"🧾 SRT:   {srt_path}")

    # Load model
    processor, model = load_local_model(local_model_dir, device)

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Parse SRT
    parsed = parse_srt(srt_path)
    print(f"🔹 Found {len(parsed)} segments.")

    # Generate new SRT
    generate_new_srt(parsed, y, sr, output_path, processor, model, device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="Translate existing SRTs using local Whisper Prime model.")
    parser.add_argument("--audio_dir", required=True, help="Path to folder with input audio files")
    parser.add_argument("--srt_dir", required=True, help="Path to folder with corresponding SRT files")
    parser.add_argument("--output_dir", default="srt_output", help="Output folder for translated SRTs")
    parser.add_argument("--model_dir", required=True, help="Local folder containing Whisper Prime model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    # Safeguard: create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔧 Device: {args.device.upper()}")
    print(f"📂 Audio folder: {args.audio_dir}")
    print(f"📄 SRT folder: {args.srt_dir}")
    print(f"💾 Output folder: {args.output_dir}")

    # Match each audio file with its corresponding SRT
    audio_files = [
        f for f in os.listdir(args.audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]

    if not audio_files:
        print("⚠️ No audio files found.")
        return

    for filename in tqdm(audio_files, desc="Processing files", colour="green"):
        audio_path = os.path.join(args.audio_dir, filename)
        srt_name = os.path.splitext(filename)[0] + ".srt"
        srt_path = os.path.join(args.srt_dir, srt_name)

        if not os.path.exists(srt_path):
            print(f"⚠️ Missing SRT for {filename}, skipping.")
            continue

        process_file(audio_path, srt_path, args.output_dir, args.model_dir, args.device)

    print("\n✅ All files processed successfully!")


if __name__ == "__main__":
    main()
