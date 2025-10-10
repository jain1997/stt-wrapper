import os
import torch
import torchaudio
import whisper
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from deepmultilingualpunctuation import PunctuationModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import tempfile
import argparse


# ---------- TRANSCRIPTION ----------
def transcribe_audio(audio_path, model_name, device, language="hi"):
    print(f"\n🎙️ Transcribing: {os.path.basename(audio_path)} using Whisper-{model_name} [{language}]")
    model = whisper.load_model(model_name, device=device)
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            condition_on_previous_text=False,
            verbose=False,
        )
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("⚠️ GPU OOM — retrying on CPU...")
            model = whisper.load_model(model_name, device="cpu")
            result = model.transcribe(audio_path, language=language, condition_on_previous_text=False)
        else:
            raise e
    return result["text"], result["segments"]



# ---------- PUNCTUATION ----------
def restore_punctuation(text):
    model = PunctuationModel(model="kredor/punctuate-all")
    words = text.split()
    labeled_words = model.predict(words, chunk_size=230)
    final = []
    for word, label in zip(words, labeled_words):
        _, punct, _ = label
        if punct != "0" and not word.endswith(punct):
            word += punct
        final.append(word)
    return " ".join(final)


# ---------- DIARIZATION ----------
def diarize_audio(audio_path, device, n_speakers=2, window=3.0, hop=1.0):
    print("🗣️ Running SpeechBrain diarization...")

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/speechbrain_encoder",
        run_opts={"device": device},
    )

    window_samples = int(window * 16000)
    hop_samples = int(hop * 16000)
    embeddings, timestamps = [], []

    total = len(waveform[0]) - window_samples + 1
    if total <= 0:
        print("⚠️ Audio too short for diarization window.")
        return [(0, len(waveform[0]) / 16, 0)]

    for start_sample in tqdm(range(0, total, hop_samples), desc="🔍 Extracting embeddings", leave=False):
        end_sample = start_sample + window_samples
        window_seg = waveform[:, start_sample:end_sample]
        emb = encoder.encode_batch(window_seg)
        embeddings.append(emb.squeeze().cpu().numpy())
        timestamps.append((start_sample / 16000, end_sample / 16000))

    embeddings = np.array(embeddings)
    affinity = cosine_similarity(embeddings)

    from speechbrain.processing.diarization import spectral_clustering_sb
    labels = spectral_clustering_sb(affinity, n_clusters=n_speakers)

    diarization = [(int(s * 1000), int(e * 1000), int(spk)) for (s, e), spk in zip(timestamps, labels)]

    # Merge nearby same-speaker segments
    merged = []
    if diarization:
        cur_s, cur_e, cur_spk = diarization[0]
        for s, e, spk in diarization[1:]:
            if spk == cur_spk and s <= cur_e + 1000:
                cur_e = e
            else:
                merged.append((cur_s, cur_e, cur_spk))
                cur_s, cur_e, cur_spk = s, e, spk
        merged.append((cur_s, cur_e, cur_spk))
    return merged


# ---------- MERGE ----------
def merge_segments(whisper_segments, diarization_labels):
    merged = []
    cur_spk, cur_text, cur_start, cur_end = None, "", 0, 0
    for seg in whisper_segments:
        start, end, text = seg["start"], seg["end"], seg["text"].strip()
        spk = 0
        for s, e, label in diarization_labels:
            if start * 1000 >= s and end * 1000 <= e:
                spk = label
                break
        if spk == cur_spk:
            cur_text += " " + text
            cur_end = end
        else:
            if cur_text:
                merged.append({"speaker": cur_spk, "start": cur_start, "end": cur_end, "text": cur_text.strip()})
            cur_spk, cur_text, cur_start, cur_end = spk, text, start, end
    if cur_text:
        merged.append({"speaker": cur_spk, "start": cur_start, "end": cur_end, "text": cur_text.strip()})
    return merged


# ---------- WRITE SRT ----------
def write_srt(segments, output_path):
    def fmt(t):
        h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\nSPEAKER_{seg['speaker']:02d}: {seg['text']}\n\n")


# ---------- PROCESS ONE FILE ----------
def process_audio_file(audio_path, output_dir, model_name, device, n_speakers):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.srt")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = os.path.join(tmpdir, "temp.wav")
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            torchaudio.save(tmp_wav, waveform, 16000)

            # Step 1: Transcribe
            text, whisper_segments = transcribe_audio(tmp_wav, model_name, device)

            # Step 2: Diarize
            diar_labels = diarize_audio(tmp_wav, device, n_speakers=n_speakers)

            # Step 3: Merge
            merged = merge_segments(whisper_segments, diar_labels)

            # Step 4: Save
            write_srt(merged, output_path)
            print(f"✅ Saved SRT: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="Generate first-level SRTs using Whisper + SpeechBrain")
    parser.add_argument("--input_dir", required=True, help="Path to folder with audio files")
    parser.add_argument("--output_dir", default="first_timestamp_srt", help="Where to save output SRTs")
    parser.add_argument("--model", default="small", help="Whisper model (tiny, base, small, medium)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--n_speakers", type=int, default=2, help="Expected number of speakers")
    args = parser.parse_args()

    # Ensure output folder exists before processing
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔧 Using device: {args.device.upper()}")
    print(f"📂 Input folder: {args.input_dir}")
    print(f"💾 Output folder: {args.output_dir}")

    audio_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]

    if not audio_files:
        print("⚠️ No audio files found in input folder.")
        return

    for path in tqdm(audio_files, desc="Processing files", colour="green"):
        process_audio_file(path, args.output_dir, args.model, args.device, args.n_speakers)

    print("\n✅ All files processed successfully.")


if __name__ == "__main__":
    main()
