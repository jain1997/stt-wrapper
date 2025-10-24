"""Main diarization + transcription orchestration."""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .settings import AppConfig
from .audio_utils import load_audio, resample, split_audio
from .diarizer import Diarizer, merge_segments
from .transcriber import Transcriber
from .output_writer import write_srt, write_txt

log = logging.getLogger(__name__)


def process_file(audio_path: Path, cfg: AppConfig, out_dir: Path | None, overwrite: bool = False) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    out_dir = out_dir or audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / f"{audio_path.stem}{cfg.output_suffix}.srt"
    txt_path = out_dir / f"{audio_path.stem}{cfg.output_suffix}.txt"

    if srt_path.exists() and not overwrite:
        log.info("Skipping existing output: %s", srt_path)
        return

    log.info("Processing %s ...", audio_path.name)
    audio, sr = load_audio(audio_path)
    audio, sr = resample(audio, sr, cfg.target_sample_rate)

    diarizer = Diarizer(cfg)
    chunks, durations = split_audio(audio, sr, cfg.chunk_duration_sec)

    all_seg = [diarizer.run_chunk(c, sr) for c in tqdm(chunks, desc="Diarizing", unit="chunk")]
    merged = merge_segments(all_seg, durations)

    transcriber = Transcriber(cfg)
    mono = audio.mean(axis=1).astype("float32")

    srt_records = []
    for idx, seg in enumerate(tqdm(merged, desc="Transcribing", unit="seg"), start=1):
        s, e = int(seg.start * sr), int(seg.end * sr)
        snippet = mono[s:e]
        try:
            text = transcriber.transcribe(snippet, sr)
        except Exception as ex:  # noqa: BLE001
            text = f"[ERROR: {ex}]"
        line = f"speaker_{seg.speaker:02}: {text}"
        srt_records.append((idx, seg.start, seg.end, line))

    write_srt(srt_path, srt_records)
    if cfg.export_txt:
        write_txt(txt_path, srt_records)
    log.info("✅ Completed: %s", srt_path.name)
