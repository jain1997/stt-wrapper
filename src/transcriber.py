"""Whisper-based transcription with CUDA/CPU support."""

from __future__ import annotations
import logging
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .settings import AppConfig

log = logging.getLogger(__name__)


class Transcriber:
    """Handles Whisper transcription."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        model_id = self.cfg.whisper_model_id
        device_idx = 0 if self.cfg.device == "cuda" else "cpu"
        dtype = torch.float16 if self.cfg.device == "cuda" else torch.float32

        log.info("Loading Whisper model on %s ...", self.cfg.device)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.cfg.device)

        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device_idx,
            torch_dtype=dtype,
            generate_kwargs={"task": self.cfg.task, "language": self.cfg.asr_language},
        )
        log.info("Whisper ready on %s.", self.cfg.device)
        return pipe

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        out = self.pipe({"array": audio, "sampling_rate": sr})
        return (out.get("text") or "").strip()
