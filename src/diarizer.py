"""Speaker diarization with sherpa-onnx (GPU-aware and version-safe)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import logging
import numpy as np
import sherpa_onnx
from .settings import AppConfig

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    speaker: int


class Diarizer:
    """Performs speaker diarization using sherpa-onnx."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = self._init_model()

    def _init_model(self):
        log.info("Initializing sherpa-onnx diarizer...")

        # Handle both pyannote/pyanote naming
        seg_cls = getattr(sherpa_onnx, "OfflineSpeakerSegmentationPyannoteModelConfig", None)
        if seg_cls is None:
            seg_cls = getattr(sherpa_onnx, "OfflineSpeakerSegmentationPyanoteModelConfig", None)
        if seg_cls is None:
            raise RuntimeError("Pyannote segmentation config not found in sherpa_onnx.")

        seg_cfg = seg_cls(model=self.cfg.segmentation_model)

        # Choose correct argument name (pyannote or pyanote)
        try:
            segmentation_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=seg_cfg, provider="cuda" if self.cfg.device == "cuda" else "cpu"
            )
        except TypeError:
            segmentation_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyanote=seg_cfg, provider="cuda" if self.cfg.device == "cuda" else "cpu"
            )

        embed_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.cfg.embedding_extractor_model,
            provider="cuda" if self.cfg.device == "cuda" else "cpu",
        )

        cluster_cfg = sherpa_onnx.FastClusteringConfig(
            num_clusters=self.cfg.num_speakers,
            threshold=self.cfg.cluster_threshold,
        )

        diar_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=segmentation_cfg,
            embedding=embed_cfg,
            clustering=cluster_cfg,
            min_duration_on=self.cfg.min_duration_on,
            min_duration_off=self.cfg.min_duration_off,
        )

        if not diar_cfg.validate():
            raise RuntimeError("Invalid diarization config (check model paths).")

        log.info("Sherpa-onnx diarizer initialized successfully (%s).", self.cfg.device)
        return sherpa_onnx.OfflineSpeakerDiarization(diar_cfg)

    def run_chunk(self, chunk: np.ndarray, sr: int) -> List[Segment]:
        mono = chunk.mean(axis=1).astype("float32")
        try:
            res = self.model.process(mono, sr)
        except TypeError:
            res = self.model.process(mono)
        res = res.sort_by_start_time()
        return [Segment(s.start, s.end, s.speaker) for s in res]


def merge_segments(results: List[List[Segment]], durations: List[float]) -> List[Segment]:
    merged, offset = [], 0.0
    for segs, dur in zip(results, durations):
        for s in segs:
            merged.append(Segment(s.start + offset, s.end + offset, s.speaker))
        offset += dur
    merged.sort(key=lambda s: s.start)
    log.info("Merged %d chunks → %d segments total", len(results), len(merged))
    return merged
