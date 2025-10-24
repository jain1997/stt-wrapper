"""Centralized settings and logging for the speech pipeline."""

from __future__ import annotations
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
import yaml


# ==========================
# Logging Configuration
# ==========================

def setup_logger(verbose: bool = False) -> None:
    """Set up global logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    logging.basicConfig(level=level, handlers=[handler])
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


# ==========================
# App Configuration
# ==========================

@dataclass
class AppConfig:
    """Stores pipeline runtime configuration."""

    device_preference: str = "cuda"
    num_threads: int = 1
    chunk_duration_sec: int = 300
    output_suffix: str = "_diarized"
    export_txt: bool = True

    segmentation_model: str = ""
    embedding_extractor_model: str = ""
    whisper_model_id: str = "openai/whisper-small"

    num_speakers: int = 2
    cluster_threshold: float = 0.5
    min_duration_on: float = 0.1
    min_duration_off: float = 0.1

    target_sample_rate: int = 16000
    glob_audio_extensions: Optional[List[str]] = None

    asr_language: str = "en"
    task: str = "transcribe"

    def __post_init__(self) -> None:
        if self.glob_audio_extensions is None:
            self.glob_audio_extensions = ["*.wav"]
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    @property
    def device(self) -> str:
        """Returns the best available compute device."""
        if self.device_preference.lower() == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"


def load_config(path: str | Path) -> AppConfig:
    """Load YAML configuration."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = AppConfig(**data)
    logging.getLogger(__name__).info("Configuration loaded from %s", path)
    return cfg


# ==========================
# Auto-load Global Config
# ==========================

def _find_default_config() -> Path:
    """Locate config.yaml in project root or via env var."""
    env_cfg = os.getenv("SPEECH_PIPELINE_CONFIG")
    if env_cfg and Path(env_cfg).exists():
        return Path(env_cfg)
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config.yaml"
    if cfg_path.exists():
        return cfg_path
    raise FileNotFoundError("config.yaml not found in project root or env variable.")


try:
    CONFIG_PATH = _find_default_config()
    CONFIG = load_config(CONFIG_PATH)
    logging.getLogger(__name__).info("✅ CONFIG loaded: %s", CONFIG_PATH)
except Exception as e:  # noqa: BLE001
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).warning("⚠️ Failed to load config automatically: %s", e)
    CONFIG = AppConfig()
