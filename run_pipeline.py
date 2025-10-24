"""Entry point for diarization + transcription pipeline."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import torch

from src.settings import CONFIG, setup_logger
from src.pipeline import process_file

log = logging.getLogger(__name__)


def _find_audio_files(folder: Path, patterns: list[str]) -> list[Path]:
    """Find all audio files matching given patterns recursively."""
    files = []
    for pat in patterns:
        files.extend(folder.rglob(pat))
    return [f for f in sorted(set(files)) if f.is_file()]


def main() -> None:
    """Run diarization + ASR pipeline with optional CUDA/CPU selection."""
    parser = argparse.ArgumentParser(
        description="Run speaker diarization (sherpa-onnx) + Whisper ASR."
    )
    parser.add_argument("--input", required=True, help="Path to file or folder.")
    parser.add_argument("--output", default=None, help="Optional output directory.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force compute device (overrides config).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    setup_logger(verbose=args.verbose)
    cfg = CONFIG

    # Override device if passed
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            log.warning("⚠️ CUDA not available! Falling back to CPU.")
            cfg.device_preference = "cpu"
        else:
            cfg.device_preference = args.device
        log.info("⚙️ Using device: %s", cfg.device)

    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else None

    if input_path.is_file():
        process_file(input_path, cfg, out_dir, overwrite=args.overwrite)
        return

    if input_path.is_dir():
        files = _find_audio_files(input_path, cfg.glob_audio_extensions)
        log.info("Found %d audio file(s).", len(files))
        for f in files:
            try:
                process_file(f, cfg, out_dir, overwrite=args.overwrite)
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed: %s (%s)", f.name, exc)
        log.info("Batch completed.")
        return

    raise FileNotFoundError(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
