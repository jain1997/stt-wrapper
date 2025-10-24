"""Write SRT and TXT transcript files."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import logging
from .audio_utils import srt_time

log = logging.getLogger(__name__)


def write_srt(path: Path, segments: Iterable[tuple[int, float, float, str]]) -> None:
    lines = []
    for idx, start, end, text in segments:
        lines += [str(idx), f"{srt_time(start)} --> {srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved SRT → %s", path)


def write_txt(path: Path, segments: Iterable[tuple[int, float, float, str]]) -> None:
    lines = [
        f"[{srt_time(start).replace(',', '.')} → {srt_time(end).replace(',', '.')}] {text}"
        for _, start, end, text in segments
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved TXT → %s", path)
