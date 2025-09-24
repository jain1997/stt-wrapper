# server_offline_local.py
import asyncio
import io
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import JSONResponse

# faster-whisper must be installed in the environment
from faster_whisper import WhisperModel

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Faster-Whisper Offline Local Loader")

# executor to run blocking model calls
EXECUTOR = ThreadPoolExecutor(max_workers=2)

# model cache
_WHISPER_CACHE: Dict[str, WhisperModel] = {}

# default config (can be overridden by client start message)
DEFAULT_CFG = {
    "whisper_model_dir": "./models/whisper/small-ct2",
    "device": "cpu",
    "compute_type": "float32",   # or "float16" on CUDA GPUs
    "local_files_only": True,    # IMPORTANT: force local-only loading
    "sample_rate": 16000,
    "language": "auto",
    "translate_to": ["en"],      # server focuses on language -> English using built-in 'translate'
    "vad_filter": True,
    "beam_size": 5,
}


# -------------------------
# Helper: safe local loader
# -------------------------
def load_whisper_local_safe(model_spec: str,
                           device: str = "cpu",
                           compute_type: str = "float32",
                           local_files_only: bool = True,
                           files: Optional[Dict[str, Any]] = None) -> WhisperModel:
    """
    Load WhisperModel ensuring no network calls at runtime.

    model_spec: either a path to a CT2-converted folder OR a model id/name that is already cached locally.
    local_files_only: if True, pass to WhisperModel to avoid downloads; if missing, WhisperModel will raise.
    files: optional dict mapping filename -> bytes/fileobj (advanced - in-memory load)

    Returns a cached WhisperModel instance.
    """
    # build cache key (if model_spec is a path, use abs path)
    if os.path.exists(model_spec):
        model_ident = os.path.abspath(model_spec)
    else:
        model_ident = model_spec
    key = f"{model_ident}|{device}|{compute_type}|{local_files_only}"
    if key in _WHISPER_CACHE:
        return _WHISPER_CACHE[key]

    # quick sanity checks & helpful errors
    if os.path.exists(model_spec) and not os.path.isdir(model_spec):
        raise FileNotFoundError(f"Model path exists but is not a directory: {model_spec}")

    init_kwargs: Dict[str, Any] = {"device": device, "compute_type": compute_type}
    if local_files_only:
        init_kwargs["local_files_only"] = True
    if files is not None:
        init_kwargs["files"] = files

    logger.info("Loading WhisperModel: model_spec=%s device=%s compute_type=%s local_only=%s",
                model_spec, device, compute_type, local_files_only)

    # This will either:
    # - load from the provided CT2 folder if model_spec is a path, OR
    # - load from local HF cache when model_spec is a model id and local_files_only=True,
    #   or raise an error if not present.
    model = WhisperModel(model_spec, **init_kwargs)
    _WHISPER_CACHE[key] = model
    return model


# -------------------------
# Audio utilities
# -------------------------
def read_audio_bytes(buf: bytes, expected_sr: int) -> Tuple[np.ndarray, int]:
    """
    Try reading common audio containers with soundfile; else interpret buf as PCM16 mono @ expected_sr.
    Returns (float32_numpy_array, sr)
    """
    bio = io.BytesIO(buf)
    try:
        audio, sr = sf.read(bio, dtype="int16")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return audio, sr
    except Exception:
        # fallback raw PCM16
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, expected_sr


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear interpolation resampling fallback."""
    if orig_sr == target_sr:
        return audio
    old_idx = np.arange(len(audio))
    new_len = int(math.ceil(len(audio) * target_sr / orig_sr))
    new_idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def normalize_whisper_result(res: Any) -> Dict[str, Any]:
    """Normalize faster-whisper results into a uniform dict."""
    out = {"text": "", "segments": [], "language": None}
    if isinstance(res, dict):
        out["text"] = res.get("text", "")
        out["segments"] = res.get("segments", []) or []
        out["language"] = res.get("language") or getattr(res, "language", None)
    elif isinstance(res, tuple) and len(res) == 2:
        segments, info = res
        out["segments"] = [{"start": getattr(s, "start", None), "end": getattr(s, "end", None),
                            "text": getattr(s, "text", str(s))} for s in segments]
        out["text"] = " ".join([s["text"] for s in out["segments"]]) if out["segments"] else ""
        out["language"] = getattr(info, "language", None) or (info.get("language") if isinstance(info, dict) else None)
    else:
        out["text"] = str(res)
    return out


# blocking model call wrapper (run in executor)
def blocking_transcribe(model: WhisperModel, audio: np.ndarray, sr: int, language: Optional[str],
                        task: str, vad_filter: bool, beam_size: int):
    return model.transcribe(audio, beam_size=beam_size, language=language, task=task, vad_filter=vad_filter)


async def transcribe_bytes_in_executor(model: WhisperModel, audio_bytes: bytes, expected_sr: int,
                                       language: Optional[str], task: str, vad_filter: bool, beam_size: int):
    loop = asyncio.get_running_loop()
    audio, sr = await loop.run_in_executor(None, read_audio_bytes, audio_bytes, expected_sr)
    if sr != expected_sr:
        audio = await loop.run_in_executor(None, resample_linear, audio, sr, expected_sr)
    res = await loop.run_in_executor(EXECUTOR, blocking_transcribe, model, audio, expected_sr, language, task, vad_filter, beam_size)
    return normalize_whisper_result(res)


# -------------------------
# WebSocket handler
# -------------------------
@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket, sample_rate: int = Query(DEFAULT_CFG["sample_rate"])):
    """
    WebSocket control protocol:
      - client sends {"action":"start", ...cfg} as TEXT (must include whisper_model_dir)
      - client streams binary audio frames (PCM16 or WAV bytes)
      - client may send {"action":"partial_request"} to get interim transcription
      - client sends {"action":"end"} to finalize -> server returns final transcription or English translation
    """
    await ws.accept()
    logger.info("WS connected")
    buffer: List[bytes] = []
    cfg = DEFAULT_CFG.copy()

    try:
        while True:
            msg = await ws.receive()

            # binary frames (audio)
            if "bytes" in msg:
                buffer.append(msg["bytes"])
                continue

            # text control message
            text = msg.get("text")
            if not text:
                continue
            data = json.loads(text)
            action = data.get("action")

            if action == "start":
                # merge runtime config into cfg
                for k in cfg.keys():
                    if k in data:
                        cfg[k] = data[k]
                # require whisper_model_dir present and offline
                if not cfg.get("whisper_model_dir"):
                    await ws.send_text(json.dumps({"type": "error", "message": "whisper_model_dir missing in start message"}))
                    continue
                if not os.path.exists(cfg["whisper_model_dir"]):
                    await ws.send_text(json.dumps({"type":"error", "message": f"whisper_model_dir not found: {cfg['whisper_model_dir']}"}))
                    continue

                # load model locally (throws if files missing)
                try:
                    model = load_whisper_local_safe(cfg["whisper_model_dir"], device=cfg["device"],
                                                    compute_type=cfg["compute_type"],
                                                    local_files_only=cfg.get("local_files_only", True))
                except Exception as exc:
                    logger.exception("Failed local model load")
                    await ws.send_text(json.dumps({"type":"error", "message": f"failed to load whisper model locally: {exc}"}))
                    await ws.close()
                    return

                await ws.send_text(json.dumps({"type":"ack", "config": cfg}))
                continue

            if action == "partial_request":
                if not buffer:
                    await ws.send_text(json.dumps({"type":"partial", "text": ""}))
                    continue
                audio_buf = b"".join(buffer)
                model = load_whisper_local_safe(cfg["whisper_model_dir"], device=cfg["device"], compute_type=cfg["compute_type"], local_files_only=cfg.get("local_files_only", True))
                res = await transcribe_bytes_in_executor(model, audio_buf, sample_rate,
                                                         None if cfg["language"] == "auto" else cfg["language"],
                                                         "transcribe", cfg["vad_filter"], cfg["beam_size"])
                await ws.send_text(json.dumps({"type":"partial", "text": res["text"]}))
                continue

            if action == "end":
                audio_buf = b"".join(buffer)
                model = load_whisper_local_safe(cfg["whisper_model_dir"], device=cfg["device"], compute_type=cfg["compute_type"], local_files_only=cfg.get("local_files_only", True))

                # If client asked for English output, use built-in translate task to let Whisper produce English directly
                wants_en = "en" in (cfg.get("translate_to") or [])
                task = "translate" if wants_en else "transcribe"

                res = await transcribe_bytes_in_executor(model, audio_buf, sample_rate,
                                                         None if cfg["language"] == "auto" else cfg["language"],
                                                         task, cfg["vad_filter"], cfg["beam_size"])

                out = {
                    "type": "final",
                    "text": res["text"],
                    "segments": res["segments"],
                    "detected_language": res["language"],
                }

                await ws.send_text(json.dumps(out))
                await ws.close()
                break

            await ws.send_text(json.dumps({"type":"error", "message":"unknown action"}))

    except Exception as e:
        logger.exception("websocket handler error")
        try:
            await ws.send_text(json.dumps({"type":"error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    return JSONResponse({"status":"ok", "note":"offline local whisper loader; using translate->English if requested"})
