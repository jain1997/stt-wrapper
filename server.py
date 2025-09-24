# server.py
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import JSONResponse

# faster-whisper + transformers
from faster_whisper import WhisperModel
from transformers import pipeline

# Optional high-quality resampling if torchaudio is installed
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Faster-Whisper Streaming Transcribe+Translate")

# Defaults (can be overridden per-WS "start" message)
DEFAULTS = {
    "model": "small",            # small/medium/large
    "device": "cpu",             # "cpu" or "cuda"
    "compute_type": "float32",   # float32 | float16 | int8_float16 | int8
    "sample_rate": 16000,
    "chunk_sec": 0.5,
    "language": "auto",          # "auto" or language code like "en","hi"
    "translate": [],             # e.g. ["en", "hi"]
    "vad_filter": True,          # faster-whisper VAD filter
    "beam_size": 5,
    "task": "transcribe",        # or "translate"
}

# Thread pool for blocking model calls
EXECUTOR = ThreadPoolExecutor(max_workers=2)

# model / pipeline caches
_WHISPER_MODELS: Dict[str, WhisperModel] = {}
_TRANSLATION_PIPES: Dict[str, Any] = {}

# --- Utilities ---

def load_whisper_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    """
    Load & cache WhisperModel. compute_type is passed to WhisperModel(..., compute_type=...).
    Note: choose compute_type supported by your GPU/backend (float16, int8, etc).
    """
    key = f"{model_name}|{device}|{compute_type}"
    if key in _WHISPER_MODELS:
        return _WHISPER_MODELS[key]
    logger.info("Loading WhisperModel %s on %s compute_type=%s", model_name, device, compute_type)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _WHISPER_MODELS[key] = model
    return model

def get_translation_pipeline(target_lang: str, use_cuda: bool) -> Any:
    """Helsinki-NLP Marian models for en<->hi. Cache pipeline per target_lang & device."""
    device_id = 0 if use_cuda else -1
    key = f"{target_lang}|{device_id}"
    if key in _TRANSLATION_PIPES:
        return _TRANSLATION_PIPES[key]

    if target_lang == "en":
        model_name = "Helsinki-NLP/opus-mt-mul-en"
    elif target_lang == "hi":
        model_name = "Helsinki-NLP/opus-mt-en-hi"
    else:
        raise ValueError("Unsupported translation target: " + target_lang)

    logger.info("Loading translation pipeline %s (device=%s)", model_name, device_id)
    pipe = pipeline("translation", model=model_name, device=device_id, framework="pt")
    _TRANSLATION_PIPES[key] = pipe
    return pipe

def read_audio_bytes(buf: bytes, expected_sr: int) -> Tuple[np.ndarray, int]:
    """
    Try to read with soundfile (wav etc). Fallback: interpret as PCM16 mono at expected_sr.
    Return float32 normalized numpy array, samplerate.
    """
    import io
    bio = io.BytesIO(buf)
    try:
        audio, sr = sf.read(bio, dtype="int16")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return audio, sr
    except Exception:
        # raw PCM16 mono fallback
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, expected_sr

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Prefer torchaudio if available (better quality), else use numpy interpolation."""
    if orig_sr == target_sr:
        return audio
    if TORCHAUDIO_AVAILABLE:
        # torchaudio expects tensors
        import torch
        tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
        resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
        return resampled.squeeze(0).numpy().astype(np.float32)
    else:
        # simple linear interpolation
        import math
        old_idx = np.arange(len(audio))
        new_len = int(math.ceil(len(audio) * target_sr / orig_sr))
        new_idx = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(new_idx, old_idx, audio).astype(np.float32)

def normalize_transcribe_result(res: Any) -> Dict[str, Any]:
    """
    faster-whisper versions differ: some return dict, some return (segments, info), etc.
    Normalize to dict {'text':..., 'segments':[...], 'language':...}
    """
    out = {"text": "", "segments": [], "language": None}
    if isinstance(res, dict):
        out["text"] = res.get("text", "")
        out["segments"] = res.get("segments", []) or []
        out["language"] = res.get("language") or getattr(res, "language", None)
    elif isinstance(res, tuple) and len(res) == 2:
        # (segments, info)
        segments, info = res
        out["segments"] = [{"start": getattr(s, "start", None), "end": getattr(s, "end", None), "text": getattr(s, "text", str(s))} for s in segments]
        out["text"] = " ".join([s["text"] for s in out["segments"]]) if out["segments"] else ""
        out["language"] = getattr(info, "language", None) or (info.get("language") if isinstance(info, dict) else None)
    else:
        # try best-effort
        try:
            out["text"] = str(res)
        except Exception:
            out["text"] = ""
    return out

# Blocking transcription call (runs in executor)
def blocking_transcribe(model: WhisperModel, audio: np.ndarray, sr: int, language: Optional[str], task: str, vad_filter: bool, beam_size: int):
    # faster-whisper's transcribe signature may vary; call with safe kwargs
    return model.transcribe(audio, beam_size=beam_size, language=language, task=task, vad_filter=vad_filter)

async def transcribe_bytes_in_executor(model: WhisperModel, audio_bytes: bytes, expected_sr: int,
                                       language: Optional[str], task: str, vad_filter: bool, beam_size: int):
    loop = asyncio.get_running_loop()
    audio, orig_sr = await loop.run_in_executor(None, read_audio_bytes, audio_bytes, expected_sr)
    if orig_sr != expected_sr:
        audio = await loop.run_in_executor(None, resample_audio, audio, orig_sr, expected_sr)
    # blocking model call
    res = await loop.run_in_executor(EXECUTOR, blocking_transcribe, model, audio, expected_sr, language, task, vad_filter, beam_size)
    return normalize_transcribe_result(res)

# --- Websocket handler ---

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket, sample_rate: int = Query(DEFAULTS["sample_rate"])):
    """
    Protocol:
     - client sends text JSON: {"action":"start", ...config...}
     - client sends binary audio frames (PCM16 WAV bytes or raw PCM16)
     - client may request partial with {"action":"partial_request"}
     - client ends with {"action":"end"} -> server transcribes buffered data, translates if requested, sends final result then closes.
    """
    await ws.accept()
    logger.info("WS connected")
    # state
    buffer: List[bytes] = []
    cfg = DEFAULTS.copy()
    try:
        while True:
            msg = await ws.receive()
            # binary frames
            if "bytes" in msg:
                buffer.append(msg["bytes"])
                continue

            # text messages
            text = msg.get("text")
            if not text:
                continue
            data = json.loads(text)
            action = data.get("action")
            if action == "start":
                # merge configuration
                for k in cfg.keys():
                    if k in data:
                        cfg[k] = data[k]
                # validate compute_type permitted values
                # (faster-whisper will error if unsupported)
                # load model
                _ = load_whisper_model(cfg["model"], cfg["device"], cfg["compute_type"])
                await ws.send_text(json.dumps({"type":"ack", "config": cfg}))
            elif action == "partial_request":
                if not buffer:
                    await ws.send_text(json.dumps({"type":"partial", "text": ""}))
                    continue
                audio_buf = b"".join(buffer)
                model = load_whisper_model(cfg["model"], cfg["device"], cfg["compute_type"])
                res = await transcribe_bytes_in_executor(model, audio_buf, sample_rate,
                                                        None if cfg["language"] == "auto" else cfg["language"],
                                                        cfg["task"], cfg["vad_filter"], cfg["beam_size"])
                await ws.send_text(json.dumps({"type":"partial", "text": res["text"]}))
            elif action == "end":
                audio_buf = b"".join(buffer)
                model = load_whisper_model(cfg["model"], cfg["device"], cfg["compute_type"])
                await ws.send_text(json.dumps({"type":"info", "message":"finalizing"}))
                res = await transcribe_bytes_in_executor(model, audio_buf, sample_rate,
                                                        None if cfg["language"] == "auto" else cfg["language"],
                                                        cfg["task"], cfg["vad_filter"], cfg["beam_size"])
                out = {"type":"final", "text": res["text"], "segments": res["segments"], "detected_language": res["language"]}
                # translations if requested
                if cfg.get("translate"):
                    trans = {}
                    loop = asyncio.get_running_loop()
                    for tgt in cfg["translate"]:
                        try:
                            pipe = get_translation_pipeline(tgt, use_cuda=(cfg["device"].startswith("cuda")))
                            # blocking; run in executor
                            translated = await loop.run_in_executor(EXECUTOR, lambda t=pipe: t(res["text"], max_length=512)[0]["translation_text"])
                            trans[tgt] = translated
                        except Exception as e:
                            trans[tgt] = {"error": str(e)}
                    out["translations"] = trans
                await ws.send_text(json.dumps(out))
                await ws.close()
                break
            else:
                await ws.send_text(json.dumps({"type":"error", "message":"Unknown action"}))
    except Exception as e:
        logger.exception("websocket error")
        try:
            await ws.send_text(json.dumps({"type":"error", "message": str(e)}))
            await ws.close()
        except Exception:
            pass

@app.get("/health")
async def health():
    return JSONResponse({"status":"ok"})
