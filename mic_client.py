# mic_client.py
import argparse
import asyncio
import json
import signal
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

# Convert float32 [-1,1] to 16-bit PCM little-endian
def float_to_pcm16_bytes(float32_arr: np.ndarray) -> bytes:
    pcm16 = (float32_arr * 32767).astype(np.int16)
    return pcm16.tobytes()

async def mic_stream(ws_url: str,
                     whisper_model_dir: str,
                     device: str = "cpu",
                     compute_type: str = "float32",
                     local_files_only: bool = True,
                     translate_to: Optional[list] = None,
                     language: str = "auto",
                     sample_rate: int = 16000,
                     chunk_ms: int = 500):
    if translate_to is None:
        translate_to = ["en"]

    # prepare websocket
    async with websockets.connect(ws_url, max_size=None) as ws:
        # send start config
        start_cfg = {
            "action": "start",
            "whisper_model_dir": whisper_model_dir,
            "device": device,
            "compute_type": compute_type,
            "local_files_only": local_files_only,
            "language": language,
            "translate_to": translate_to,
            "vad_filter": True,
            "beam_size": 5
        }
        await ws.send(json.dumps(start_cfg))
        print("Sent start cfg:", start_cfg)

        q = asyncio.Queue()
        is_recording = True

        # sounddevice callback
        def callback(indata, frames, time, status):
            if status:
                print("Sounddevice status:", status, file=sys.stderr)
            # indata is float32 array in [-1,1], shape (frames, channels)
            arr = indata[:, 0] if indata.ndim > 1 else indata
            # push copy to queue
            q.put_nowait(arr.copy())

        # start input stream
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
                print("Recording... Press Ctrl+C to stop and finalize.")
                # schedule periodic sending loop
                async def sender_loop():
                    buff = []
                    buff_frames = 0
                    target_frames = int(sample_rate * (chunk_ms / 1000.0))
                    while True:
                        frame = await q.get()
                        buff.append(frame)
                        buff_frames += len(frame)
                        if buff_frames >= target_frames:
                            merged = np.concatenate(buff)
                            pcm_bytes = float_to_pcm16_bytes(merged)
                            await ws.send(pcm_bytes)
                            buff = []
                            buff_frames = 0

                sender_task = asyncio.create_task(sender_loop())

                # concurrently listen for server messages and print them
                async def receiver_loop():
                    try:
                        async for msg in ws:
                            # server is expected to send text JSON
                            if isinstance(msg, bytes):
                                print("Received binary message:", len(msg), "bytes")
                            else:
                                try:
                                    obj = json.loads(msg)
                                    print("SERVER:", obj)
                                except Exception:
                                    print("SERVER TEXT:", msg)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        print("Receiver error:", e)

                receiver_task = asyncio.create_task(receiver_loop())

                # Wait until user interrupts (Ctrl+C)
                loop = asyncio.get_running_loop()
                stop_event = asyncio.Event()

                def _on_signal(signum, frame):
                    stop_event.set()

                signal.signal(signal.SIGINT, _on_signal)
                await stop_event.wait()

                # finalize: cancel sender/receiver tasks gracefully
                sender_task.cancel()
                # flush any queued frames
                leftover = []
                while not q.empty():
                    leftover.append(await q.get())
                if leftover:
                    merged = np.concatenate(leftover)
                    await ws.send(float_to_pcm16_bytes(merged))
                    print("Flushed final chunk")

                # send 'end' control
                await ws.send(json.dumps({"action": "end"}))
                print("Sent end; awaiting final server responses...")

                # give receiver some time to print final messages, then cancel
                try:
                    await asyncio.wait_for(receiver_task, timeout=10.0)
                except asyncio.TimeoutError:
                    receiver_task.cancel()
        except Exception as exc:
            print("Error with audio stream:", exc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", default="ws://localhost:8000/ws/transcribe")
    parser.add_argument("--model_dir", required=True, help="Path to server model dir (server uses this to load model locally). Example: /opt/models/whisper/small-ct2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="float32")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--translate", nargs="*", default=["en"])
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--chunk_ms", type=int, default=500)
    args = parser.parse_args()

    asyncio.run(mic_stream(args.ws, whisper_model_dir=args.model_dir, device=args.device,
                           compute_type=args.compute_type, language=args.language,
                           translate_to=args.translate, sample_rate=args.sr, chunk_ms=args.chunk_ms))
