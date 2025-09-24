# mic_client.py
import argparse
import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets

async def mic_stream(ws_url: str, model: str = "small", device: str = "cpu",
                     compute_type: str = "float32", translate: list = None,
                     language: str = "auto", sample_rate: int = 16000, chunk_sec: float = 0.5):
    if translate is None:
        translate = []
    async with websockets.connect(ws_url, max_size=None) as ws:
        # send start config
        await ws.send(json.dumps({
            "action": "start",
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "language": language,
            "translate": translate,
            "chunk_sec": chunk_sec
        }))
        q = asyncio.Queue()

        def callback(indata, frames, time, status):
            if status:
                print("Sounddevice status:", status)
            # sounddevice default dtype float32 -> convert to PCM16
            pcm16 = (indata.flatten() * 32767).astype(np.int16)
            q.put_nowait(pcm16.tobytes())

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
            print("Recording... Ctrl+C to stop")
            try:
                while True:
                    data = await q.get()
                    await ws.send(data)
                    # non-blocking check for server messages
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                        print("SERVER:", msg)
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("Finalizing...")
                await ws.send(json.dumps({"action":"end"}))
                try:
                    async for message in ws:
                        print("SERVER:", message)
                except Exception:
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", default="ws://localhost:8000/ws/transcribe")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="float32")
    parser.add_argument("--translate", nargs="*", default=["en","hi"])
    parser.add_argument("--language", default="auto")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--chunk", type=float, default=0.5)
    args = parser.parse_args()
    asyncio.run(mic_stream(args.ws, model=args.model, device=args.device,
                          compute_type=args.compute_type, translate=args.translate,
                          language=args.language, sample_rate=args.sr, chunk_sec=args.chunk))
