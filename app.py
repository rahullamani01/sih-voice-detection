import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import optimized detector functions
from detector import (
    predict_batch,
    make_ledger_hash,
    SAMPLE_RATE,
    WINDOW_SIZE,
    PROVISIONAL_MIN,
)

app = FastAPI(title="SIH26104 Real-Time Voice Guard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dedicated thread pool for ONNX runtime inference execution
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Analysis stride: Trigger inference every 200 ms
INFERENCE_STRIDE_SEC = 0.2


@app.get("/")
def health_check():
    """Root route to verify server status via browser."""
    return {
        "status": "online",
        "service": "Real-Time AI Voice Clone Detection API",
        "websocket_endpoint": "ws://localhost:8000/ws/live",
    }


class HighPerformanceLiveSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]

        # Preallocated float32 NumPy Ring Buffer
        self.max_buffer_samples = WINDOW_SIZE * 2
        self.buffer = np.zeros(self.max_buffer_samples, dtype=np.float32)
        self.buffer_len = 0

        self.scores_history = []
        self.chunk_count = 0
        self.start_time = time.time()
        self.last_inference_time = 0.0
        self.busy = False
        self.dropped = 0

    def add_audio(self, samples: np.ndarray):
        """Appends new samples to the ring buffer efficiently using NumPy rolls."""
        num_samples = len(samples)
        if num_samples == 0:
            return

        if self.buffer_len + num_samples <= self.max_buffer_samples:
            self.buffer[self.buffer_len : self.buffer_len + num_samples] = samples
            self.buffer_len += num_samples
        else:
            shift = num_samples
            self.buffer = np.roll(self.buffer, -shift)
            self.buffer[-num_samples:] = samples
            self.buffer_len = self.max_buffer_samples

        self.chunk_count += 1

    def _get_window(self):
        """Extracts the audio slice directly without type conversions."""
        if self.buffer_len >= WINDOW_SIZE:
            return self.buffer[self.buffer_len - WINDOW_SIZE : self.buffer_len], True
        if self.buffer_len >= PROVISIONAL_MIN:
            partial = self.buffer[: self.buffer_len]
            padded = np.pad(partial, (0, WINDOW_SIZE - len(partial)), mode="constant")
            return padded, False
        return None, False

    async def analyze_async(self) -> Optional[dict]:
        now = time.perf_counter()

        if (now - self.last_inference_time) < INFERENCE_STRIDE_SEC:
            return None

        if self.busy:
            self.dropped += 1
            return None

        window, confident = self._get_window()
        if window is None:
            return {
                "ready": False,
                "message": f"Buffering… ({self.buffer_len}/{PROVISIONAL_MIN} min samples)",
            }

        self.busy = True
        self.last_inference_time = now
        t0 = time.perf_counter()

        try:
            loop = asyncio.get_running_loop()

            # Shape for batch ONNX inference: (1, WINDOW_SIZE)
            batched_window = np.expand_dims(window, axis=0)

            # Offload heavy ONNX runtime call to worker thread
            probabilities = await loop.run_in_executor(
                EXECUTOR, predict_batch, batched_window
            )
            ai_prob = float(probabilities[0])

        finally:
            self.busy = False

        inference_ms = round((time.perf_counter() - t0) * 1000, 1)
        risk_score = round(ai_prob * 100, 1)

        if confident:
            self.scores_history.append(risk_score)
            if len(self.scores_history) > 10:
                self.scores_history.pop(0)

        stability = (
            round(float(np.std(self.scores_history)), 1)
            if len(self.scores_history) > 1
            else 0.0
        )

        if not confident:
            status, risk_level = "ANALYZING: Provisional read", "PROVISIONAL"
        elif risk_score >= 70:
            status, risk_level = "ALERT: AI Voice Detected", "CRITICAL"
        elif risk_score >= 40:
            status, risk_level = "INCONCLUSIVE: Review Required", "MODERATE"
        else:
            status, risk_level = "VERIFIED: Human Voice", "LOW"

        timestamp, ledger_hash = make_ledger_hash(self.session_id, risk_score)

        return {
            "ready": True,
            "confident": confident,
            "session_id": self.session_id,
            "risk_score": risk_score,
            "ai_probability": risk_score,
            "human_probability": round((1.0 - ai_prob) * 100, 1),
            "status": status,
            "risk_level": risk_level,
            "stability": stability,
            "windows_analyzed": self.chunk_count,
            "buffer_seconds": round(self.buffer_len / SAMPLE_RATE, 2),
            "inference_ms": inference_ms,
            "dropped_chunks": self.dropped,
            "timestamp": timestamp,
            "ledger_hash": ledger_hash,
            "elapsed_sec": round(time.time() - self.start_time, 1),
        }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    session = HighPerformanceLiveSession()

    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session.session_id,
            "message": "Live session started. Send 16 kHz mono PCM chunks.",
        })

        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]

                # Automatically convert int16 audio streams or float32 arrays
                try:
                    samples = np.frombuffer(raw_bytes, dtype=np.float32)
                except ValueError:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(
                        np.float32
                    ) / 32768.0

                session.add_audio(samples)

                # Process chunk conditionally
                result = await session.analyze_async()
                if result is not None:
                    await websocket.send_json({"type": "score", **result})

            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"Session {session.session_id} disconnected")
    except Exception as e:
        print(f"Error in session {session.session_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass