

import asyncio
import json
import time
import hashlib
import uuid
from collections import deque
from typing import Deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import predict, SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE  # reuse your code

app = FastAPI(title="SIH26104 Real-Time Voice Guard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Per-connection rolling buffer state
# ---------------------------------------------------------------------------
class LiveSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.buffer: Deque[float] = deque(maxlen=WINDOW_SIZE * 2)  # keep ~6 s
        self.last_score = 0.0
        self.last_status = "WAITING"
        self.scores_history = []          # for stability
        self.chunk_count = 0
        self.start_time = time.time()

    def add_audio(self, samples: np.ndarray):
        """Append new PCM float32 samples"""
        self.buffer.extend(samples.tolist())
        self.chunk_count += 1

    def get_window(self) -> np.ndarray | None:
        """Return the latest WINDOW_SIZE samples if we have enough"""
        if len(self.buffer) < WINDOW_SIZE:
            return None
        window = np.array(list(self.buffer)[-WINDOW_SIZE:], dtype=np.float32)
        return window

    def analyze(self) -> dict:
        window = self.get_window()
        if window is None:
            return {
                "ready": False,
                "message": f"Buffering… ({len(self.buffer)}/{WINDOW_SIZE} samples)"
            }

        # Run your existing ONNX predict
        ai_prob = predict(window)          # 0.0 – 1.0
        risk_score = round(ai_prob * 100, 1)

        self.scores_history.append(risk_score)
        if len(self.scores_history) > 10:
            self.scores_history.pop(0)

        stability = round(float(np.std(self.scores_history)), 1) if len(self.scores_history) > 1 else 0.0

        if risk_score >= 70:
            status = "ALERT: AI Voice Detected"
            risk_level = "CRITICAL"
        elif risk_score >= 40:
            status = "INCONCLUSIVE: Review Required"
            risk_level = "MODERATE"
        else:
            status = "VERIFIED: Human Voice"
            risk_level = "LOW"

        self.last_score = risk_score
        self.last_status = status

        # Ledger hash (same style as your detector)
        timestamp = str(int(time.time()))
        payload = f"{timestamp}:{self.session_id}:{risk_score:.2f}".encode()
        ledger_hash = hashlib.sha256(payload).hexdigest()

        return {
            "ready": True,
            "session_id": self.session_id,
            "risk_score": risk_score,
            "ai_probability": round(ai_prob * 100, 1),
            "human_probability": round((1 - ai_prob) * 100, 1),
            "status": status,
            "risk_level": risk_level,
            "stability": stability,
            "windows_analyzed": self.chunk_count,
            "buffer_seconds": round(len(self.buffer) / SAMPLE_RATE, 2),
            "timestamp": timestamp,
            "ledger_hash": ledger_hash,
            "elapsed_sec": round(time.time() - self.start_time, 1)
        }


# ---------------------------------------------------------------------------
# WebSocket endpoint – the core of live call detection
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    session = LiveSession()

    try:
        # Send initial hello
        await websocket.send_json({
            "type": "connected",
            "session_id": session.session_id,
            "message": "Live session started. Send 16 kHz mono float32 PCM chunks."
        })

        while True:
            # Receive binary audio chunk (preferred) or JSON
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Client sent raw float32 PCM bytes
                raw = message["bytes"]
                samples = np.frombuffer(raw, dtype=np.float32)
            elif "text" in message:
                # Optional: client can send base64 or just a keep-alive
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                else:
                    continue
            else:
                continue

            # Add to rolling buffer
            session.add_audio(samples)

            # Analyze latest window (we do it every chunk; client can throttle)
            result = session.analyze()

            await websocket.send_json({
                "type": "score",
                **result
            })

    except WebSocketDisconnect:
        print(f"Session {session.session_id} disconnected")
    except Exception as e:
        print(f"Error in session {session.session_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# ---------------------------------------------------------------------------
# Simple test page so you can try it immediately
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return HTMLResponse(LIVE_HTML)


LIVE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>SIH26104 · Live Call Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background: #0b0f19; color: #e2e8f0; font-family: system-ui, sans-serif; }
    .gauge { transition: stroke-dashoffset 0.4s ease; }
  </style>
</head>
<body class="min-h-screen p-6">
  <div class="max-w-3xl mx-auto">
    <h1 class="text-2xl font-bold mb-1">🛡️ Live Voice Cloning Detection</h1>
    <p class="text-slate-400 text-sm mb-6">SIH26104 · Real-time WebSocket stream</p>

    <div class="bg-slate-900 border border-slate-700 rounded-2xl p-6 mb-6">
      <div class="flex items-center gap-4 mb-4">
        <button id="btn" class="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-semibold">
          Start Live Call
        </button>
        <span id="status" class="text-sm text-slate-400">Idle</span>
      </div>

      <div class="grid grid-cols-2 gap-4 text-center">
        <div>
          <div class="text-4xl font-mono font-bold" id="score">--%</div>
          <div class="text-xs text-slate-400 mt-1">AI Risk Score</div>
        </div>
        <div>
          <div class="text-lg font-semibold" id="verdict">Waiting</div>
          <div class="text-xs text-slate-400 mt-1" id="level"></div>
        </div>
      </div>

      <div class="mt-6 text-xs font-mono text-slate-500 space-y-1">
        <div>Session: <span id="sid">—</span></div>
        <div>Buffer: <span id="buf">0.0</span> s · Windows: <span id="win">0</span></div>
        <div>Stability: <span id="stab">—</span>% · Hash: <span id="hash" class="truncate">—</span></div>
      </div>
    </div>

    <div class="text-xs text-slate-500">
      Open browser console for detailed logs. Microphone permission required.
    </div>
  </div>

<script>
const SAMPLE_RATE = 16000;
const CHUNK_MS = 500;               // send every 500 ms
let ws = null;
let audioCtx = null;
let processor = null;
let source = null;
let stream = null;
let isRunning = false;

const btn = document.getElementById('btn');
const statusEl = document.getElementById('status');

btn.onclick = async () => {
  if (isRunning) {
    stop();
  } else {
    await start();
  }
};

async function start() {
  try {
    // 1. Connect WebSocket
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/live`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      statusEl.textContent = 'Connected – requesting mic…';
      statusEl.className = 'text-sm text-emerald-400';
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'connected') {
        document.getElementById('sid').textContent = msg.session_id;
      }
      if (msg.type === 'score' && msg.ready) {
        document.getElementById('score').textContent = msg.risk_score + '%';
        document.getElementById('verdict').textContent = msg.status;
        document.getElementById('level').textContent = msg.risk_level;
        document.getElementById('buf').textContent = msg.buffer_seconds;
        document.getElementById('win').textContent = msg.windows_analyzed;
        document.getElementById('stab').textContent = msg.stability;
        document.getElementById('hash').textContent = msg.ledger_hash.slice(0, 16) + '…';

        // Color feedback
        const scoreEl = document.getElementById('score');
        if (msg.risk_score >= 70) scoreEl.className = 'text-4xl font-mono font-bold text-red-400';
        else if (msg.risk_score >= 40) scoreEl.className = 'text-4xl font-mono font-bold text-yellow-400';
        else scoreEl.className = 'text-4xl font-mono font-bold text-emerald-400';
      }
    };

    ws.onclose = () => {
      statusEl.textContent = 'Disconnected';
      statusEl.className = 'text-sm text-slate-400';
    };

    // 2. Get microphone
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    source = audioCtx.createMediaStreamSource(stream);

    // ScriptProcessor is deprecated but still the simplest cross-browser way
    const bufferSize = 4096;
    processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    let leftover = new Float32Array(0);

    processor.onaudioprocess = (e) => {
      if (!isRunning || ws.readyState !== WebSocket.OPEN) return;

      const input = e.inputBuffer.getChannelData(0);
      // Concat leftover + new
      const combined = new Float32Array(leftover.length + input.length);
      combined.set(leftover);
      combined.set(input, leftover.length);

      // We send roughly every CHUNK_MS
      const samplesPerChunk = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000);

      let offset = 0;
      while (offset + samplesPerChunk <= combined.length) {
        const chunk = combined.slice(offset, offset + samplesPerChunk);
        ws.send(chunk.buffer);          // send raw Float32 ArrayBuffer
        offset += samplesPerChunk;
      }
      leftover = combined.slice(offset);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination); // needed for some browsers

    isRunning = true;
    btn.textContent = 'Stop Live Call';
    btn.className = 'px-5 py-2.5 rounded-lg bg-red-600 hover:bg-red-500 font-semibold';
    statusEl.textContent = '🔴 Live – streaming';
    statusEl.className = 'text-sm text-red-400';

  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error: ' + err.message;
    statusEl.className = 'text-sm text-red-400';
  }
}

function stop() {
  isRunning = false;
  if (processor) { processor.disconnect(); processor = null; }
  if (source) { source.disconnect(); source = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws) { ws.close(); ws = null; }

  btn.textContent = 'Start Live Call';
  btn.className = 'px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-semibold';
  statusEl.textContent = 'Stopped';
  statusEl.className = 'text-sm text-slate-400';
}
</script>
</body>
</html>
"""