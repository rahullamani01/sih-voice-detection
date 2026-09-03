import hashlib
import io
import os
import time
import numpy as np
import onnxruntime as ort
import torch
import torchaudio

MODEL_PATH = "models/best_model.onnx"
SAMPLE_RATE = 16000
WINDOW_SIZE = 48000       # 3 seconds at 16kHz
HOP_SIZE = 24000          # 1.5 seconds overlap
PROVISIONAL_MIN = 16000   # 1 second minimum required for initial read

_session = None


def get_session():
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Place best_model.onnx inside models/"
            )
        # Load ONNX session with hardware execution providers
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _session = ort.InferenceSession(MODEL_PATH, providers=providers)
    return _session


def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Fast in-memory audio loading replacing librosa using torchaudio."""
    buffer = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buffer)

    # Convert multi-channel audio to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample on the fly if sample rate differs
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    y = waveform.squeeze(0).numpy().astype(np.float32)

    if len(y) == 0:
        raise ValueError("Empty audio payload received.")

    return y


def pad_to_window(y: np.ndarray) -> np.ndarray:
    """Pads short audio arrays up to WINDOW_SIZE using zero-padding."""
    if len(y) >= WINDOW_SIZE:
        return y[:WINDOW_SIZE].astype(np.float32)
    return np.pad(y, (0, WINDOW_SIZE - len(y)), mode="constant").astype(np.float32)


def make_windows_batched(y: np.ndarray) -> np.ndarray:
    """Creates overlapping audio windows into a 2D matrix for batched inference."""
    if len(y) <= WINDOW_SIZE:
        padded = pad_to_window(y)
        return np.expand_dims(padded, axis=0)

    windows = []
    for start in range(0, len(y) - WINDOW_SIZE + 1, HOP_SIZE):
        windows.append(y[start : start + WINDOW_SIZE])

    if len(y) > WINDOW_SIZE:
        last = y[-WINDOW_SIZE:]
        if not np.array_equal(windows[-1], last):
            windows.append(last)

    return np.array(windows, dtype=np.float32)


def predict(window: np.ndarray) -> float:
    """Single window inference wrapper compatible with LiveSession calls."""
    if window.ndim == 1:
        window = np.expand_dims(window, axis=0)
    probabilities = predict_batch(window)
    return float(probabilities[0])


def predict_batch(windows: np.ndarray) -> np.ndarray:
    """Normalizes windows and executes vectorized ONNX inference."""
    session = get_session()

    # Z-score normalization along axis 1
    means = np.mean(windows, axis=1, keepdims=True)
    vars_ = np.var(windows, axis=1, keepdims=True)
    normalized = (windows - means) / np.sqrt(vars_ + 1e-5)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: normalized})[0]
    logits = np.asarray(logits, dtype=np.float32)

    # Numerically stable Softmax
    logits_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

    # Return AI Probability column (index 1)
    return probabilities[:, 1]


def make_ledger_hash(session_id: str, risk_score: float):
    """Generates an audit hash for the blockchain verification layer."""
    timestamp = str(int(time.time()))
    payload = f"{timestamp}:{session_id}:{risk_score:.2f}".encode("utf-8")
    ledger_hash = hashlib.sha256(payload).hexdigest()
    return timestamp, ledger_hash


def analyze_audio(file_path: str) -> dict:
    """Batch file analyzer retained for testing static audio files."""
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    y = waveform.squeeze(0).numpy().astype(np.float32)
    windows = make_windows_batched(y)
    ai_probabilities = predict_batch(windows)

    ai_probability = float(np.mean(ai_probabilities))
    stability = float(np.std(ai_probabilities))
    risk_score = round(ai_probability * 100, 2)

    if risk_score >= 70:
        status = "ALERT: AI Voice Detected"
    elif risk_score >= 40:
        status = "INCONCLUSIVE: Review Required"
    else:
        status = "VERIFIED: Human Voice"

    timestamp, ledger_hash = make_ledger_hash("session_file", risk_score)

    return {
        "risk_score": risk_score,
        "status": status,
        "timestamp": timestamp,
        "ledger_hash": ledger_hash,
        "ai_probability": risk_score,
        "human_probability": round((1 - ai_probability) * 100, 2),
        "windows_analyzed": len(windows),
        "stability": round(stability * 100, 2),
    }