import hashlib
import time
import os
import librosa
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/best_model.onnx"
SAMPLE_RATE = 16000
WINDOW_SIZE = 48000
HOP_SIZE = 24000

_session = None

def get_session():
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Put best_model.onnx inside models/"
            )
        _session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
    return _session

def load_audio(file_path):
    y, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("No usable audio found.")

    y = y.astype(np.float32)

    return y

def make_windows(y):
    if len(y) <= WINDOW_SIZE:
        return [
            np.pad(
                y,
                (0, WINDOW_SIZE - len(y)),
                mode="constant"
            )
        ]

    windows = []

    for start in range(
        0,
        len(y) - WINDOW_SIZE + 1,
        HOP_SIZE
    ):
        windows.append(
            y[start:start + WINDOW_SIZE]
        )

    if len(y) > WINDOW_SIZE:
        last = y[-WINDOW_SIZE:]

        if not np.array_equal(windows[-1], last):
            windows.append(last)

    return windows

def predict(window):
    session = get_session()

    window = (
        window - np.mean(window)
    ) / np.sqrt(
        np.var(window) + 1e-5
    )

    window = window.astype(
        np.float32
    ).reshape(
        1,
        WINDOW_SIZE
    )

    input_name = session.get_inputs()[0].name

    logits = session.run(
        None,
        {input_name: window}
    )[0]

    logits = np.asarray(logits)

    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    probabilities = np.exp(logits)

    probabilities /= np.sum(
        probabilities,
        axis=1,
        keepdims=True
    )

    return float(
        probabilities[0][1]
    )

def analyze_audio(file_path):
    y = load_audio(file_path)

    windows = make_windows(y)

    predictions = [
        predict(window)
        for window in windows
    ]

    ai_probability = float(
        np.mean(predictions)
    )

    stability = float(
        np.std(predictions)
    )

    risk_score = round(
        ai_probability * 100,
        2
    )

    if risk_score >= 70:
        status = "ALERT: AI Voice Detected"
    elif risk_score >= 40:
        status = "INCONCLUSIVE: Review Required"
    else:
        status = "VERIFIED: Human Voice"

    timestamp = str(int(time.time()))

    audit_payload = (
        f"{timestamp}:"
        f"user_session_01:"
        f"{risk_score:.2f}"
    ).encode("utf-8")

    ledger_hash = hashlib.sha256(
        audit_payload
    ).hexdigest()

    return {
        "risk_score": risk_score,
        "status": status,
        "timestamp": timestamp,
        "ledger_hash": ledger_hash,
        "ai_probability": round(
            ai_probability * 100,
            2
        ),
        "human_probability": round(
            (1 - ai_probability) * 100,
            2
        ),
        "windows_analyzed": len(windows),
        "stability": round(
            stability * 100,
            2
        )
    }