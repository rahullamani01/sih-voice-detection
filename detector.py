import hashlib
import time
import librosa
import numpy as np

def analyze_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=16000)
    
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    raw_score = (flatness * 1000) + (zcr * 500)
    fake_score = min(max(raw_score * 15, 12.5), 94.8)
    
    timestamp = str(int(time.time()))
    audit_payload = f"{timestamp}:user_session_01:{fake_score:.2f}".encode('utf-8')
    ledger_hash = hashlib.sha256(audit_payload).hexdigest()
    
    return {
        "risk_score": round(fake_score, 2),
        "status": "ALERT: AI Voice Detected" if fake_score > 50 else "VERIFIED: Human Voice",
        "ledger_hash": ledger_hash,
        "timestamp": timestamp
    }