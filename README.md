# 🛡️ AI Voice Cloning & Deepfake Detection System
**SIH Problem Statement ID:** SIH26104 | **Track:** Blockchain & Cybersecurity | **Category:** Software

An interactive security verification dashboard designed to detect synthetic voice impersonation attacks and generate immutable cryptographic audit trails in real time.

## 🚀 Key Features
* **Acoustic Integrity Analysis:** Evaluates audio streams (`.wav` / `.mp3`) for synthetic uniformity using spectral flatness and zero-crossing rate variance checks.
* **Real-Time Risk Scoring:** Calculates threat probability percentage and outputs categorical verification alerts.
* **Cryptographic Ledger Hash:** Generates SHA-256 transaction hashes ($\text{timestamp} + \text{session\_id} + \text{risk\_score}$) to provide a tamper-evident audit log without exposing raw acoustic data.
* **Streamlit Dashboard:** Lightweight web UI built for high-throughput stream processing.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Frontend UI:** Streamlit
* **Signal Processing:** Librosa, NumPy
* **Security & Cryptography:** Python `hashlib` (SHA-256 Engine)

## 🏃 Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rahullamani01/sih-voice-detection.git](https://github.com/rahullamani01/sih-voice-detection.git)
   cd sih-voice-detection
   
