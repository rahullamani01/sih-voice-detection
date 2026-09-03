# 🛡️ Real-Time AI Voice Cloning & Deepfake Detection System

> **SIH Problem Statement ID:** SIH26104  
> **Latency Performance:** Reduced end-to-end processing from >5s to **<50ms** via ONNX batched windowing and zero-copy ring buffering.

---

## 📸 Demo & Interface

| Live Streamlit UI | Fast-Inference WebSocket Backend |
| :---: | :---: |
| ![Dashboard](assets/dashboard.png) | ![Terminal Output](assets/terminal.png) |

---

## ⚡ Core Features
* **Sub-50ms Inference:** Vectorized `torchaudio` tensor conversion feeding directly into ONNX Runtime on CPU.
* **Full-Duplex WebSockets:** Asynchronous streaming backend built on FastAPI handling client audio chunks in real time.
* **In-Memory Buffering:** Zero-copy NumPy ring buffering eliminates disk I/O latency bottlenecks.
* **Audit Trail Ledger:** Hashes classification metrics with SHA-256 signatures per streaming window.

---

## 🏗️ Architecture & Tech Stack
* **Frontend:** Streamlit, JavaScript MediaDevices API
* **Backend:** FastAPI, Uvicorn, WebSockets
* **Machine Learning:** ONNX Runtime, PyTorch, TorchAudio, NumPy

---

## 🚦 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/rahullamani01/sih-voice-detection.git](https://github.com/rahullamani01/sih-voice-detection.git)
cd sih-voice-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
