import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Voice Clone Detection", layout="wide")
st.title("🛡️ Real-Time AI Voice Clone Detection")

# Pure HTML/JS WebSocket interface (No external Python dependencies needed)
components.html(
    """
    <div style="font-family: system-ui, -apple-system, sans-serif; padding: 24px; background: #1a1a1e; color: #fff; border-radius: 12px; border: 1px solid #333;">
        <h3 style="margin-top: 0;">Live Audio Detector Streamer</h3>
        <p id="status" style="color: #aaa;">Status: Disconnected</p>
        
        <button id="toggleBtn" style="padding: 12px 24px; font-size: 16px; font-weight: bold; background: #2563eb; color: #fff; border: none; border-radius: 8px; cursor: pointer; transition: 0.2s;">
            Start Microphone
        </button>

        <div style="margin-top: 24px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div style="background: #26262c; padding: 16px; border-radius: 8px;">
                <p style="margin: 0; color: #888; font-size: 14px;">Risk Score</p>
                <h2 style="margin: 8px 0 0 0;"><span id="risk">0</span>%</h2>
            </div>
            <div style="background: #26262c; padding: 16px; border-radius: 8px;">
                <p style="margin: 0; color: #888; font-size: 14px;">Latency</p>
                <h2 style="margin: 8px 0 0 0;"><span id="latency">0</span> ms</h2>
            </div>
            <div style="background: #26262c; padding: 16px; border-radius: 8px;">
                <p style="margin: 0; color: #888; font-size: 14px;">Status</p>
                <h4 style="margin: 8px 0 0 0;" id="risk-status">IDLE</h4>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let audioContext;
        let mediaStream;
        let processor;
        const btn = document.getElementById('toggleBtn');
        const status = document.getElementById('status');

        btn.onclick = async () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
                if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
                btn.innerText = "Start Microphone";
                btn.style.background = "#2563eb";
                status.innerText = "Status: Stopped";
                return;
            }

            try {
                ws = new WebSocket("ws://localhost:8000/ws/live");

                ws.onopen = () => {
                    status.innerText = "Status: Connected & Streaming";
                    btn.innerText = "Stop Microphone";
                    btn.style.background = "#dc2626";
                    startAudio();
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === "score") {
                        document.getElementById('risk').innerText = data.risk_score;
                        document.getElementById('latency').innerText = data.inference_ms;
                        document.getElementById('risk-status').innerText = data.risk_level;
                    }
                };

                ws.onerror = (err) => {
                    status.innerText = "Status: Connection Error (Ensure FastAPI backend is running on port 8000)";
                };
            } catch (e) {
                status.innerText = "Error: " + e.message;
            }
        };

        async function startAudio() {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(mediaStream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (e) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    ws.send(inputData.buffer);
                }
            };
        }
    </script>
    """,
    height=320,
)