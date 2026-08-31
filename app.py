import streamlit as st
import tempfile
from detector import analyze_audio

st.set_page_config(page_title="SIH 2026 - Voice Detection", layout="centered")

st.title("🛡️ AI Voice Cloning Detection System")
st.caption("SIH Problem Statement ID: SIH26104 | Theme: Blockchain & Cybersecurity")

st.divider()

uploaded_file = st.file_uploader("Choose an audio file (.wav / .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Analyze Audio Integrity", type="primary"):
        with st.spinner("Analyzing spectral features & pitch variance..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            res = analyze_audio(tmp_path)
            
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric(label="Calculated Risk Score", value=f"{res['risk_score']}%")
            col2.metric(label="Authentication Status", value=res['status'])
            
            if res['risk_score'] > 50:
                st.error("🚨 HIGH RISK ALERT: Impersonation Attack Detected!")
            else:
                st.success("✅ LOW RISK: Genuine Human Speech")
                
            st.divider()
            st.subheader("🔒 Immutable Ledger Audit Log")
            st.code(f"""
Timestamp:   {res['timestamp']}
Session ID:  User_Call_Stream_9823
Risk Score:  {res['risk_score']}%
SHA256 Hash: {res['ledger_hash']}
State:       COMMITTED TO LEDGER
            """, language="json")

st.sidebar.markdown("**Team Details:**")
st.sidebar.markdown("* **Category:** Software")
st.sidebar.markdown("* **Track:** Blockchain & Cybersecurity")