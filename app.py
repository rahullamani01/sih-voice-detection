import streamlit as st
import tempfile
from detector import analyze_audio

st.set_page_config(
    page_title="SIH 2026 - Voice Detection",
    layout="centered"
)

st.title("🛡️ AI Voice Cloning Detection System")
st.caption(
    "SIH Problem Statement ID: SIH26104 | "
    "Theme: Blockchain & Cybersecurity"
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button(
        "Analyze Audio",
        type="primary"
    ):
        with st.spinner(
            "Analyzing audio with AI deepfake detector..."
        ):
            suffix = "." + uploaded_file.name.split(".")[-1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp_file:
                tmp_file.write(
                    uploaded_file.getbuffer()
                )
                tmp_path = tmp_file.name

            try:
                res = analyze_audio(tmp_path)

                st.divider()

                col1, col2 = st.columns(2)

                col1.metric(
                    "AI Probability",
                    f"{res['ai_probability']}%"
                )

                col2.metric(
                    "Human Probability",
                    f"{res['human_probability']}%"
                )

                if res["risk_score"] >= 70:
                    st.error(
                        "🚨 LIKELY AI-GENERATED VOICE"
                    )
                elif res["risk_score"] >= 40:
                    st.warning(
                        "⚠️ INCONCLUSIVE — REVIEW REQUIRED"
                    )
                else:
                    st.success(
                        "✅ LIKELY HUMAN VOICE"
                    )

                st.write(
                    f"**Detection Status:** {res['status']}"
                )

                st.write(
                    f"**Windows Analyzed:** "
                    f"{res['windows_analyzed']}"
                )

                st.write(
                    f"**Prediction Stability:** "
                    f"{res['stability']}%"
                )

                st.divider()

                st.subheader(
                    "🔒 Immutable Ledger Audit Log"
                )

                st.code(
                    f"""
Timestamp: {res['timestamp']}
Session ID: User_Call_Stream_9823
AI Probability: {res['ai_probability']}%
Human Probability: {res['human_probability']}%
Risk Score: {res['risk_score']}%
SHA256 Hash: {res['ledger_hash']}
State: COMMITTED TO LEDGER
""",
                    language="text"
                )

            except Exception as e:
                st.error(
                    f"Detection failed: {str(e)}"
                )

st.sidebar.markdown("**Team Details:**")
st.sidebar.markdown("* Category: Software")
st.sidebar.markdown("* Track: Blockchain & Cybersecurity")