import streamlit as st
import requests

APIFY_ACTOR = "philip.boyedoku~audio-summariser-apify-deployment" 
APIFY_TOKEN = st.secrets["APIFY_TOKEN"]

st.title("Audio Transcription & Summary")

audio_url = st.text_input("Paste audio URL")
task = st.selectbox("Task", ["summary", "repurpose", "transcript"])

if st.button("Run"):
    if not audio_url:
        st.error("Audio URL required")
    else:
        with st.spinner("Processing..."):
            response = requests.post(
                f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/runs",
                params={"waitForFinish": 180},
                headers={
                    "Authorization": f"Bearer {APIFY_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "audio_url": audio_url,
                    "task": task,
                },
                timeout=300,
            )

        if response.ok:
            data = response.json()["data"]["output"]
            st.subheader("Transcript")
            st.write(data["transcript"])
            st.subheader("Result")
            st.write(data["result"])
        else:

            st.error("Actor execution failed")
