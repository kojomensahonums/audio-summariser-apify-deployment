import streamlit as st
import requests
import base64
import time

APIFY_ACTOR = "philip.boyedoku~audio-summariser-apify-deployment"
APIFY_TOKEN = st.secrets["APIFY_TOKEN"]

st.title("Audio Transcription & Summary")

# Upload local file or use URL
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3","wav","m4a","ogg"])
audio_url = st.text_input("Or paste audio URL")
task = st.selectbox("Task", ["summary", "copywrite", "transcript"])

if st.button("Run"):
    if not uploaded_file and not audio_url:
        st.error("Please provide a local file or a URL")
    else:
        with st.spinner("Processing..."):
            # Prepare payload
            payload = {"task": task}
            
            if uploaded_file:
                # Encode local file in base64
                file_bytes = uploaded_file.read()
                payload["audio_b64"] = base64.b64encode(file_bytes).decode("utf-8")
            else:
                payload["audio_url"] = audio_url

            # Trigger actor
            resp = requests.post(
                f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/runs?waitForFinish=300",
                headers={
                    "Authorization": f"Bearer {APIFY_TOKEN}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            if not resp.ok:
                st.error("Actor execution failed")
            else:
                run_id = resp.json()["data"]["id"]
                # Fetch dataset output
                while True:
                    status_resp = requests.get(
                        # f"https://api.apify.com/v2/actor-runs/{run_id}",
                        f"https://api.apify.com/v2/key-value-stores/actor-default/records/OUTPUT",
                        headers={"Authorization": f"Bearer {APIFY_TOKEN}"}
                    )
                    status = status_resp.json()["data"]["status"]
                
                    if status == "SUCCEEDED":
                        break
                    elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                        st.error(f"Actor failed with status: {status}")
                        st.stop()
                
                    time.sleep(3)
                
                # Fetch Actor output (NOT dataset)
                output_resp = requests.get(
                    f"https://api.apify.com/v2/actor-runs/{run_id}/output",
                    headers={"Authorization": f"Bearer {APIFY_TOKEN}"}
                )
                
                if output_resp.ok:
                    data = output_resp.json()
                    st.subheader("Transcript")
                    st.write(data.get("transcript"))
                    st.subheader("Result")
                    st.write(data.get("result"))
                else:
                    st.error("Failed to fetch actor output")


