import streamlit as st
import requests
import base64

APIFY_ACTOR = "philip.boyedoku~audio-summariser-apify-deployment"
APIFY_TOKEN = st.secrets["APIFY_TOKEN"]

st.title("Audio Transcription & Summary")

# Upload local file or use URL
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3","wav","m4a","ogg"])
audio_url = st.text_input("Or paste audio URL")
task = st.selectbox("Task", ["summary", "repurpose", "transcript"])

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
                dataset_resp = requests.get(
                    f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?clean=true",
                    headers={"Authorization": f"Bearer {APIFY_TOKEN}"}
                )

                if dataset_resp.ok:
                    dataset = dataset_resp.json()
                    if dataset:
                        result = dataset[0]
                        st.subheader("Transcript")
                        st.write(result.get("transcript"))
                        st.subheader("Result")
                        st.write(result.get("result"))
                    else:
                        st.warning("No output in dataset")
                else:
                    st.error("Failed to fetch dataset")
