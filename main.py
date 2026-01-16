from apify import Actor
import asyncio
import requests
import os
from pydub import AudioSegment
from urllib.parse import urlparse
import tempfile
from tempfile import NamedTemporaryFile
import json
import base64

# -----------------------------
# Helper functions
# -----------------------------

def download_audio(url: str) -> str:
    """
    Downloads an audio file from a URL and saves it to a temporary file.
    Returns the path to the temporary file.
    """

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    if not filename:
        raise ValueError("Invalid audio URL")

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
            return f.name

def transcribe_with_voxtral(audio_b64: str) -> str:
    """
    Transcribes audio using the Voxtral model via OpenRouter API.
    """

    payload = {
        "model": "mistralai/voxtral-small-24b-2507",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe this audio accurately."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ]
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                               "Content-Type": "application/json"},
                      data=json.dumps(payload))
    resp_json = r.json()
    return resp_json["choices"][0]["message"]["content"][0]["text"]


def llm_task_with_deepseek(transcript: str, task_type: str = "summary") -> str:
    """
    Performs a task (summary or copywriting) on the transcript using DeepSeek model via OpenRouter API.
    """

    if task_type == "summary":
        prompt = f"Summarize the following transcript:\n{transcript}"
    elif task_type == "copywrite":
        prompt = f"Create social media content from this transcript:\n{transcript}"

    payload = {
        "model": "deepseek/deepseek-chat-v3.1",
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                               "Content-Type": "application/json"},
                      data=json.dumps(payload))
    resp_json = r.json()
    msg_content = resp_json["choices"][0]["message"]["content"]
    if isinstance(msg_content, list):
        text = "".join([c.get("text", "") for c in msg_content])
    else:
        text = msg_content
    return text


# -----------------------------
# Actor entry point
# -----------------------------
# Get input
async def main():
    # Initialize Actor
    await Actor.init()

    # Get input safely inside async function
    input_data = await Actor.get_input() or {}
    Actor.log.info(f"Received input: {input_data}")

    # Your existing processing
    audio_url = input_data.get("audio_url")
    audio_b64 = input_data.get("audio_b64")
    task = input_data.get("task", "summary")

    if audio_b64:
        # Decode the uploaded audio bytes
        audio_bytes = base64.b64decode(audio_b64)

        # Write to a temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        tmp_file.close()
        audio_path = tmp_file.name
    elif audio_url:
        audio_path = download_audio(audio_url)
    else:
        raise ValueError("No audio provided")

    # Convert to WAV (for Voxtral) no matter what
    audio_segment = AudioSegment.from_file(audio_path)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_segment.export(wav_file.name, format="wav")
    audio_path = wav_file.name

    # Read WAV and encode to base64 for Voxtral
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64_for_voxtral = base64.b64encode(audio_bytes).decode("utf-8")

    transcript = transcribe_with_voxtral(audio_b64_for_voxtral)
    if task == "summary":
        output = llm_task_with_deepseek(transcript, "summary")
    elif task == "copywrite":
        output = llm_task_with_deepseek(transcript, "copywrite")
    else:
        output = transcript

    await Actor.push_data({
        "transcript": transcript,
        "result": output,
        "task": task
    })

    # Clean up temp file
    if audio_b64 and os.path.exists(audio_path):
        os.remove(audio_path)
    if not audio_b64 and audio_url and os.path.exists(audio_path):
        os.remove(audio_path)


# This runs the async main function
if __name__ == "__main__":
    asyncio.run(main())
