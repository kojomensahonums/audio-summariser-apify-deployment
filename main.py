from apify import Actor
import asyncio
import torch
import requests
import os
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from pydub import AudioSegment
from urllib.parse import urlparse
import tempfile
from tempfile import NamedTemporaryFile
import os
import json

# -----------------------------
# Model loading (global)
# -----------------------------
device = "cpu"
torch_dtype = torch.float16

hf_token = os.environ.get("HF_TOKEN")
WHISPER_ID = "distil-whisper/distil-small.en"
LLM_ID = "microsoft/phi-2"

asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_ID,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(WHISPER_ID)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=25,
    batch_size=8,
    device=-1,
)

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", dtype=torch.float16)


# -----------------------------
# Helpers
# -----------------------------
def download_audio(url: str) -> str:
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


def transcribe(path: str) -> str:
    result = asr_pipeline(path)
    return result["text"]


def generate_llm(prompt: str, context: str) -> str:
    full_prompt = f"{context}\n\n{prompt}\n\nAnswer:"
    inputs = llm_tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    decoded = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(full_prompt):].strip()


def summarise(text: str) -> str:
    return generate_llm(
        f"Summarize the following transcript:\n{text}",
        "You are a helpful assistant.",
    )


def repurpose(text: str) -> str:
    return generate_llm(
        f"Create social media content from this transcript:\n{text}",
        "You are a creative assistant.",
    )


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
        # decode base64 to temp file
        import base64, tempfile, os
        audio_bytes = base64.b64decode(audio_b64)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        tmp_file.close()
        audio_path = tmp_file.name
    elif audio_url:
        audio_path = download_audio(audio_url)
    else:
        raise ValueError("No audio provided")

    transcript = transcribe(audio_path)
    if task == "summary":
        output = summarise(transcript)
    elif task == "repurpose":
        output = repurpose(transcript)
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

# This runs the async main function
if __name__ == "__main__":
    asyncio.run(main())














