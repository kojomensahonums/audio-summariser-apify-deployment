from apify import Actor
import asyncio
import torch
import requests
import os
from tempfile import NamedTemporaryFile
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from pydub import AudioSegment
from urllib.parse import urlparse


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
llm_model = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", torch_dtype=torch.float16)


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
async def main():
    input_data = await Actor.get_input() or {}

    audio_url = input_data.get("audio_url")
    task = input_data.get("task", "summary")

    if not audio_url:
        raise ValueError("audio_url is required")

    audio_path = download_audio(audio_url)
    transcript = transcribe(audio_path)

    if task == "summary":
        output = summarise(transcript)
    elif task == "repurpose":
        output = repurpose(transcript)
    else:
        output = transcript

    # Return output to Actor
    await Actor.set_output({
        "transcript": transcript,
        "result": output,
        "task": task,
    })

if __name__ == "__main__":
    asyncio.run(main())



