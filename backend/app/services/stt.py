# stt.py — Speech-to-text using OpenAI Whisper (local model).
# Transcribes audio files uploaded via the /api/voice endpoint. Requires ffmpeg installed.

import tempfile

import whisper
from fastapi import UploadFile

from app.config import settings


def transcribe_audio(file: UploadFile) -> str:
    model = whisper.load_model(settings.whisper_model)
    suffix = file.filename[file.filename.rfind(".") :] if "." in file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        result = model.transcribe(tmp.name)
    return (result.get("text") or "").strip()
