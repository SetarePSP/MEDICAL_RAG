# stt.py — Speech-to-text using OpenAI Whisper (local model).
# Transcribes audio files uploaded via the /api/voice endpoint. Requires ffmpeg installed.
# Model is loaded once and reused (Cloud Run: use WHISPER_MODEL=tiny and ≥2Gi memory).

import logging
import tempfile
import threading

import whisper
from fastapi import UploadFile

from app.config import settings

_lock = threading.Lock()
_model: whisper.Whisper | None = None


def _get_model() -> whisper.Whisper:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                logging.getLogger("uvicorn.error").info(
                    "Loading Whisper model=%s (first voice request may be slow)",
                    settings.whisper_model,
                )
                _model = whisper.load_model(settings.whisper_model)
    return _model


def transcribe_audio(file: UploadFile) -> str:
    model = _get_model()
    suffix = file.filename[file.filename.rfind(".") :] if "." in file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        kwargs: dict = {}
        if settings.whisper_language.strip():
            kwargs["language"] = settings.whisper_language.strip().lower()[:8]
        if (settings.whisper_task or "").strip().lower() == "translate":
            kwargs["task"] = "translate"
        result = model.transcribe(tmp.name, **kwargs)
    return (result.get("text") or "").strip()
