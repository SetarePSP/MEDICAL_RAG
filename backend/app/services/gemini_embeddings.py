# gemini_embeddings.py — Vector embedding generation via Gemini API key, REST, or Vertex AI.
# Supports multiple backends (auto/gemini/rest/vertex) with automatic fallback.
# Used by ingest_embeddings.py (indexing) and embeddings.py (search queries).

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from app.config import settings

# Lazy SDK import avoids FutureWarning when ingest runs with backend=vertex only.
DEFAULT_EMBEDDING_MODELS = (
    "models/text-embedding-004",
    "text-embedding-004",
)

_vertex_init_key: tuple[str, str] | None = None


def _normalize_model(name: str) -> list[str]:
    name = (name or "").strip()
    if not name:
        return list(DEFAULT_EMBEDDING_MODELS)
    candidates = [name]
    if not name.startswith("models/") and "text-embedding" in name:
        candidates.append(f"models/{name}")
    for d in DEFAULT_EMBEDDING_MODELS:
        if d not in candidates:
            candidates.append(d)
    return candidates


def _vertex_project() -> str:
    return (
        (settings.google_cloud_project or "").strip()
        or (os.getenv("VERTEX_AI_PROJECT") or "").strip()
        or (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
        or (os.getenv("GCP_PROJECT") or "").strip()
    )


def _vertex_location() -> str:
    return (
        (os.getenv("VERTEX_AI_LOCATION") or "").strip()
        or (os.getenv("VERTEX_LOCATION") or "").strip()
        or (settings.vertex_ai_location or "us-central1").strip()
    )


def _embedding_backend() -> str:
    return (os.getenv("GEMINI_EMBEDDING_BACKEND") or settings.gemini_embedding_backend or "auto").strip().lower()


def _vertex_failure_hint(exc: Exception) -> str:
    """Short guidance when Vertex returns 403 SERVICE_DISABLED (API not enabled / billing)."""
    msg = str(exc)
    if "SERVICE_DISABLED" not in msg and "has not been used in project" not in msg:
        return "Check billing on the GCP project and Vertex AI API quota."
    proj = _vertex_project() or "YOUR_PROJECT_ID"
    m = re.search(r"project ([a-z0-9-]+) before", msg, re.I)
    if m:
        proj = m.group(1)
    return (
        f"Enable Vertex AI API for project {proj!r} (Billing must be on the project):\n"
        f"  https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project={proj}\n"
        "Click Enable, wait 2–5 minutes, then rerun ingest."
    )


def _blocked_message(exc: Exception) -> str:
    msg = str(exc)
    if "API_KEY_SERVICE_BLOCKED" in msg or "SERVICE_BLOCKED" in msg:
        return (
            "Gemini embedding is blocked for this API key (API_KEY_SERVICE_BLOCKED).\n"
            "Use Vertex instead: set GEMINI_EMBEDDING_BACKEND=vertex, GOOGLE_CLOUD_PROJECT=..., "
            "run `gcloud auth application-default login`, enable Vertex AI API, "
            "and `pip install google-cloud-aiplatform`.\n"
        )
    return msg


def _embed_gemini_sdk(text: str, task_type: str, models_to_try: list[str]) -> list[float]:
    import google.generativeai as genai  # noqa: PLC0415

    if not settings.gemini_api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    genai.configure(api_key=settings.gemini_api_key.strip())
    last_err: Exception | None = None
    for model_name in models_to_try:
        try:
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type=task_type,
            )
            embedding = result.get("embedding")
            if embedding:
                return list(embedding)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(_blocked_message(last_err) if last_err else "Embedding failed")


def _embed_gemini_rest(text: str, task_type: str, models_to_try: list[str]) -> list[float]:
    key = (settings.gemini_api_key or "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY for REST embedding")
    task_map = {
        "retrieval_document": "RETRIEVAL_DOCUMENT",
        "retrieval_query": "RETRIEVAL_QUERY",
    }
    rest_task = task_map.get(task_type.lower(), "RETRIEVAL_DOCUMENT")

    last_err: Exception | None = None
    for model_name in models_to_try:
        resource = model_name if model_name.startswith("models/") else f"models/{model_name}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{resource}:embedContent"
        payload: dict[str, Any] = {
            "content": {"parts": [{"text": text}]},
            "taskType": rest_task,
        }
        try:
            resp = requests.post(
                url,
                params={"key": key},
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60,
            )
            if resp.status_code != 200:
                last_err = RuntimeError(f"{resp.status_code} {resp.text[:500]}")
                continue
            data = resp.json()
            emb = data.get("embedding") or {}
            values = emb.get("values")
            if values:
                return [float(x) for x in values]
            last_err = RuntimeError("Empty embedding in REST response")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(_blocked_message(last_err) if last_err else "REST embedding failed")


def _embed_vertex(text: str, task_type: str) -> list[float]:
    global _vertex_init_key  # noqa: PLW0603

    try:
        import vertexai  # type: ignore
        from vertexai.language_models import TextEmbeddingModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Vertex embeddings require: pip install google-cloud-aiplatform\n"
            "Then: gcloud auth application-default login"
        ) from exc

    project = _vertex_project()
    location = _vertex_location()
    if not project:
        raise RuntimeError(
            "Vertex embeddings need GOOGLE_CLOUD_PROJECT (or google_cloud_project in .env / VERTEX_AI_PROJECT)."
        )

    model_id = (settings.vertex_embedding_model or "text-embedding-004").strip()

    init_key = (project, location)
    if _vertex_init_key != init_key:
        try:
            vertexai.init(project=project, location=location)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"vertexai.init failed for project={project!r} location={location!r}: {exc}\n"
                "Enable 'Vertex AI API' on this GCP project and ensure ADC is valid "
                "(`gcloud auth application-default login`)."
            ) from exc
        _vertex_init_key = init_key

    try:
        model = TextEmbeddingModel.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load Vertex embedding model {model_id!r} in {location!r}: {exc}\n"
            "Try another region (e.g. VERTEX_AI_LOCATION=us-central1 or europe-west1) "
            "where text-embedding-004 is available."
        ) from exc

    _ = task_type
    try:
        outs = model.get_embeddings([text])
    except Exception as exc:  # noqa: BLE001
        hint = _vertex_failure_hint(exc)
        brief = str(exc)
        if len(brief) > 400:
            brief = brief[:400] + "…"
        raise RuntimeError(
            f"Vertex get_embeddings failed: {brief}\n{hint}"
        ) from exc

    if not outs or not getattr(outs[0], "values", None):
        raise RuntimeError("Vertex returned empty embedding")
    return [float(x) for x in outs[0].values]


def embed_document(text: str) -> list[float]:
    """Embedding for indexing documents (professionals)."""
    return embed_text(text, task_type="retrieval_document")


def embed_query_text(text: str) -> list[float] | None:
    """Embedding for search queries."""
    if not settings.gemini_api_key and not _vertex_project():
        return None
    try:
        return embed_text(text, task_type="retrieval_query")
    except Exception:
        return None


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    text = text.strip()
    if not text:
        raise ValueError("Empty text for embedding")

    backend = _embedding_backend()
    models_to_try = _normalize_model(settings.gemini_embedding_model)
    vertex_first = os.getenv("EMBEDDING_VERTEX_FIRST", "").strip() in ("1", "true", "yes")

    if backend == "vertex":
        return _embed_vertex(text, task_type)
    if backend == "gemini":
        return _embed_gemini_sdk(text, task_type, models_to_try)
    if backend == "rest":
        return _embed_gemini_rest(text, task_type, models_to_try)

    # auto: try Vertex first only if explicitly requested (Option B)
    if vertex_first and _vertex_project():
        try:
            return _embed_vertex(text, task_type)
        except Exception:
            pass

    try:
        return _embed_gemini_sdk(text, task_type, models_to_try)
    except Exception as sdk_exc:
        try:
            return _embed_gemini_rest(text, task_type, models_to_try)
        except Exception:
            raise sdk_exc
