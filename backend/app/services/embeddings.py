# embeddings.py — Builds structured query text from patient entities and generates
# vector embeddings for hybrid search. Delegates to gemini_embeddings for the actual API call.

import os
from typing import Any

from app.config import settings
from app.services.gemini_embeddings import embed_query_text as _embed_query_text


def build_query_text(entities: dict[str, Any], user_message: str = "") -> str:
    parts = []
    if user_message.strip():
        parts.append(f"User request: {user_message.strip()}")
    if entities.get("specialty"):
        parts.append(f"Specialty: {entities['specialty']}")
    if entities.get("city"):
        parts.append(f"City: {entities['city']}, Italy")
    if entities.get("weight_support_kg"):
        parts.append(f"Minimum weight support: {entities['weight_support_kg']} kg")
    if entities.get("gender_preference"):
        parts.append(f"Gender preference: {entities['gender_preference']}")
    if entities.get("age"):
        parts.append(f"Patient age: {entities['age']}")
    if entities.get("chief_complaint"):
        parts.append(f"Chief complaint: {entities['chief_complaint']}")
    if entities.get("symptom_context"):
        parts.append(f"Symptoms: {entities['symptom_context']}")
    if entities.get("conversation_context"):
        parts.append(f"Conversation context: {entities['conversation_context']}")
    return " | ".join(parts)


def embed_query(entities: dict[str, Any], user_message: str = "") -> list[float] | None:
    if not settings.gemini_api_key and not (
        os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_AI_PROJECT")
    ):
        return None
    query_text = build_query_text(entities, user_message=user_message).strip()
    if not query_text:
        return None
    return _embed_query_text(query_text)


def embed_semantic_query(user_message: str) -> list[float] | None:
    """Pure semantic embedding based only on the user's raw request."""
    if not settings.gemini_api_key and not (
        os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_AI_PROJECT")
    ):
        return None
    text = (user_message or "").strip()
    if not text:
        return None
    return _embed_query_text(text)
