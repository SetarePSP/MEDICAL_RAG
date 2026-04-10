# supabase_search.py — Hybrid RAG search combining structured filters with vector similarity.
# Tries: RPC with exact filters → specialty variants → semantic-only → table fallback.
# Specialty alias groups ensure "neurology" matches "Neurologist" in the database.

import logging
from typing import Any

from app.services.embeddings import embed_query, embed_semantic_query
from app.services.supabase_client import get_supabase_client

log = logging.getLogger("uvicorn.error")


_SPECIALTY_GROUPS: list[list[str]] = [
    ["general medicine", "general practitioner", "family doctor", "medicina generale"],
    ["physiotherapy", "physiotherapist", "fisioterapia", "fisioterapista"],
    ["psychiatry", "psychiatrist", "psichiatria", "psichiatra"],
    ["neurology", "neurologist", "neurologia", "neurologo"],
    ["cardiology", "cardiologist", "cardiologia", "cardiologo"],
    ["dermatology", "dermatologist", "dermatologia", "dermatologo"],
    ["orthopedics", "orthopedic surgeon", "orthopedic", "ortopedia", "ortopedico"],
    ["gynecology", "gynecologist", "ginecologia", "ginecologo"],
    ["pediatrics", "pediatrician", "pediatria", "pediatra"],
    ["pulmonology", "pulmonologist", "pneumologia", "pneumologo"],
    ["endocrinology", "endocrinologist", "endocrinologia", "endocrinologo"],
    ["urology", "urologist", "urologia", "urologo"],
    ["rheumatology", "rheumatologist", "reumatologia", "reumatologo"],
    ["psychology", "psychologist", "psicologia", "psicologo"],
    ["geriatrics", "geriatrician", "geriatria", "geriatra"],
    ["nursing", "nurse", "infermiere", "infermiera"],
    ["occupational therapy", "occupational therapist", "terapia occupazionale"],
    ["speech therapy", "speech therapist", "logopedia", "logopedista"],
    ["dietetics", "dietitian", "dietologia", "nutrizionista", "nutrition"],
]


def _specialty_variants(value: str | None) -> list[str]:
    base = str(value or "").strip().lower()
    if not base:
        return []
    for group in _SPECIALTY_GROUPS:
        if base in group or any(base in v or v in base for v in group):
            return group
    return [base]


def _table_fallback(client: Any, entities: dict[str, Any]) -> list[dict[str, Any]]:
    """Fallback query using table select + ilike for maximum tolerance."""
    query = client.table("professionals").select("*")
    if entities.get("city"):
        city = str(entities["city"]).strip()
        query = query.ilike("city", f"%{city}%")
    if entities.get("specialty"):
        variants = _specialty_variants(entities.get("specialty"))
        if variants:
            or_clause = ",".join([f"specialty.ilike.%{v}%" for v in variants])
            query = query.or_(or_clause)
    filtered = query.limit(5).execute().data or []
    if filtered:
        log.info(
            "table_fallback: found %d results (city=%s, specialty_variants=%s)",
            len(filtered),
            entities.get("city"),
            variants if entities.get("specialty") else [],
        )
        return filtered
    # Drop specialty filter, keep city.
    query2 = client.table("professionals").select("*")
    if entities.get("city"):
        query2 = query2.ilike("city", f"%{str(entities['city']).strip()}%")
    city_only = query2.limit(5).execute().data or []
    if city_only:
        log.info("table_fallback: found %d results city-only (city=%s)", len(city_only), entities.get("city"))
        return city_only
    everything = client.table("professionals").select("*").limit(5).execute().data or []
    log.info("table_fallback: returning %d results unfiltered", len(everything))
    return everything


def hybrid_match(entities: dict[str, Any], user_message: str = "") -> list[dict[str, Any]]:
    client = get_supabase_client()
    if client is None:
        return [
            {
                "id": "mock-1",
                "name": "Dr.ssa Giulia Bianchi",
                "specialty": entities.get("specialty", "fisioterapia"),
                "city": entities.get("city", "Padova"),
                "score": 0.92,
                "supports_weight_kg": entities.get("weight_support_kg", 90),
            }
        ]

    specialty_options = _specialty_variants(entities.get("specialty"))
    payload = {
        "p_city": entities.get("city"),
        "p_specialty": specialty_options[0] if specialty_options else entities.get("specialty"),
        "p_gender": entities.get("gender_preference"),
        "p_min_weight_kg": entities.get("weight_support_kg"),
        "p_query_embedding": embed_query(entities, user_message=user_message),
        "p_limit": 5,
    }
    try:
        data = client.rpc("match_professionals_hybrid", payload).execute().data or []
        log.info("rpc: got %d rows (city=%s, specialty=%s)", len(data), payload["p_city"], payload["p_specialty"])
        if data:
            return data

        for variant in specialty_options[1:]:
            alt_payload = dict(payload)
            alt_payload["p_specialty"] = variant
            alt_data = client.rpc("match_professionals_hybrid", alt_payload).execute().data or []
            log.info("rpc variant: got %d rows (specialty=%s)", len(alt_data), variant)
            if alt_data:
                return alt_data

        if entities.get("city") and user_message.strip():
            semantic_payload = {
                "p_city": None,
                "p_specialty": None,
                "p_gender": None,
                "p_min_weight_kg": None,
                "p_query_embedding": embed_semantic_query(user_message),
                "p_limit": 5,
            }
            semantic_data = client.rpc("match_professionals_hybrid", semantic_payload).execute().data or []
            log.info("rpc semantic: got %d rows", len(semantic_data))
            if semantic_data:
                return semantic_data

        log.info("rpc: all paths empty, falling through to table fallback")
        return _table_fallback(client, entities)
    except Exception as exc:
        log.warning("rpc failed (%s), using table fallback", exc)
        return _table_fallback(client, entities)
