# graph.py — LangGraph state machine that orchestrates the medical intake conversation.
# Nodes: triage_extract (LLM analysis) → emergency | ask_missing | match (provider search).
# Handles entity merging, specialty alias resolution, multi-pass search with auto-broadening.

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.services.google_places import enrich_with_places
from app.services.llm import (
    analyze_intake,
    extract_entities_from_text,
    rerank_candidates_by_symptoms,
)
from app.services.supabase_search import hybrid_match


class IntakeState(TypedDict):
    user_message: str
    known_entities: dict[str, Any]
    transcript: list[dict[str, str]]
    is_emergency: bool
    emergency_reason: str
    entities: dict[str, Any]
    missing_fields: list[str]
    reply: str
    candidates: list[dict[str, Any]]
    status: str
    decision_mode: str
    decision_reason: str
    ready_to_search: bool


def _merge_non_empty(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for k, v in (updates or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        base[k] = v
    return base


def _build_conversation_context(transcript: list[dict[str, str]], user_message: str) -> str:
    user_turns = [t.get("content", "").strip() for t in transcript if t.get("role") == "user" and t.get("content")]
    user_turns.append((user_message or "").strip())
    recent = [t for t in user_turns if t][-6:]
    return " | ".join(recent)


def _extract_acknowledged_entities(text: str) -> dict[str, Any]:
    """Force state sync from assistant acknowledgements like 'you're 28' / 'you're in Milano'."""
    out: dict[str, Any] = {}
    lowered = (text or "").lower()
    age_match = re.search(r"\b(?:you(?:'re|\s+are)|patient(?:\s+is)?)\s+(\d{1,3})\b", lowered)
    if age_match:
        age = int(age_match.group(1))
        if 0 < age < 120:
            out["age"] = age
    city_match = re.search(r"\b(?:you(?:'re|\s+are)\s+in)\s+([a-z][a-z\s'-]{1,40})\b", lowered)
    if city_match:
        raw_city = city_match.group(1).strip(" .,!?:;")
        normalized = _normalize_city_for_match(raw_city)
        if normalized:
            out["city"] = normalized.title()
    if any(k in lowered for k in ("depress", "anxiety", "panic", "sad", "mental health")):
        out["chief_complaint"] = "depression and mental health symptoms"
        if "symptom_context" not in out:
            out["symptom_context"] = text.strip()
    return out


def _normalize_city_for_match(city: str) -> str:
    c = (city or "").strip().lower()
    aliases = {
        "milan": "milano",
        "mi": "milano",
        "rome": "roma",
        "naples": "napoli",
        "florence": "firenze",
        "turin": "torino",
    }
    return aliases.get(c, c)


def _is_same_city(candidate_city: str, requested_city: str) -> bool:
    cand = _normalize_city_for_match(candidate_city)
    req = _normalize_city_for_match(requested_city)
    return bool(cand and req and (cand == req or cand in req or req in cand))


SPECIALTY_ALIAS_GROUPS: list[set[str]] = [
    {"general medicine", "general practitioner", "family doctor", "medicina generale", "gp"},
    {"physiotherapy", "physiotherapist", "fisioterapia", "fisioterapista", "physical therapy"},
    {"psychiatry", "psychiatrist", "psichiatria", "psichiatra"},
    {"neurology", "neurologist", "neurologia", "neurologo"},
    {"cardiology", "cardiologist", "cardiologia", "cardiologo"},
    {"dermatology", "dermatologist", "dermatologia", "dermatologo"},
    {"orthopedics", "orthopedic surgeon", "ortopedia", "orthopedic", "ortopedico"},
    {"gynecology", "gynecologist", "ginecologia", "ginecologo"},
    {"pediatrics", "pediatrician", "pediatria", "pediatra"},
    {"pulmonology", "pulmonologist", "pneumologia", "pneumologo"},
    {"endocrinology", "endocrinologist", "endocrinologia", "endocrinologo"},
    {"urology", "urologist", "urologia", "urologo"},
    {"rheumatology", "rheumatologist", "reumatologia", "reumatologo"},
    {"psychology", "psychologist", "psicologia", "psicologo"},
    {"geriatrics", "geriatrician", "geriatria", "geriatra"},
    {"nursing", "nurse", "infermiere", "infermiera"},
    {"occupational therapy", "occupational therapist", "terapia occupazionale"},
    {"speech therapy", "speech therapist", "logopedia", "logopedista"},
    {"dietetics", "dietitian", "dietologia", "nutrizionista", "nutrition"},
]


def _is_same_specialty(a: str, b: str) -> bool:
    la, lb = a.strip().lower(), b.strip().lower()
    if not la or not lb:
        return False
    if la in lb or lb in la:
        return True
    for group in SPECIALTY_ALIAS_GROUPS:
        if any(alias in la or la in alias for alias in group) and any(alias in lb or lb in alias for alias in group):
            return True
    return False


def _postfilter_and_rank(
    candidates: list[dict[str, Any]],
    entities: dict[str, Any],
    *,
    allow_other_cities: bool,
    allow_specialty_relax: bool = False,
) -> list[dict[str, Any]]:
    req_city = str(entities.get("city") or "")
    req_specialty = str(entities.get("specialty") or "").strip().lower()
    age = entities.get("age")
    symptom_text = (str(entities.get("symptom_context") or "") + " " + str(entities.get("conversation_context") or "")).lower()

    ranked: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        score = float(c.get("score") or 0.0)
        city = str(c.get("city") or "")
        specialty = str(c.get("specialty") or "").lower()

        in_city = _is_same_city(city, req_city) if req_city else True
        if req_city and not in_city and not allow_other_cities:
            continue
        if in_city:
            score += 10.0
        elif allow_other_cities:
            score -= 1.0

        if req_specialty:
            same_specialty = _is_same_specialty(req_specialty, specialty)
            if not same_specialty and not allow_specialty_relax:
                continue
            if same_specialty:
                score += 5.0
            else:
                score -= 3.0

        # Age appropriateness: downrank geriatric suggestions for younger patients.
        if isinstance(age, int) and age < 55 and "geriat" in specialty:
            score -= 8.0

        # Symptom intent preference for headache-type cases.
        if "headache" in symptom_text or "migraine" in symptom_text:
            if "neuro" in specialty:
                score += 4.0
            if "general" in specialty or "gp" in specialty:
                score += 3.0
            if "geriat" in specialty:
                score -= 4.0

        ranked.append((score, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked][:3]


def _suggest_related_specialty(entities: dict[str, Any]) -> str:
    text = (str(entities.get("symptom_context") or "") + " " + str(entities.get("conversation_context") or "")).lower()
    current = str(entities.get("specialty") or "").lower()
    if "depress" in text or "anxiety" in text or "panic" in text or "mental" in text:
        return "psychiatry" if current != "psychiatry" else "general medicine"
    if "headache" in text or "migraine" in text or "vision" in text:
        return "neurology" if current != "neurology" else "general medicine"
    if "back" in text or "neck" in text or "shoulder" in text:
        return "orthopedics" if current != "orthopedics" else "physiotherapy"
    if "broken" in text or "fracture" in text or "sprain" in text or "cannot walk" in text:
        return "orthopedics" if current != "orthopedics" else "physiotherapy"
    if current == "general medicine":
        return "neurology"
    return "general medicine"


def triage_and_extract(state: IntakeState) -> IntakeState:
    result = analyze_intake(
        state["user_message"],
        known_entities=state.get("known_entities", {}),
        transcript=state.get("transcript", []),
    )
    state["is_emergency"] = bool(result.get("is_emergency", False))
    state["emergency_reason"] = result.get("emergency_reason", "")
    merged_entities = dict(state.get("known_entities", {}))
    _merge_non_empty(merged_entities, dict(result.get("entities", {})))
    user_message = state.get("user_message", "")
    rejected_specialties = {
        str(x).strip().lower() for x in (merged_entities.get("rejected_specialties") or []) if str(x).strip()
    }
    if merged_entities.get("specialty") and str(merged_entities.get("specialty", "")).lower() in rejected_specialties:
        merged_entities.pop("specialty", None)
        merged_entities.pop("required_specialty", None)
    # Safety-net extraction from raw user text guarantees persistence even if LLM misses a slot.
    user_entities = extract_entities_from_text(user_message, known_entities=merged_entities)
    _merge_non_empty(merged_entities, user_entities)
    # Also recover entities from LLM natural-language reply when JSON extraction is weak.
    _merge_non_empty(
        merged_entities,
        extract_entities_from_text(str(result.get("assistant_reply", "")), known_entities=merged_entities),
    )
    _merge_non_empty(merged_entities, _extract_acknowledged_entities(str(result.get("assistant_reply", ""))))
    merged_entities["conversation_context"] = _build_conversation_context(
        state.get("transcript", []), state.get("user_message", "")
    )
    if result.get("chief_complaint"):
        merged_entities["chief_complaint"] = str(result.get("chief_complaint")).strip()
    elif user_entities.get("symptom_context") and not merged_entities.get("chief_complaint"):
        merged_entities["chief_complaint"] = str(user_entities.get("symptom_context", "")).strip()
    # Keep an explicit specialty field for downstream tools.
    if merged_entities.get("specialty") and not merged_entities.get("required_specialty"):
        merged_entities["required_specialty"] = merged_entities["specialty"]
    state["entities"] = merged_entities
    missing_fields = [str(f) for f in (result.get("missing_fields") or []) if not merged_entities.get(str(f))]
    state["missing_fields"] = missing_fields
    state["reply"] = str(result.get("assistant_reply", "")).strip()
    if not state["reply"]:
        state["reply"] = "Tell me a bit more so I can continue the medical interview."

    state["decision_mode"] = str(result.get("decision_mode", "fallback"))
    state["decision_reason"] = str(result.get("error", ""))
    # LLM is the state manager for interview completeness and transition to search.
    state["ready_to_search"] = bool(result.get("ready_to_search", False))
    return state


def route_after_triage(state: IntakeState) -> str:
    if state["is_emergency"]:
        return "emergency"
    if state.get("ready_to_search", False):
        return "match"
    return "ask_missing"


def emergency_response(state: IntakeState) -> IntakeState:
    state["status"] = "emergency"
    state["reply"] = (
        "Possible medical emergency detected. Call 112 immediately or go to the nearest emergency room."
    )
    return state


def ask_missing_info(state: IntakeState) -> IntakeState:
    state["status"] = "needs_info"
    # Trust LLM-generated assistant reply as the source of conversational behavior.
    if not state["reply"]:
        state["reply"] = "I still need some missing details: " + ", ".join(state["missing_fields"])
    return state


def run_match(state: IntakeState) -> IntakeState:
    user_message = (state.get("user_message") or "").lower()
    entities = dict(state["entities"])
    chief = str(entities.get("chief_complaint") or entities.get("symptom_context") or "").lower()
    if not entities.get("specialty") and any(x in chief for x in ("depress", "anxiety", "panic", "mental")):
        entities["specialty"] = "psychiatry"
        entities["required_specialty"] = "psychiatry"
        state["entities"] = dict(entities)
    retrieval_query = " | ".join(
        x
        for x in [
            str(state.get("user_message") or "").strip(),
            str(entities.get("chief_complaint") or "").strip(),
            str(entities.get("symptom_context") or "").strip(),
            str(entities.get("conversation_context") or "").strip(),
        ]
        if x
    )
    rejected_specialties = {
        str(x).strip().lower() for x in (entities.get("rejected_specialties") or []) if str(x).strip()
    }
    if entities.get("specialty") and str(entities.get("specialty", "")).lower() in rejected_specialties:
        entities.pop("specialty", None)
        entities.pop("required_specialty", None)
        state["entities"] = dict(entities)

    city_note = ""
    if entities.get("city_input_raw") and entities.get("city"):
        raw = str(entities.get("city_input_raw", "")).strip()
        norm = str(entities.get("city", "")).strip()
        if raw and norm and raw.lower() != norm.lower():
            city_note = f"I corrected '{raw}' to '{norm}'. "

    expand_tokens = ("expand", "nearby", "other cities", "all cities", "outside city")
    related_tokens = ("related specialty", "suggest related", "another one", "another specialty", "change specialty")
    explicit_expand = any(token in user_message for token in expand_tokens)
    wants_related_specialty = any(token in user_message for token in related_tokens)

    # If user asks for a related specialty after no-results, actively propose and wait for confirmation.
    if wants_related_specialty:
        suggested = _suggest_related_specialty(entities)
        entities["specialty"] = suggested
        entities["required_specialty"] = suggested
        entities["specialty_confirmed"] = False
        state["entities"] = entities
        state["candidates"] = []
        state["status"] = "needs_info"
        state["reply"] = (
            f"I suggest {suggested.replace('_', ' ')} as a related option. "
            "Do you want me to search with this specialty?"
        )
        return state

    # First pass: strict match with current constraints.
    candidates = hybrid_match(entities, user_message=retrieval_query)
    reranked = rerank_candidates_by_symptoms(candidates, entities, retrieval_query)
    filtered_ranked = _postfilter_and_rank(
        reranked,
        entities,
        allow_other_cities=False,
        allow_specialty_relax=False,
    )
    enriched = enrich_with_places(filtered_ranked)
    if enriched:
        state["candidates"] = enriched
        state["status"] = "matched"
        state["reply"] = f"{city_note}I found compatible providers. Select a time slot and proceed to payment."
        return state

    # Second pass: auto-broaden (RPC city=exact often fails for Milan/Milano mismatch).
    relaxed = dict(entities)
    relaxed["city"] = ""
    relaxed_candidates = hybrid_match(relaxed, user_message=retrieval_query)
    relaxed_ranked = rerank_candidates_by_symptoms(relaxed_candidates, relaxed, retrieval_query)
    relaxed_filtered = _postfilter_and_rank(
        relaxed_ranked,
        entities,
        allow_other_cities=True,
        allow_specialty_relax=False,
    )
    relaxed_enriched = enrich_with_places(relaxed_filtered)
    if relaxed_enriched:
        state["candidates"] = relaxed_enriched
        state["status"] = "matched"
        state["reply"] = f"{city_note}I found compatible providers. Select a time slot and proceed to payment."
        return state

    # Third pass: drop specialty filter too (pure semantic).
    semantic_relaxed = dict(entities)
    semantic_relaxed["specialty"] = ""
    semantic_relaxed["required_specialty"] = ""
    semantic_relaxed["city"] = ""
    semantic_candidates = hybrid_match(semantic_relaxed, user_message=retrieval_query)
    semantic_ranked = rerank_candidates_by_symptoms(semantic_candidates, semantic_relaxed, retrieval_query)
    semantic_filtered = _postfilter_and_rank(
        semantic_ranked,
        entities,
        allow_other_cities=True,
        allow_specialty_relax=True,
    )
    semantic_enriched = enrich_with_places(semantic_filtered)
    if semantic_enriched:
        state["candidates"] = semantic_enriched
        state["status"] = "matched"
        state["reply"] = (
            f"{city_note}I found providers based on your symptoms and preferences."
        )
        return state

    # Final fallback.
    state["candidates"] = []
    state["status"] = "needs_info"
    city = state["entities"].get("city")
    specialty = state["entities"].get("specialty")
    if city and specialty:
        state["reply"] = (
            f"I could not find available {specialty} providers in {city}. "
            "I can either expand to nearby cities or suggest a related specialty. Which do you prefer?"
        )
    else:
        state["reply"] = "I couldn't find a match yet. Share your city and symptoms, and I'll widen the search."
    return state


def build_graph():
    graph = StateGraph(IntakeState)
    graph.add_node("triage_extract", triage_and_extract)
    graph.add_node("emergency", emergency_response)
    graph.add_node("ask_missing", ask_missing_info)
    graph.add_node("match", run_match)

    graph.set_entry_point("triage_extract")
    graph.add_conditional_edges(
        "triage_extract",
        route_after_triage,
        {"emergency": "emergency", "ask_missing": "ask_missing", "match": "match"},
    )
    graph.add_edge("emergency", END)
    graph.add_edge("ask_missing", END)
    graph.add_edge("match", END)
    return graph.compile()


app_graph = build_graph()
