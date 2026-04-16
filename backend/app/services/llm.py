# llm.py — LLM-powered medical intake engine (Gemini via API key or Vertex AI).
# Handles: symptom analysis, entity extraction, specialty inference, emergency detection,
# city normalization, and candidate reranking. Falls back to rule-based logic if LLM is unavailable.

import json
import logging
import os
import re
from difflib import get_close_matches
from typing import Any

from app.config import settings

log = logging.getLogger("uvicorn.error")

_vertex_chat_initialized = False


def _init_vertex_chat() -> None:
    """Initialize Vertex AI SDK for chat (uses ADC, not API key)."""
    global _vertex_chat_initialized
    if _vertex_chat_initialized:
        return
    project = (
        (settings.google_cloud_project or "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1").strip()
    if project:
        import vertexai
        vertexai.init(project=project, location=location)
        _vertex_chat_initialized = True
        log.info("Vertex AI chat initialized (project=%s, location=%s)", project, location)


SYSTEM_PROMPT = """
You are a medically safe intake and triage assistant for booking home healthcare professionals in Italy.
Tasks:
1) Detect medical emergency risk.
2) Extract entities:
   - age
   - city (specific location in Italy)
   - specialty
   - weight_support_kg
   - gender_preference
3) Auto-map symptom descriptions to a likely specialty when possible.
4) Return strict JSON with keys:
{
  "is_emergency": bool,
  "emergency_reason": string,
  "entities": { ... },
  "missing_fields": [string],
  "assistant_reply": string,
  "chief_complaint": string,
  "ready_to_search": bool
}

Reasoning policy (internal, do not expose chain-of-thought):
- Build a brief internal hypothesis from symptoms.
- Ask only the single most useful next question per turn.
- Prefer clarifying questions before locking specialty when evidence is weak.
- Never ask again for fields already present in known entities.
- Set "ready_to_search": true ONLY when:
  1) city is known,
  2) specialty is explicitly proposed and confirmed by the patient,
  3) enough symptom detail exists for safe matching.
- For elderly/home-care scenarios, ask needed logistics before search (weight_support_kg, gender_preference).
- If no exact city match is likely, keep user informed and suggest nearby-city search as an explicit next step.
- You are the primary medical reasoning engine for symptom interpretation and specialty selection.
- Do not rely on keyword matching. Use clinical reasoning from the full conversation context.

Critical fields before booking search: age, city, specialty.
Conversation style requirements:
- If the user greets (e.g., "hi"), do not ask for all fields immediately.
- First ask what symptoms or care need they have.
- If user does not know specialty, infer and propose one from symptoms, then ask for confirmation.
- For a single vague symptom (e.g., just "headache"), ask 1-2 clarifying questions (duration, severity, red flags)
  before finalizing specialty.
- Only move to provider search after specialty is confirmed and city is known.
- Normalize city names to their canonical Italian spelling before saving entities (e.g., milan/miano -> Milano).
- Never ask again for values already present in known entities.
- Set "ready_to_search": true only when the case is ready for provider search.
 - Put specialty confirmation status into entities.specialty_confirmed as true/false.
If user shares symptoms, infer specialty using medical reasoning from context and ask clarifying questions as needed.
Context-first extraction policy:
- Extract and normalize ALL entities from the full conversation history in one pass (not only latest sentence).
- If a typo city appears (e.g., "Milno"), normalize to canonical form in entities.city (e.g., "Milano").
- If age/city/symptom is already present in history, do not ask it again.
- Fill "chief_complaint" immediately from the user's main symptom intent (e.g., depression, back pain, fracture).
Use conversational follow-ups, for example:
- "Got it, you're 28. Which city in Italy are you in so I can find someone nearby?"
- "I understand. I'll look for a General Practitioner for you in Milan. I just need a second to check providers."
Keep responses medically cautious.

Language policy (CRITICAL):
- Italy is only the geographic setting for providers. It does NOT mean you should answer in Italian.
- The conversation language is locked to how the user started: if their first substantive messages are in English,
  keep assistant_reply in English for the whole chat; if they started in Italian, keep Italian throughout.
- Do not switch languages mid-conversation (e.g. do not switch to Italian because you ask for an Italian city).
- Use English role names when the conversation is in English ("general practitioner" / "GP", not "Medico di Base").
- A separate instruction will state the mandatory reply language for this turn — follow it exactly.
"""


def _ordered_user_messages(transcript: list[dict[str, str]] | None, user_message: str) -> list[str]:
    """Chronological user texts; current turn is not yet in transcript when analyze_intake runs."""
    out: list[str] = []
    for t in transcript or []:
        if t.get("role") == "user":
            c = (t.get("content") or "").strip()
            if c:
                out.append(c)
    um = (user_message or "").strip()
    if um:
        out.append(um)
    return out


def _classify_snippet_language(text: str) -> str | None:
    """Return 'English', 'Italian', or None if too little signal."""
    t = (text or "").strip()
    if not t:
        return None
    if any(ch in t for ch in "àèéìòù"):
        return "Italian"
    if re.search(r"\b(salve|ciao|buongiorno|buonasera)\b", t, re.IGNORECASE):
        return "Italian"
    pad = f" {t.lower()} "
    italian_markers = (
        " ciao ",
        "perché",
        " grazie ",
        " sono ",
        " ho un ",
        " mi fa male ",
        " mal di testa",
        " medico di base",
        " dove sei",
        "in quale città",
        " quale città ",
        " mi trovo ",
        " sto a ",
        " buongiorno",
        " buonasera",
        " salve ",
    )
    if any(m in pad for m in italian_markers):
        return "Italian"
    english_markers = (
        " headache ",
        " severity ",
        " i have",
        " hi ",
        " hello ",
        " about ",
        " weeks",
        " pain ",
        " hurt ",
        " been ",
        " thank",
    )
    if any(m in pad for m in english_markers):
        return "English"
    if len(t) <= 6:
        low = t.lower().strip("!.?")
        if low in ("ciao", "salve"):
            return "Italian"
        if low in ("hi", "hey", "hello", "ok", "yes", "no", "thanks"):
            return "English"
    return None


def _conversation_reply_language(transcript: list[dict[str, str]] | None, user_message: str) -> str:
    """Lock assistant language to the user's opening language (English vs Italian)."""
    msgs = _ordered_user_messages(transcript, user_message)
    if not msgs:
        return "English"
    for snippet in msgs[:5]:
        lang = _classify_snippet_language(snippet)
        if lang is not None:
            return lang
    return "English"


def _model(model_name: str) -> Any:
    """Return a GenerativeModel via API-key SDK (preferred) or Vertex AI."""
    api_key = (settings.gemini_api_key or "").strip()
    if api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    project = (
        (settings.google_cloud_project or "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    if project:
        _init_vertex_chat()
        from vertexai.generative_models import GenerativeModel
        return GenerativeModel(model_name)
    raise RuntimeError("No GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT configured for LLM chat.")


SPECIALTY_KEYWORDS = {
    "dermatologist": "dermatology",
    "dermatology": "dermatology",
    "cardiologist": "cardiology",
    "cardiology": "cardiology",
    "neurologist": "neurology",
    "neurology": "neurology",
    "pediatrician": "pediatrics",
    "pediatrics": "pediatrics",
    "physiotherapist": "physiotherapy",
    "physiotherapy": "physiotherapy",
    "orthopedic": "orthopedics",
    "orthopedics": "orthopedics",
    "gynecologist": "gynecology",
    "gynecology": "gynecology",
    "general practitioner": "general medicine",
    "family doctor": "general medicine",
}

SYMPTOM_SPECIALTY_MAP = {
    "headache": "neurology",
    "migraine": "neurology",
    "fever": "general medicine",
    "cough": "general medicine",
    "cold": "general medicine",
    "sore throat": "general medicine",
    "back pain": "physiotherapy",
    "backache": "physiotherapy",
    "back ache": "physiotherapy",
    "bachache": "physiotherapy",
    "bach ache": "physiotherapy",
    "neck pain": "physiotherapy",
    "shoulder pain": "physiotherapy",
    "knee pain": "orthopedics",
    "joint pain": "orthopedics",
    "broken leg": "orthopedics",
    "leg broken": "orthopedics",
    "fracture": "orthopedics",
    "broken bone": "orthopedics",
    "sprain": "orthopedics",
    "ankle injury": "orthopedics",
    "rash": "dermatology",
    "skin": "dermatology",
    "acne": "dermatology",
    "depression": "psychiatry",
    "depressed": "psychiatry",
    "anxiety": "psychiatry",
    "panic": "psychiatry",
}

EMERGENCY_KEYWORDS = [
    "ambulance",
    "cannot breathe",
    "can't breathe",
    "chest pain",
    "stroke",
    "heart attack",
    "severe bleeding",
    "unconscious",
    "emergency",
]


GREETING_PATTERN = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b[!. ]*$", re.IGNORECASE)
AFFIRMATIVE_TOKENS = {"yes", "yep", "yeah", "correct", "right", "sure", "ok", "okay", "confirm"}
NEGATIVE_TOKENS = {"no", "nope", "not really", "wrong", "different", "another"}
CANONICAL_ITALY_CITIES = [
    "Milano",
    "Roma",
    "Napoli",
    "Torino",
    "Palermo",
    "Genova",
    "Bologna",
    "Firenze",
    "Bari",
    "Catania",
    "Venezia",
    "Verona",
    "Padova",
]


def _extract_age(text: str) -> int | None:
    patterns = [
        r"\b(\d{1,3})\s*(?:years?\s*old|yo)\b",
        r"\bi\s*am\s*(\d{1,3})\b",
        r"\byou(?:'re|\s+are)\s*(\d{1,3})\b",
        r"\bi\s*have\s*(\d{1,3})\b",
        r"\bage\s*(?:is\s*)?(\d{1,3})\b",
        r"^\s*(\d{1,3})\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 0 < value < 120:
                return value
    return None


def _extract_city(text: str) -> str | None:
    raw_city = _extract_city_raw(text)
    if not raw_city:
        return None
    return _normalize_city_name(raw_city)


def _extract_city_raw(text: str) -> str | None:
    def _clean_city_candidate(raw: str) -> str:
        cleaned = re.sub(r"[^A-Za-z\s'-]", " ", raw or "").strip()
        tokens = [t for t in cleaned.split() if t]
        if not tokens:
            return ""
        # Stop when sentence continues with symptom/context words.
        stop_after = {
            "i",
            "im",
            "i'm",
            "feel",
            "have",
            "with",
            "and",
            "but",
            "because",
            "since",
            "for",
            "pain",
            "depression",
            "anxiety",
            "fever",
            "cough",
            "injury",
        }
        city_tokens: list[str] = []
        for tok in tokens:
            low = tok.lower()
            if low in stop_after and city_tokens:
                break
            city_tokens.append(tok)
            if len(city_tokens) >= 3:
                break
        return " ".join(city_tokens).strip()

    patterns = [
        r"\bin\s+([A-Za-z][A-Za-z\s'-]{1,40})\b",
        r"\bnear\s+([A-Za-z][A-Za-z\s'-]{1,40})\b",
        r"\blive\s+in\s+([A-Za-z][A-Za-z\s'-]{1,40})\b",
        r"\bfrom\s+([A-Za-z][A-Za-z\s'-]{1,40})\b",
    ]
    stop_words = {"italy", "city", "doctor", "specialty", "appointment"}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_city_candidate(match.group(1))
        if not candidate:
            continue
        if candidate.split()[0].lower() in stop_words:
            continue
        return candidate
    # Accept bare city input like "milan" / "milano" (no "in ...")
    bare = re.sub(r"[^A-Za-z\s'-]", " ", text or "").strip()
    if bare:
        tokens = [t for t in bare.split() if t]
        if 1 <= len(tokens) <= 3:
            candidate = " ".join(tokens).strip()
            if candidate and candidate.lower() not in stop_words:
                return candidate
    return None


def _normalize_city_name(city_text: str) -> str | None:
    normalized = city_text.title()
    aliases = {
        "Milan": "Milano",
        "Rome": "Roma",
        "Florence": "Firenze",
        "Turin": "Torino",
        "Naples": "Napoli",
    }
    canonical = aliases.get(normalized, normalized).strip()
    if canonical in CANONICAL_ITALY_CITIES:
        return canonical
    # Strict validation: accept only close matches to known Italian cities.
    fuzzy = get_close_matches(canonical, CANONICAL_ITALY_CITIES, n=1, cutoff=0.82)
    return fuzzy[0] if fuzzy else None


def _extract_specialty(text: str) -> str | None:
    lowered = text.lower()
    for key, value in SPECIALTY_KEYWORDS.items():
        if key in lowered:
            return value
    return None


def _infer_specialty_from_symptoms(text: str) -> str | None:
    lowered = text.lower()
    for symptom, specialty in SYMPTOM_SPECIALTY_MAP.items():
        if symptom in lowered:
            return specialty
    return None


def _has_symptom_detail(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(r"\b(severity|pain\s*scale)\s*(is|=)?\s*\d{1,2}\b", lowered):
        return True
    if re.search(r"\b\d{1,2}\s*/\s*10\b", lowered):
        return True
    detail_tokens = (
        "since",
        "day",
        "yesterday",
        "week",
        "month",
        "severe",
        "mild",
        "moderate",
        "nausea",
        "vomit",
        "fever",
        "vision",
        "dizzy",
        "worse",
        "broken",
        "fracture",
        "injury",
        "swelling",
        "cannot walk",
        "depression",
        "depressed",
        "anxiety",
        "panic",
        "mental",
    )
    return any(token in lowered for token in detail_tokens)


def _extract_weight_support(text: str) -> int | None:
    match = re.search(r"\b(\d{2,3})\s*(kg|kilo|كيلو)\b", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    val = int(match.group(1))
    if 20 <= val <= 250:
        return val
    return None


def _extract_gender_preference(text: str) -> str | None:
    lowered = (text or "").lower()
    if any(x in lowered for x in ("female", "woman", "lady", "donna", "خانم")):
        return "female"
    if any(x in lowered for x in ("male", "man", "uomo", "آقا")):
        return "male"
    return None


def _is_elderly_homecare_context(text: str, known: dict[str, Any] | None = None) -> bool:
    lowered = (text or "").lower()
    known = known or {}
    age = known.get("age")
    cues = ("mother", "father", "elderly", "grandma", "grandpa", "home care", "at home", "bed")
    return bool((isinstance(age, int) and age >= 75) or any(c in lowered for c in cues))


def _looks_like_symptom_statement(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered or len(lowered) < 6:
        return False
    if lowered in {"yes", "no", "ok", "okay", "sure"}:
        return False
    if _extract_age(lowered) is not None:
        return False
    if _extract_city(lowered):
        return False
    # Broad intent cues so we don't hardcode every illness.
    symptom_cues = (
        "pain",
        "issue",
        "problem",
        "feel",
        "symptom",
        "anxiety",
        "stress",
        "depress",
        "mental",
        "broken",
        "injury",
        "fever",
        "cough",
        "head",
        "back",
        "arm",
        "leg",
    )
    return any(c in lowered for c in symptom_cues)


def extract_entities_from_text(text: str, known_entities: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic extractor used as a safety net for state persistence."""
    entities: dict[str, Any] = {}
    known = known_entities or {}
    rejected = {str(x).strip().lower() for x in (known.get("rejected_specialties") or []) if str(x).strip()}
    age = _extract_age(text)
    city_raw = _extract_city_raw(text)
    city = _normalize_city_name(city_raw) if city_raw else None
    explicit_specialty = _extract_specialty(text)
    if age is not None:
        entities["age"] = age
    if city:
        entities["city"] = city
        if city_raw and city_raw.lower() != city.lower():
            entities["city_input_raw"] = city_raw
    # Keep specialty extraction deterministic only when user explicitly names one.
    if explicit_specialty and explicit_specialty.lower() not in rejected:
        entities["specialty"] = explicit_specialty
        entities["required_specialty"] = explicit_specialty
        entities["specialty_confirmed"] = True
    weight = _extract_weight_support(text)
    if weight is not None:
        entities["weight_support_kg"] = weight
    gender_pref = _extract_gender_preference(text)
    if gender_pref:
        entities["gender_preference"] = gender_pref
    lowered = (text or "").lower().strip()
    symptom_tokens = (
        "pain",
        "ache",
        "headache",
        "back",
        "fever",
        "cough",
        "nausea",
        "dizzy",
        "rash",
        "injury",
        "broken",
        "fracture",
        "sprain",
        "swelling",
        "cannot walk",
    )
    if len(lowered) >= 8 and (any(token in lowered for token in symptom_tokens) or _looks_like_symptom_statement(text)):
        entities["symptom_context"] = text.strip()
    return entities


def has_explicit_specialty_mention(text: str) -> bool:
    """True only when user explicitly names a specialty term (not symptom inference)."""
    return _extract_specialty(text or "") is not None


def rerank_candidates_by_symptoms(
    candidates: list[dict[str, Any]],
    entities: dict[str, Any],
    user_message: str,
) -> list[dict[str, Any]]:
    """Use LLM to rerank providers by symptom/context fit."""
    if len(candidates) <= 1 or not settings.gemini_api_key:
        return candidates
    symptom_context = (entities.get("symptom_context") or user_message or "").strip()
    if not symptom_context:
        return candidates

    compact = []
    for i, c in enumerate(candidates):
        compact.append(
            {
                "idx": i,
                "name": c.get("name"),
                "specialty": c.get("specialty"),
                "city": c.get("city"),
                "bio": c.get("clinical_summary") or c.get("description") or "",
            }
        )

    prompt = (
        "Rank these doctors by best match to the patient context.\n"
        "Return strict JSON: {\"ranked_indices\": [..]} only.\n\n"
        f"Patient context: {symptom_context}\n"
        f"Age: {entities.get('age', '')}\n"
        f"Preferred city: {entities.get('city', '')}\n\n"
        f"Doctors: {json.dumps(compact)}"
    )
    try:
        response = _model(settings.gemini_chat_model).generate_content(
            prompt,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        payload = _parse_model_json((response.text or "").strip())
        if not payload:
            return candidates
        ranked_indices = payload.get("ranked_indices") or []
        if not isinstance(ranked_indices, list):
            return candidates
        safe_indices = [int(i) for i in ranked_indices if isinstance(i, int) and 0 <= i < len(candidates)]
        if not safe_indices:
            return candidates
        # Keep provided order first, then append missing ones.
        seen = set(safe_indices)
        safe_indices.extend([i for i in range(len(candidates)) if i not in seen])
        return [candidates[i] for i in safe_indices]
    except Exception:
        return candidates


def _is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERN.match(text or ""))


def _contains_any_token(text: str, tokens: set[str]) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    for token in tokens:
        t = (token or "").strip().lower()
        if not t:
            continue
        # Match as full word/phrase, not substring (prevents "no" matching inside "Milno").
        pattern = r"(?<!\w)" + re.escape(t).replace(r"\ ", r"\s+") + r"(?!\w)"
        if re.search(pattern, lowered):
            return True
    return False


def _friendly_specialty_name(name: str) -> str:
    return str(name or "").replace("_", " ").title()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_model_json(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    candidates = [cleaned]
    snippet = _extract_first_json_object(cleaned)
    if snippet and snippet not in candidates:
        candidates.append(snippet)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    return None


def _normalize_specialty_name(value: str) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return ""
    if lowered in SPECIALTY_KEYWORDS.values():
        return lowered
    for key, normalized in SPECIALTY_KEYWORDS.items():
        if key in lowered:
            return normalized
    return lowered


def _postprocess_llm_payload(
    payload: dict[str, Any],
    *,
    user_message: str,
    known_entities: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize LLM JSON so graph can trust it as the primary state source."""
    out = dict(payload or {})
    entities = dict(out.get("entities") or {})
    known = dict(known_entities or {})

    city_raw = str(entities.get("city") or "").strip()
    if city_raw:
        normalized_city = _normalize_city_name(city_raw)
        if normalized_city:
            entities["city"] = normalized_city
            if city_raw.lower() != normalized_city.lower():
                entities["city_input_raw"] = city_raw

    specialty_raw = str(entities.get("specialty") or "").strip()
    if specialty_raw:
        normalized_specialty = _normalize_specialty_name(specialty_raw)
        if normalized_specialty:
            entities["specialty"] = normalized_specialty
            entities.setdefault("required_specialty", normalized_specialty)

    chief = str(out.get("chief_complaint") or "").strip()
    if not chief:
        chief = str(entities.get("chief_complaint") or entities.get("symptom_context") or known.get("symptom_context") or user_message).strip()
    out["chief_complaint"] = chief
    if chief and not entities.get("symptom_context"):
        entities["symptom_context"] = chief

    # Safety fallback for mental-health intake when model forgets specialty.
    lowered = chief.lower()
    if ("depress" in lowered or "anxiety" in lowered or "panic" in lowered) and not entities.get("specialty"):
        entities["specialty"] = "psychiatry"
        entities["required_specialty"] = "psychiatry"
        entities.setdefault("specialty_confirmed", False)

    out["entities"] = entities
    if not isinstance(out.get("missing_fields"), list):
        out["missing_fields"] = []
    out["assistant_reply"] = str(out.get("assistant_reply") or "").strip()
    out["ready_to_search"] = bool(out.get("ready_to_search", False))
    return out


def _repair_json_with_model(raw_text: str, model_name: str) -> dict[str, Any] | None:
    """Ask model to reformat its previous output as strict JSON only."""
    repair_prompt = (
        "Convert the following assistant output into STRICT JSON only, no markdown.\n"
        "Required keys: is_emergency, emergency_reason, entities, missing_fields, assistant_reply, chief_complaint, ready_to_search.\n"
        "Preserve meaning, keep it concise.\n\n"
        f"Output to convert:\n{raw_text}"
    )
    try:
        response = _model(model_name).generate_content(
            repair_prompt,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
    except Exception:  # noqa: BLE001
        return None
    return _parse_model_json(getattr(response, "text", "") or "")


def _fallback_intake(user_message: str, known_entities: dict[str, Any] | None) -> dict[str, Any]:
    known = dict(known_entities or {})
    lowered = user_message.lower()
    is_emergency = any(token in lowered for token in EMERGENCY_KEYWORDS)
    if is_emergency:
        return {
            "is_emergency": True,
            "emergency_reason": "Potential emergency language detected",
            "entities": known,
            "missing_fields": [],
            "ready_to_search": False,
            "decision_mode": "fallback",
            "chief_complaint": str(user_message).strip(),
            "assistant_reply": (
                "This may be a medical emergency. Please call 112 immediately or go to the nearest emergency room."
            ),
        }

    extracted_age = _extract_age(user_message)
    extracted_city = _extract_city(user_message)
    explicit_specialty = _extract_specialty(user_message)
    current_symptom_context = str(known.get("symptom_context") or "").strip()
    extracted_entities = extract_entities_from_text(user_message, known_entities=known)
    if extracted_entities.get("symptom_context"):
        current_symptom_context = str(extracted_entities["symptom_context"]).strip()
        known["symptom_context"] = current_symptom_context
    combined_symptom_text = " ".join(
        x for x in [current_symptom_context, user_message] if str(x).strip()
    ).strip()
    inferred_specialty = _infer_specialty_from_symptoms(combined_symptom_text)
    has_affirmation = _contains_any_token(user_message, AFFIRMATIVE_TOKENS)
    has_rejection = _contains_any_token(user_message, NEGATIVE_TOKENS)

    if extracted_age is not None:
        known["age"] = extracted_age
    if extracted_city:
        known["city"] = extracted_city
    extracted_weight = _extract_weight_support(user_message)
    if extracted_weight is not None:
        known["weight_support_kg"] = extracted_weight
    extracted_gender = _extract_gender_preference(user_message)
    if extracted_gender:
        known["gender_preference"] = extracted_gender

    if explicit_specialty:
        known["specialty"] = explicit_specialty
        known["required_specialty"] = explicit_specialty
        known["specialty_confirmed"] = True
        known.pop("suggested_specialty", None)
    elif inferred_specialty and not known.get("specialty"):
        if _has_symptom_detail(combined_symptom_text):
            known["suggested_specialty"] = inferred_specialty
        else:
            return {
                "is_emergency": False,
                "emergency_reason": "",
                "entities": known,
                "missing_fields": ["symptom_details"],
                "ready_to_search": False,
                "decision_mode": "fallback",
                "chief_complaint": str(current_symptom_context or user_message).strip(),
                "assistant_reply": (
                    "I understand. Before I suggest a specialist, tell me: "
                    "how long you’ve had this, how severe it is (1-10), and any nausea/vision changes/fever."
                ),
            }

    # User confirms the previously suggested specialty.
    if known.get("suggested_specialty") and has_affirmation:
        chosen = str(known["suggested_specialty"])
        known["specialty"] = chosen
        known["required_specialty"] = chosen
        known["specialty_confirmed"] = True
        known.pop("suggested_specialty", None)
    elif known.get("suggested_specialty") and has_rejection:
        # Avoid repeating the same recommendation in a loop.
        known["specialty_confirmed"] = False
        rejected_value = str(known.pop("suggested_specialty"))
        known["last_rejected_specialty"] = rejected_value
        rejected_list = [str(x) for x in (known.get("rejected_specialties") or [])]
        if rejected_value not in rejected_list:
            rejected_list.append(rejected_value)
        known["rejected_specialties"] = rejected_list
        if known.get("specialty") == rejected_value:
            known.pop("specialty", None)
            known.pop("required_specialty", None)

    has_specialty = bool(known.get("specialty"))
    specialty_confirmed = bool(known.get("specialty_confirmed")) or has_specialty

    if _is_greeting(user_message) and not known.get("age") and not known.get("city") and not has_specialty:
        return {
            "is_emergency": False,
            "emergency_reason": "",
            "entities": known,
            "missing_fields": ["symptoms"],
            "ready_to_search": False,
            "decision_mode": "fallback",
            "chief_complaint": str(user_message).strip(),
            "assistant_reply": (
                "Hi, I can help you find the right medical specialist. "
                "What symptoms or concern are you having?"
            ),
        }

    # If we can infer specialty from symptoms, ask for confirmation first.
    if known.get("suggested_specialty") and not specialty_confirmed:
        suggested = _friendly_specialty_name(str(known["suggested_specialty"]))
        return {
            "is_emergency": False,
            "emergency_reason": "",
            "entities": known,
            "missing_fields": ["specialty_confirmation"],
            "ready_to_search": False,
            "decision_mode": "fallback",
            "chief_complaint": str(known.get("symptom_context") or user_message).strip(),
            "assistant_reply": (
                f"Based on your symptoms, the best match is likely {suggested}. "
                "Does that sound right?"
            ),
        }

    missing_fields = [field for field in ("age", "city", "specialty") if not known.get(field)]
    if _is_elderly_homecare_context(user_message, known):
        if not known.get("weight_support_kg"):
            missing_fields.append("weight_support_kg")
        if not known.get("gender_preference"):
            missing_fields.append("gender_preference")

    if not missing_fields and specialty_confirmed:
        friendly_specialty = _friendly_specialty_name(str(known["specialty"]))
        reply = (
            f"I understand. I'll look for a {friendly_specialty} for you in {known['city']}. "
            "I just need a second to check the providers."
        )
    else:
        # Keep follow-ups natural and only ask for truly missing info.
        if known.get("last_rejected_specialty") and not known.get("specialty"):
            rejected = _friendly_specialty_name(str(known.get("last_rejected_specialty", "")))
            reply = (
                f"Understood, not {rejected}. "
                "Can you share 1-2 more details (for example: pain location, duration, fever, nausea, injury)?"
            )
            known.pop("last_rejected_specialty", None)
        elif has_rejection and known.get("suggested_specialty"):
            reply = "No problem. Tell me your preferred specialty, or describe symptoms in more detail."
        elif "city" in missing_fields and "age" not in missing_fields:
            reply = f"Got it, you're {known.get('age')}. Which city in Italy are you in so I can find someone nearby?"
        elif "age" in missing_fields and "city" not in missing_fields:
            reply = f"Understood, you're in {known.get('city')}. What's the patient's age?"
        elif "specialty" in missing_fields and known.get("city"):
            reply = (
                f"Got it, you're in {known.get('city')}. "
                "What symptoms are most important so I can suggest the right specialty?"
            )
        else:
            prompts = {
                "age": "patient age",
                "city": "city in Italy",
                "specialty": "main symptoms (or preferred specialty)",
            }
            items = [prompts[item] for item in missing_fields]
            request_text = ", ".join(items[:-1]) + (f", and {items[-1]}" if len(items) > 1 else items[0])
            reply = f"Thanks. Please share the {request_text} so I can continue."

    return {
        "is_emergency": False,
        "emergency_reason": "",
        "entities": known,
        "missing_fields": missing_fields,
        "ready_to_search": (len(missing_fields) == 0 and specialty_confirmed),
        "decision_mode": "fallback",
        "chief_complaint": str(known.get("symptom_context") or user_message).strip(),
        "assistant_reply": reply,
    }


def analyze_intake(
    user_message: str,
    known_entities: dict[str, Any] | None = None,
    transcript: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not settings.gemini_api_key:
        return _fallback_intake(user_message, known_entities)

    reply_lang = _conversation_reply_language(transcript, user_message)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"MANDATORY language for assistant_reply: {reply_lang} only (locked to how the user started the chat). "
        f"Every word of assistant_reply must be in {reply_lang}. "
        "Do not mix languages or switch mid-conversation.\n\n"
        f"Known entities from previous turns: {json.dumps(known_entities or {})}\n"
        f"Full transcript history: {json.dumps(transcript or [])}\n\n"
        f"User: {user_message}\n\n"
        "When possible, preserve previous valid entities and only overwrite if user corrects them.\n"
        "Return only JSON.\n"
        "Behavior constraints:\n"
        "- One best next question only.\n"
        "- If evidence is weak, do not finalize specialty yet.\n"
        "- If user already provided age/city/symptoms in any previous turn, never ask those again.\n"
        "- Always include chief_complaint with the best short clinical summary."
    )
    model_candidates = [
        settings.gemini_chat_model,
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    text = ""
    last_model_name = model_candidates[0]
    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            last_model_name = model_name
            response = _model(model_name).generate_content(
                prompt,
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
            )
            text = (response.text or "").strip()
            if text:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if not text:
        fallback = _fallback_intake(user_message, known_entities)
        fallback["error"] = str(last_error) if last_error else "empty_model_response"
        return fallback

    parsed = _parse_model_json(text)
    if parsed is not None:
        parsed = _postprocess_llm_payload(parsed, user_message=user_message, known_entities=known_entities)
        parsed.setdefault("decision_mode", "llm")
        parsed.setdefault("chief_complaint", str((known_entities or {}).get("symptom_context") or user_message).strip())
        parsed.setdefault("ready_to_search", False)
        return parsed

    repaired = _repair_json_with_model(text, last_model_name)
    if repaired is not None:
        repaired = _postprocess_llm_payload(repaired, user_message=user_message, known_entities=known_entities)
        repaired.setdefault("decision_mode", "llm")
        repaired.setdefault("chief_complaint", str((known_entities or {}).get("symptom_context") or user_message).strip())
        repaired.setdefault("ready_to_search", False)
        return repaired

    fallback = _fallback_intake(user_message, known_entities)
    fallback["error"] = "llm_json_parse_failed"
    return fallback
