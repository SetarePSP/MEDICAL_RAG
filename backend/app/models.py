# models.py — Pydantic request/response schemas for the API.
# Defines ChatRequest, ChatResponse, CheckoutRequest, BookingStatusResponse,
# and ProfessionalCandidate (the matched doctor card shown to the user).

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ProfessionalCandidate(BaseModel):
    id: str
    name: str
    specialty: str
    city: str
    score: float = 0.0
    address: str | None = None
    phone: str | None = None
    maps_url: str | None = None
    supports_weight_kg: int | None = None


class TtsRequest(BaseModel):
    text: str = Field(..., max_length=4096)


class TranscribeResponse(BaseModel):
    """Speech-to-text only; user sends the message via /api/chat when ready."""

    text: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: Literal["needs_info", "emergency", "matched", "booked"] = "needs_info"
    decision_mode: Literal["llm", "fallback"] = "fallback"
    decision_reason: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    candidates: list[ProfessionalCandidate] = Field(default_factory=list)
    # Set only for /api/voice: what Whisper transcribed (so the UI can show the user message).
    transcript: str | None = None


class CheckoutRequest(BaseModel):
    session_id: str
    professional_name: str
    appointment_date: str
    amount_cents: int = 5000
    success_url: str
    cancel_url: str


class BookingStatusResponse(BaseModel):
    booking_id: str
    status: Literal["pending_payment", "confirmed", "failed"]
    confirmation_message: str | None = None
