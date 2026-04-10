# main.py — FastAPI application entry point.
# Exposes: /api/chat, /api/voice, /api/checkout, /api/bookings, /api/availability.
# Manages session persistence and routes user messages through the LangGraph pipeline.

import logging

import stripe
from datetime import datetime, timedelta

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.graph import app_graph
from pydantic import BaseModel as PydanticBaseModel
from app.models import BookingStatusResponse, ChatRequest, ChatResponse, CheckoutRequest, ProfessionalCandidate
from app.services.booking_store import (
    attach_checkout_session,
    confirm_booking_by_checkout_session,
    create_pending_booking,
    get_booking,
    init_booking_store,
)
from app.services.session_store import get_or_create_session, init_session_store, save_session
from app.services.stt import transcribe_audio
from app.services.stripe_checkout import create_checkout_session

app = FastAPI(title="Medical Agentic RAG API")
stripe.api_key = settings.stripe_secret_key

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.on_event("startup")
def startup() -> None:
    init_session_store()
    init_booking_store()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_data = get_or_create_session(req.session_id)
    initial_state = {
        "user_message": req.message,
        "known_entities": session_data["last_entities"],
        "transcript": session_data["transcript"],
        "is_emergency": False,
        "emergency_reason": "",
        "entities": {},
        "missing_fields": [],
        "reply": "",
        "candidates": [],
        "status": "needs_info",
        "decision_mode": "fallback",
        "decision_reason": "",
    }
    try:
        result = app_graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        result = {
            "reply": f"Backend error while processing request: {exc}",
            "status": "needs_info",
            "decision_mode": "fallback",
            "decision_reason": str(exc),
            "missing_fields": [],
            "entities": {},
            "candidates": [],
        }
    decision_mode = str(result.get("decision_mode", "fallback"))
    decision_reason = str(result.get("decision_reason", "") or result.get("error", ""))
    # Use uvicorn logger so this always appears in terminal output.
    logging.getLogger("uvicorn.error").info(
        "chat decision_mode=%s status=%s reason=%s session_id=%s",
        decision_mode,
        result.get("status", "needs_info"),
        decision_reason or "-",
        session_data["session_id"],
    )
    save_session(
        session_id=session_data["session_id"],
        entities=result["entities"],
        user_message=req.message,
        assistant_reply=result["reply"],
        status=result["status"],
    )
    return ChatResponse(
        session_id=session_data["session_id"],
        reply=result["reply"],
        status=result["status"],
        decision_mode=decision_mode,
        decision_reason=decision_reason or None,
        missing_fields=result["missing_fields"],
        extracted_entities=result["entities"],
        candidates=[ProfessionalCandidate(**c) for c in result["candidates"]],
    )


@app.post("/api/voice", response_model=ChatResponse)
def voice_chat(file: UploadFile = File(...), session_id: str | None = Form(default=None)):
    try:
        text = transcribe_audio(file)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Voice transcription requires ffmpeg. "
                "Install it (macOS: `brew install ffmpeg`) and retry."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Voice transcription failed: {exc}") from exc
    return chat(ChatRequest(message=text, session_id=session_id))


@app.post("/api/checkout")
def checkout(req: CheckoutRequest):
    booking_id = create_pending_booking(
        session_id=req.session_id,
        professional_name=req.professional_name,
        appointment_date=req.appointment_date,
        amount_cents=req.amount_cents,
        currency=settings.stripe_currency,
    )
    success_url = req.success_url
    if "session_id=" not in success_url and "{CHECKOUT_SESSION_ID}" not in success_url:
        sep = "&" if "?" in success_url else "?"
        success_url = f"{success_url}{sep}session_id={{CHECKOUT_SESSION_ID}}"

    checkout_url, checkout_session_id = create_checkout_session(
        booking_id=booking_id,
        session_id=req.session_id,
        professional_name=req.professional_name,
        appointment_date=req.appointment_date,
        amount_cents=req.amount_cents,
        success_url=success_url,
        cancel_url=req.cancel_url,
    )
    attach_checkout_session(booking_id, checkout_session_id)
    return {"checkout_url": checkout_url, "booking_id": booking_id}


@app.post("/api/checkout/confirm", response_model=BookingStatusResponse)
def confirm_checkout(session_id: str = Query(...)):
    try:
        checkout_session = stripe.checkout.Session.retrieve(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to verify checkout session: {exc}") from exc

    if checkout_session.get("payment_status") != "paid":
        raise HTTPException(status_code=409, detail="Payment is not completed yet")

    booking_id = confirm_booking_by_checkout_session(session_id)
    if not booking_id:
        raise HTTPException(status_code=404, detail="Booking not found for this checkout session")

    booking = get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return BookingStatusResponse(**booking)


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    if not settings.stripe_webhook_secret:
        raise HTTPException(
            status_code=400,
            detail="STRIPE_WEBHOOK_SECRET is not configured. Use /api/checkout/confirm instead.",
        )

    payload = await request.body()
    signature = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, signature, settings.stripe_webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Stripe webhook: {exc}") from exc

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        confirm_booking_by_checkout_session(session.get("id"))
    return {"received": True}


class MockBookingRequest(PydanticBaseModel):
    session_id: str
    professional_name: str
    specialty: str
    city: str
    appointment_date: str


@app.post("/api/bookings/mock")
def mock_booking(req: MockBookingRequest):
    booking_id = create_pending_booking(
        session_id=req.session_id,
        professional_name=req.professional_name,
        appointment_date=req.appointment_date,
        amount_cents=5000,
        currency=settings.stripe_currency,
    )
    from app.services.supabase_client import get_supabase_client
    sb = get_supabase_client()
    if sb:
        sb.table("bookings").update({
            "status": "confirmed",
            "confirmation_message": (
                f"Booking confirmed: {req.professional_name} ({req.specialty}) "
                f"in {req.city} on {req.appointment_date}"
            ),
        }).eq("id", booking_id).execute()
    return {"booking_id": booking_id, "status": "confirmed"}


@app.get("/api/bookings/{booking_id}", response_model=BookingStatusResponse)
def booking_status(booking_id: str):
    booking = get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return BookingStatusResponse(**booking)


@app.get("/api/availability")
def availability(professional_id: str = Query(...), days: int = Query(default=3, ge=1, le=14)):
    """Return simple upcoming slots for a professional.

    This endpoint is intentionally lightweight and can be replaced later with real calendar integration.
    """
    _ = professional_id
    base = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
    templates = (9, 11, 14, 17)
    slots: list[str] = []
    for d in range(days):
        current_day = base.date() + timedelta(days=d)
        for hour in templates:
            candidate = datetime.combine(current_day, datetime.min.time()).replace(hour=hour)
            if candidate > datetime.now():
                slots.append(candidate.isoformat())
    return {"professional_id": professional_id, "slots": slots[:8]}
