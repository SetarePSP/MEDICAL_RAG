# stripe_checkout.py — Creates Stripe Checkout sessions for real payment processing.
# Used by the /api/checkout endpoint. Currently the frontend uses mock payment instead.

import stripe

from app.config import settings

stripe.api_key = settings.stripe_secret_key


def create_checkout_session(
    booking_id: str,
    session_id: str,
    professional_name: str,
    appointment_date: str,
    amount_cents: int,
    success_url: str,
    cancel_url: str,
) -> tuple[str, str]:
    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"booking_id": booking_id, "session_id": session_id},
        line_items=[
            {
                "quantity": 1,
                "price_data": {
                    "currency": settings.stripe_currency,
                    "unit_amount": amount_cents,
                    "product_data": {
                        "name": f"Visita con {professional_name}",
                        "description": f"Data: {appointment_date}",
                    },
                },
            }
        ],
    )
    return session.url, session.id
