# booking_store.py — Persists bookings to Supabase (preferred) or local SQLite.
# Tracks booking lifecycle: pending_payment → confirmed. Used by checkout and mock payment flows.

import sqlite3
import uuid
from pathlib import Path

from app.services.supabase_client import get_supabase_client


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_booking_store() -> None:
    with _conn() as conn:
        conn.execute(
            """
            create table if not exists bookings (
              id text primary key,
              session_id text not null,
              professional_name text not null,
              appointment_date text not null,
              amount_cents integer not null,
              currency text not null,
              status text not null,
              stripe_checkout_session_id text,
              confirmation_message text
            )
            """
        )


def create_pending_booking(
    session_id: str,
    professional_name: str,
    appointment_date: str,
    amount_cents: int,
    currency: str,
) -> str:
    booking_id = str(uuid.uuid4())
    sb = get_supabase_client()
    if sb is not None:
        sb.table("bookings").insert(
            {
                "id": booking_id,
                "session_id": session_id,
                "professional_name": professional_name,
                "appointment_date": appointment_date,
                "amount_cents": amount_cents,
                "currency": currency,
                "status": "pending_payment",
            }
        ).execute()
        return booking_id

    with _conn() as conn:
        conn.execute(
            """
            insert into bookings (
              id, session_id, professional_name, appointment_date, amount_cents, currency, status
            ) values (?, ?, ?, ?, ?, ?, 'pending_payment')
            """,
            (booking_id, session_id, professional_name, appointment_date, amount_cents, currency),
        )
    return booking_id


def attach_checkout_session(booking_id: str, checkout_session_id: str) -> None:
    sb = get_supabase_client()
    if sb is not None:
        sb.table("bookings").update({"stripe_checkout_session_id": checkout_session_id}).eq(
            "id", booking_id
        ).execute()
        return

    with _conn() as conn:
        conn.execute(
            "update bookings set stripe_checkout_session_id = ? where id = ?",
            (checkout_session_id, booking_id),
        )


def confirm_booking_by_checkout_session(checkout_session_id: str) -> str | None:
    sb = get_supabase_client()
    if sb is not None:
        data = (
            sb.table("bookings")
            .select("id,professional_name,appointment_date")
            .eq("stripe_checkout_session_id", checkout_session_id)
            .limit(1)
            .execute()
            .data
        )
        if not data:
            return None
        row = data[0]
        msg = f"Booking confirmed for {row['appointment_date']} with {row['professional_name']}"
        sb.table("bookings").update({"status": "confirmed", "confirmation_message": msg}).eq(
            "stripe_checkout_session_id", checkout_session_id
        ).execute()
        return row["id"]

    with _conn() as conn:
        row = conn.execute(
            """
            select id, professional_name, appointment_date
            from bookings
            where stripe_checkout_session_id = ?
            """,
            (checkout_session_id,),
        ).fetchone()
        if not row:
            return None
        msg = f"Booking confirmed for {row['appointment_date']} with {row['professional_name']}"
        conn.execute(
            """
            update bookings
            set status = 'confirmed', confirmation_message = ?
            where stripe_checkout_session_id = ?
            """,
            (msg, checkout_session_id),
        )
        return row["id"]


def get_booking(booking_id: str) -> dict | None:
    sb = get_supabase_client()
    if sb is not None:
        data = (
            sb.table("bookings")
            .select("id,status,confirmation_message")
            .eq("id", booking_id)
            .limit(1)
            .execute()
            .data
        )
        if not data:
            return None
        row = data[0]
        return {
            "booking_id": row["id"],
            "status": row["status"],
            "confirmation_message": row.get("confirmation_message"),
        }

    with _conn() as conn:
        row = conn.execute(
            """
            select id, status, confirmation_message
            from bookings
            where id = ?
            """,
            (booking_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "booking_id": row["id"],
            "status": row["status"],
            "confirmation_message": row["confirmation_message"],
        }
