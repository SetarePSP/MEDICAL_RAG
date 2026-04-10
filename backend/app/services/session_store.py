# session_store.py — Persists chat sessions (entities + transcript) to Supabase or local SQLite.
# Each session tracks the full conversation history and extracted patient information across turns.

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from app.services.supabase_client import get_supabase_client


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_session_store() -> None:
    with _conn() as conn:
        conn.execute(
            """
            create table if not exists intake_sessions (
              id text primary key,
              last_entities text not null,
              transcript text not null,
              status text not null default 'needs_info'
            )
            """
        )


def get_or_create_session(session_id: str | None) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is not None:
        if session_id:
            data = (
                sb.table("intake_sessions")
                .select("id,last_entities,transcript,status")
                .eq("id", session_id)
                .limit(1)
                .execute()
                .data
            )
            if data:
                row = data[0]
                return {
                    "session_id": row["id"],
                    "last_entities": row.get("last_entities") or {},
                    "transcript": row.get("transcript") or [],
                    "status": row.get("status") or "needs_info",
                }

        new_id = str(uuid.uuid4())
        sb.table("intake_sessions").insert(
            {"id": new_id, "last_entities": {}, "transcript": [], "status": "needs_info"}
        ).execute()
        return {"session_id": new_id, "last_entities": {}, "transcript": [], "status": "needs_info"}

    with _conn() as conn:
        if session_id:
            row = conn.execute(
                "select id, last_entities, transcript, status from intake_sessions where id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return {
                    "session_id": row["id"],
                    "last_entities": json.loads(row["last_entities"]),
                    "transcript": json.loads(row["transcript"]),
                    "status": row["status"],
                }

        new_id = str(uuid.uuid4())
        conn.execute(
            "insert into intake_sessions (id, last_entities, transcript, status) values (?, ?, ?, ?)",
            (new_id, "{}", "[]", "needs_info"),
        )
        return {"session_id": new_id, "last_entities": {}, "transcript": [], "status": "needs_info"}


def save_session(
    session_id: str,
    entities: dict[str, Any],
    user_message: str,
    assistant_reply: str,
    status: str,
) -> None:
    sb = get_supabase_client()
    if sb is not None:
        existing = (
            sb.table("intake_sessions")
            .select("transcript")
            .eq("id", session_id)
            .limit(1)
            .execute()
            .data
        )
        transcript = (existing[0].get("transcript") if existing else []) or []
        transcript.append({"role": "user", "content": user_message})
        transcript.append({"role": "assistant", "content": assistant_reply})
        sb.table("intake_sessions").update(
            {"last_entities": entities, "transcript": transcript[-20:], "status": status}
        ).eq("id", session_id).execute()
        return

    with _conn() as conn:
        row = conn.execute(
            "select transcript from intake_sessions where id = ?",
            (session_id,),
        ).fetchone()
        transcript = json.loads(row["transcript"]) if row else []
        transcript.append({"role": "user", "content": user_message})
        transcript.append({"role": "assistant", "content": assistant_reply})
        conn.execute(
            """
            update intake_sessions
            set last_entities = ?, transcript = ?, status = ?
            where id = ?
            """,
            (json.dumps(entities), json.dumps(transcript[-20:]), status, session_id),
        )
