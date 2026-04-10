# supabase_client.py — Factory for the Supabase Python client.
# Returns None if credentials are missing, allowing graceful fallback to local SQLite.

from supabase import Client, create_client

from app.config import settings


def get_supabase_client() -> Client | None:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
