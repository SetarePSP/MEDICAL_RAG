# config.py — Application settings loaded from .env via pydantic-settings.
# Central source for all API keys, model names, and service configuration.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    # auto | gemini | rest | vertex — use vertex when Gemini API key blocks EmbedContent
    gemini_embedding_backend: str = "auto"
    google_cloud_project: str = ""
    vertex_ai_location: str = "us-central1"
    vertex_embedding_model: str = "text-embedding-004"
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    google_places_api_key: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_currency: str = "eur"
    whisper_model: str = "base"


settings = Settings()
