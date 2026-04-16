# config.py — Application settings loaded from .env via pydantic-settings.
# Central source for all API keys, model names, and service configuration.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
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
    # Whisper STT: leave language empty for auto-detect (IT speech → IT text, EN → EN).
    # Set e.g. it or en to force one language. task=translate outputs English text from any language.
    whisper_language: str = ""
    whisper_task: str = ""  # empty = transcribe in spoken language; set to "translate" for English-only text

    # LangSmith observability (auto-traces LangGraph when set)
    langchain_tracing_v2: str = ""
    langchain_api_key: str = ""
    langchain_project: str = "medical-rag"
    langchain_endpoint: str = ""

    # Sentry error tracking
    sentry_dsn: str = ""

    # OpenAI Text-to-Speech (optional — /api/tts reads replies aloud on the client)
    openai_api_key: str = ""
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "nova"
    openai_tts_speed: float = 1.12


settings = Settings()
