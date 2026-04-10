# Medical RAG — Conversational Booking System

An AI-powered medical intake assistant that conducts multi-turn patient interviews, infers the right specialist using clinical reasoning, searches a provider database via hybrid RAG, and handles appointment booking with mock payment.

## Features

- **LLM-driven conversation** — Gemini 2.5 Flash conducts the medical interview, extracts entities, and decides when to search
- **Hybrid RAG search** — Combines structured filters (city, specialty, age) with pgvector semantic similarity on Supabase
- **19 medical specialties** across 11 Italian cities (40 providers)
- **Smart entity handling** — Typo correction (Milno → Milano), alias resolution (neurology ↔ neurologist), multi-turn memory
- **Voice input** — Whisper STT for speech-to-text
- **Mock payment flow** — Card form → booking saved to database → confirmation via Email/WhatsApp/Telegram
- **Multilingual** — Responds in the same language the user writes (English, Italian, etc.)
- **Emergency detection** — Flags critical symptoms (chest pain, stroke) and advises calling 112

## Architecture

```
frontend/          React + TypeScript + Tailwind (Vite)
backend/
  app/
    main.py        FastAPI entry point
    graph.py       LangGraph state machine (triage → search → match)
    models.py      Pydantic request/response schemas
    config.py      Settings from .env
    services/
      llm.py               Gemini chat + entity extraction + reranking
      supabase_search.py   Hybrid RAG (RPC + specialty aliases + fallbacks)
      embeddings.py        Query embedding builder
      gemini_embeddings.py Vector generation (Vertex AI / Gemini API)
      session_store.py     Chat session persistence (Supabase / SQLite)
      booking_store.py     Booking persistence (Supabase / SQLite)
      google_places.py     Address/phone enrichment (optional)
      stripe_checkout.py   Stripe payment sessions (optional)
      stt.py               Whisper speech-to-text
      supabase_client.py   Supabase client factory
  scripts/
    seed_db.py             Populate professionals table
    ingest_embeddings.py   Generate vector embeddings for all providers
  supabase/
    schema.sql             Database schema (run in Supabase SQL Editor)
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- A [Supabase](https://supabase.com) project
- A [Google Cloud](https://console.cloud.google.com) project with:
  - Gemini API key (for chat)
  - Vertex AI API enabled (for embeddings)
  - `Vertex AI User` IAM role on your account
- ffmpeg (for voice input): `brew install ffmpeg`

## Setup

### 1. Database (Supabase)

Open the **SQL Editor** in your Supabase dashboard and run the contents of `backend/supabase/schema.sql`. This creates the `professionals`, `intake_sessions`, and `bookings` tables plus the hybrid search RPC function.

### 2. Backend

```bash
cd backend
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your real keys (see .env.example for descriptions)
```

Seed the database and generate embeddings:

```bash
# Authenticate with Google Cloud (needed for Vertex AI embeddings)
gcloud auth application-default login

python scripts/seed_db.py
python scripts/ingest_embeddings.py --only-missing
```

Start the backend:

```bash
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Environment Variables

All backend secrets go in `backend/.env` (never committed — listed in .gitignore):

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini API key for chat (create at aistudio.google.com) |
| `GEMINI_CHAT_MODEL` | No | Default: `gemini-2.5-flash` |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID (for Vertex AI embeddings) |
| `VERTEX_AI_LOCATION` | No | Default: `us-central1` |
| `GEMINI_EMBEDDING_BACKEND` | No | Default: `vertex` |
| `GOOGLE_PLACES_API_KEY` | No | For address enrichment |
| `STRIPE_SECRET_KEY` | No | For real Stripe payments |
| `WHISPER_MODEL` | No | Default: `base` |

Frontend uses `VITE_API_BASE_URL` (defaults to `http://localhost:8000`).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Main conversation endpoint |
| `POST` | `/api/voice` | Voice input (audio file → Whisper → chat) |
| `POST` | `/api/bookings/mock` | Save confirmed mock booking to database |
| `POST` | `/api/checkout` | Create Stripe Checkout session |
| `POST` | `/api/checkout/confirm` | Confirm payment without webhook |
| `GET` | `/api/availability` | Get available time slots for a provider |
| `GET` | `/api/bookings/{id}` | Get booking status |
| `GET` | `/health` | Health check |

## How It Works

1. User describes symptoms in natural language
2. Gemini conducts a clinical interview (asks follow-ups, extracts age/city/symptoms)
3. LLM infers the appropriate specialty and asks for confirmation
4. Hybrid search finds matching providers (structured filters + vector similarity)
5. User selects a provider, picks a time slot, completes mock payment
6. Booking is saved to database, confirmation shared via Email/WhatsApp/Telegram
