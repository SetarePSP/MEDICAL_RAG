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
- **LLM Observability** — Full trace visualization via LangSmith
- **Error Tracking** — Sentry integration for production monitoring
- **CI/CD** — GitHub Actions auto-deploys backend to Cloud Run on push

## Architecture

```
frontend/          React + TypeScript + Tailwind (Vite) → deployed on Vercel
backend/           FastAPI + LangGraph → deployed on Cloud Run
  app/
    main.py        FastAPI entry point + Sentry init
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
  Dockerfile               Cloud Run container definition
.github/workflows/
  deploy-backend.yml       CI/CD pipeline for Cloud Run
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

## Local Development

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
gcloud auth application-default login
python scripts/seed_db.py
python scripts/ingest_embeddings.py --only-missing
```

Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Deployment

### Backend → Google Cloud Run

**One-time setup:**

```bash
# 1. Create an Artifact Registry repo (once)
gcloud artifacts repositories create medical-rag \
  --repository-format=docker \
  --location=us-central1

# 2. Build & push the Docker image
cd backend
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/medical-rag/api

# 3. Deploy to Cloud Run
gcloud run deploy medical-rag-api \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/medical-rag/api \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --set-env-vars "GEMINI_API_KEY=xxx,SUPABASE_URL=xxx,SUPABASE_SERVICE_ROLE_KEY=xxx,GOOGLE_CLOUD_PROJECT=xxx"
```

Cloud Run gives you a URL like `https://medical-rag-api-xxxxx.a.run.app`.

**Automated CI/CD:** The included GitHub Actions workflow (`.github/workflows/deploy-backend.yml`) auto-deploys on every push to `main` that changes `backend/` files. Add these GitHub Secrets:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your Google Cloud project ID |
| `GCP_SA_KEY` | Service account JSON key (with Cloud Run + Artifact Registry permissions) |
| `GEMINI_API_KEY` | Gemini API key |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `SENTRY_DSN` | Sentry DSN (optional) |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional) |

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **"Add New Project"** → import `SetarePSP/MEDICAL_RAG`
3. Set **Root Directory** to `frontend`
4. Add environment variable: `VITE_API_BASE_URL` = your Cloud Run URL
5. Click **Deploy**

Vercel auto-deploys on every push to `main`.

## Monitoring & Observability

### LangSmith (LLM Traces)

Traces every LangGraph invocation — see prompts, entity extraction, search decisions, and response times.

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Create an API key under Settings
3. Add to `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-key
   LANGCHAIN_PROJECT=medical-rag
   ```

### Sentry (Error Tracking)

Catches every unhandled exception with full stack traces, request data, and performance metrics.

1. Sign up at [sentry.io](https://sentry.io)
2. Create a new project (choose **FastAPI**)
3. Copy the DSN and add to `.env`:
   ```
   SENTRY_DSN=your-dsn
   ```

### Cloud Run Metrics (Automatic)

When deployed on Cloud Run, you get these for free in GCP Console:
- Request count, latency, error rate
- Memory and CPU usage
- Cold start frequency
- Logs (stdout from uvicorn)

## Environment Variables

All backend secrets go in `backend/.env` (never committed — listed in .gitignore):

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini API key for chat |
| `GEMINI_CHAT_MODEL` | No | Default: `gemini-2.5-flash` |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID (for Vertex AI embeddings) |
| `VERTEX_AI_LOCATION` | No | Default: `us-central1` |
| `GEMINI_EMBEDDING_BACKEND` | No | Default: `vertex` |
| `GOOGLE_PLACES_API_KEY` | No | For address enrichment |
| `STRIPE_SECRET_KEY` | No | For real Stripe payments |
| `WHISPER_MODEL` | No | Default: `base` |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | Default: `medical-rag` |
| `SENTRY_DSN` | No | Sentry error tracking DSN |

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
3. LLM infers the appropriate specialty and sets `ready_to_search` flag
4. Hybrid search finds matching providers (structured filters + vector similarity)
5. Results are reranked by symptom relevance and post-filtered by city/specialty
6. User selects a provider, picks a time slot, completes mock payment
7. Booking is saved to database, confirmation shared via Email/WhatsApp/Telegram
