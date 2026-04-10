-- schema.sql — Supabase database schema for the Medical RAG system.
-- Creates: professionals (with pgvector embeddings), intake_sessions, bookings tables.
-- Includes the match_professionals_hybrid RPC function for hybrid structured+semantic search.
-- Deploy manually in Supabase SQL Editor.

-- Enable vector operations
create extension if not exists vector;
create extension if not exists pgcrypto;

-- Core professionals table (structured + vector fields)
create table if not exists public.professionals (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  specialty text not null,
  city text not null,
  gender text,
  supports_weight_kg integer,
  years_experience integer default 0,
  clinical_summary text,
  embedding vector(768), -- keep aligned with your embedding model dimension
  created_at timestamptz not null default now()
);

create index if not exists idx_professionals_city on public.professionals(city);
create index if not exists idx_professionals_specialty on public.professionals(specialty);
create index if not exists idx_professionals_embedding_ivfflat
  on public.professionals using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Conversation/session memory
create table if not exists public.intake_sessions (
  id uuid primary key default gen_random_uuid(),
  last_entities jsonb not null default '{}'::jsonb,
  transcript jsonb not null default '[]'::jsonb,
  status text not null default 'needs_info',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Booking status is only confirmed by webhook
create table if not exists public.bookings (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references public.intake_sessions(id) on delete set null,
  professional_name text not null,
  appointment_date text not null,
  amount_cents integer not null,
  currency text not null default 'eur',
  status text not null default 'pending_payment',
  stripe_checkout_session_id text unique,
  confirmation_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Trigger utility for updated_at
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_intake_sessions_updated_at on public.intake_sessions;
create trigger trg_intake_sessions_updated_at
before update on public.intake_sessions
for each row execute function public.set_updated_at();

drop trigger if exists trg_bookings_updated_at on public.bookings;
create trigger trg_bookings_updated_at
before update on public.bookings
for each row execute function public.set_updated_at();

-- Hybrid ranking RPC:
-- combines structured filtering with vector similarity score.
create or replace function public.match_professionals_hybrid(
  p_city text default null,
  p_specialty text default null,
  p_gender text default null,
  p_min_weight_kg integer default null,
  p_query_embedding vector(768) default null,
  p_limit integer default 5
)
returns table (
  id uuid,
  name text,
  specialty text,
  city text,
  supports_weight_kg integer,
  score double precision
)
language sql
stable
as $$
  select
    pr.id,
    pr.name,
    pr.specialty,
    pr.city,
    pr.supports_weight_kg,
    (
      case
        when p_query_embedding is null or pr.embedding is null then 0.5
        else 1 - (pr.embedding <=> p_query_embedding)
      end
    )::double precision as score
  from public.professionals pr
  where (p_city is null or lower(pr.city) like '%' || lower(p_city) || '%' or lower(p_city) like '%' || lower(pr.city) || '%')
    and (p_specialty is null or lower(pr.specialty) like '%' || lower(p_specialty) || '%' or lower(p_specialty) like '%' || lower(pr.specialty) || '%')
    and (p_gender is null or lower(coalesce(pr.gender, '')) = lower(p_gender))
    and (p_min_weight_kg is null or coalesce(pr.supports_weight_kg, 0) >= p_min_weight_kg)
  order by score desc, pr.years_experience desc nulls last
  limit greatest(1, least(p_limit, 20));
$$;
