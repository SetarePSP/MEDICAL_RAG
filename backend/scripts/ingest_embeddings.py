# ingest_embeddings.py — CLI script to generate and store vector embeddings for all professionals.
# Run after seeding the DB: python scripts/ingest_embeddings.py --only-missing
# Uses Vertex AI text-embedding-004 to create 768-dim vectors stored in Supabase.

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure `app` imports work when script is run directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.gemini_embeddings import embed_document
from app.services.supabase_client import get_supabase_client


def build_professional_text(row: dict[str, Any]) -> str:
    parts = [
        f"Name: {row.get('name', '')}",
        f"Specialty: {row.get('specialty', '')}",
        f"City: {row.get('city', '')}, Italy",
        f"Clinical summary: {row.get('clinical_summary', '')}",
        f"Experience years: {row.get('years_experience', 0)}",
        f"Supports weight kg: {row.get('supports_weight_kg', '')}",
    ]
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and store embeddings for professionals")
    parser.add_argument("--limit", type=int, default=0, help="Max professionals to process (0 = all)")
    parser.add_argument("--sleep-ms", type=int, default=120, help="Delay between calls to reduce rate spikes")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Process only rows where embedding is null",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first embedding error (useful to read API_KEY_SERVICE_BLOCKED guidance)",
    )
    args = parser.parse_args()

    sb = get_supabase_client()
    if sb is None:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")

    query = sb.table("professionals").select(
        "id,name,specialty,city,clinical_summary,years_experience,supports_weight_kg,embedding"
    )
    if args.only_missing:
        query = query.is_("embedding", "null")
    if args.limit > 0:
        query = query.limit(args.limit)

    rows = query.execute().data or []
    total = len(rows)
    print(f"Found {total} professionals to process")
    if total == 0:
        return

    success = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        pid = row["id"]
        try:
            text = build_professional_text(row)
            vector = embed_document(text)
            vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
            sb.table("professionals").update({"embedding": vector_str}).eq("id", pid).execute()
            success += 1
            print(f"[{i}/{total}] updated {pid}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] failed {pid}: {exc}")
            if args.fail_fast:
                print(
                    "Stopping (--fail-fast). For Vertex: enable Vertex AI API + billing on GOOGLE_CLOUD_PROJECT; "
                    "for API key: fix Generative Language API restrictions."
                )
                raise SystemExit(2) from exc
        time.sleep(max(0, args.sleep_ms) / 1000.0)

    print(f"Done. success={success}, failed={failed}")


if __name__ == "__main__":
    main()
