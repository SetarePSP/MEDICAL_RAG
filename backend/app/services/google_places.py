# google_places.py — Enriches matched professionals with real-world data from Google Places API.
# Adds address, phone number, Maps URL, and website. Skipped silently if no API key is set.

from typing import Any

import requests

from app.config import settings


def enrich_with_places(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not settings.google_places_api_key:
        return candidates

    enriched = []
    for c in candidates:
        q = f"{c.get('name', '')} {c.get('city', '')} Italy"
        search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        resp = requests.get(search_url, params={"query": q, "key": settings.google_places_api_key}, timeout=15)
        data = resp.json()
        results = data.get("results", [])
        if results:
            top = results[0]
            place_id = top.get("place_id")
            c["address"] = top.get("formatted_address")
            if place_id:
                c["maps_url"] = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_resp = requests.get(
                    details_url,
                    params={
                        "place_id": place_id,
                        "fields": "formatted_phone_number,international_phone_number,website,url,formatted_address,name",
                        "key": settings.google_places_api_key,
                    },
                    timeout=15,
                )
                details = details_resp.json().get("result", {})
                c["phone"] = details.get("formatted_phone_number") or details.get(
                    "international_phone_number"
                )
                c["address"] = details.get("formatted_address") or c.get("address")
                c["maps_url"] = details.get("url") or c.get("maps_url")
                c["place_name"] = details.get("name")
                c["website"] = details.get("website")
        enriched.append(c)
    return enriched
