"""Nominatim forward / reverse geocoding with User-Agent, ≤1 req/s and SQLite caching."""
from __future__ import annotations

import math
import os

import requests

from geoeg.tools.cache import Cache, RateLimiter

# Fallback radius when Nominatim gives no usable bbox.
LEVEL_RADIUS_KM = {"poi": 2.0, "city": 25.0, "admin1": 250.0, "country": 1000.0}

# Nominatim `addresstype` values accepted for each clue level (poi accepts anything).
_ADMIN = {"state", "province", "region", "state_district", "county", "territory", "district", "island", "archipelago"}
_CITY = {"city", "town", "village", "municipality", "hamlet", "suburb", "borough", "city_district", "county",
         "quarter", "neighbourhood", "island"}
LEVEL_ADDRESSTYPES = {"country": {"country"} | _ADMIN, "admin1": _ADMIN | {"country"}, "city": _CITY | _ADMIN}


class Geocoder:
    def __init__(self, cfg: dict, cache: Cache):
        g = cfg["geocode"]
        self.endpoint = g["endpoint"].rstrip("/")
        self.headers = {"User-Agent": g["user_agent"], "Accept-Language": "en"}
        self.email = os.environ.get(g.get("contact_env", ""), "").strip()
        self.limiter = RateLimiter(g["min_interval_s"])
        self.cache = cache
        self.calls = 0

    def _get(self, path: str, params: dict):
        params = {**params, "format": "jsonv2"}
        if self.email:
            params["email"] = self.email

        def fetch():
            self.limiter.wait()
            self.calls += 1
            r = requests.get(f"{self.endpoint}/{path}", params=params, headers=self.headers, timeout=30)
            r.raise_for_status()
            return r.json()
        return self.cache.cached("nominatim:" + path, params, fetch)

    def search(self, query: str, limit: int = 1, level: str | None = None) -> list[dict]:
        """Forward geocode. With `level`, hits whose OSM addresstype does not fit that level are dropped
        (e.g. a VLM asks for a country but Nominatim matches a street of the same name)."""
        if not query.strip():
            return []
        hits = [_hit(h) for h in self._get("search", {"q": query, "limit": max(limit, 5), "addressdetails": 1})]
        if level in LEVEL_ADDRESSTYPES:
            hits = [h for h in hits if h["addresstype"] in LEVEL_ADDRESSTYPES[level]]
        return hits[:limit]

    def reverse(self, lat: float, lon: float, zoom: int = 10) -> dict:
        h = self._get("reverse", {"lat": round(lat, 5), "lon": round(lon, 5), "zoom": zoom, "addressdetails": 1})
        if not h or "error" in h:
            return {}
        a = h.get("address", {})
        return {"country": a.get("country", ""), "country_code": a.get("country_code", ""),
                "city": a.get("city") or a.get("town") or a.get("village") or a.get("municipality")
                or a.get("county") or a.get("state", ""),
                "display_name": h.get("display_name", "")}


def _hit(h: dict) -> dict:
    lat, lon = float(h["lat"]), float(h["lon"])
    bbox = [float(x) for x in h.get("boundingbox", [])] or None   # [south, north, west, east]
    radius = None
    if bbox:
        # half-diagonal of the bbox as a rough extent
        dlat = (bbox[1] - bbox[0]) * 111.0
        dlon = (bbox[3] - bbox[2]) * 111.0 * math.cos(math.radians(lat))
        radius = 0.5 * math.hypot(dlat, dlon)
    return {"name": h.get("display_name", ""), "lat": lat, "lon": lon, "bbox": bbox, "radius_km": radius,
            "addresstype": h.get("addresstype", ""), "country_code": h.get("address", {}).get("country_code", ""),
            "country": h.get("address", {}).get("country", "")}
