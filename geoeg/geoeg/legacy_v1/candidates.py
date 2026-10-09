"""Candidate pool = VLM direct guesses (Top-5) ∪ GeoCLIP Top-K ∪ geocoded high-confidence clue regions,
merged when closer than `merge_km` (the earlier / higher-prior source keeps the slot, sources are unioned)."""
from __future__ import annotations

from geoeg.eval.metrics import haversine_km
from geoeg.schema import Candidate, Clue

CLUE_MIN_CONFIDENCE = 0.6
CLUE_LEVELS = ("poi", "city", "admin1")   # country-level regions are too coarse to be a point candidate


def from_vlm(guesses: list[dict], k: int) -> list[Candidate]:
    out = []
    for rank, g in enumerate(guesses[:k]):
        try:
            lat, lon = float(g["latitude"]), float(g["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        out.append(Candidate(lat=lat, lon=lon, country=g.get("country", ""), city=g.get("city", ""),
                             sources=["vlm"], prior=1.0 / (rank + 1),
                             evidence={"vlm_rank": rank + 1, "vlm_confidence": g.get("confidence")}))
    return out


def from_geoclip(top: list[dict]) -> list[Candidate]:
    return [Candidate(lat=t["lat"], lon=t["lon"], sources=["geoclip"], prior=t["prob"],
                      evidence={"geoclip_prob": round(t["prob"], 4)}) for t in top]


def from_clues(clues: list[Clue]) -> list[Candidate]:
    out = []
    for c in clues:
        r = c.region
        if r and r.is_point and r.level in CLUE_LEVELS and c.confidence >= CLUE_MIN_CONFIDENCE:
            out.append(Candidate(lat=r.lat, lon=r.lon, country=r.country, country_code=r.country_code,
                                 city=r.name.split(",")[0], sources=[f"clue:{c.id}"], prior=c.confidence))
    return out


def merge(groups: list[list[Candidate]], merge_km: float) -> list[Candidate]:
    pool: list[Candidate] = []
    for group in groups:
        for c in group:
            for p in pool:
                if haversine_km(p.lat, p.lon, c.lat, c.lon) <= merge_km:
                    p.sources = sorted(set(p.sources) | set(c.sources))
                    p.prior = max(p.prior, c.prior)
                    for k, v in c.evidence.items():
                        p.evidence.setdefault(k, v)
                    p.country = p.country or c.country
                    p.city = p.city or c.city
                    break
            else:
                pool.append(c)
    return pool
