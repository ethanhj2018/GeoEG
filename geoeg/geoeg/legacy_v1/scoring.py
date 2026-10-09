"""Hierarchical candidate scoring.

score = w_clues   × Σ_{regions containing the candidate} max_{clues → region} weight × confidence × consistency
      − w_contra  × Σ_{regions excluding the candidate}  max_{clues → region} weight × confidence × consistency
      + w_g3 × G3 similarity + w_geoclip × GeoCLIP similarity   (min-max normalised over the pool)
      + w_vlm × VLM prior (1/rank)

`consistency` = 1 / (1 + contradiction load in the evidence graph), so a clue that conflicts with
the rest counts less. Containment is checked per region level: bbox for geocoded areas (padded),
radius as fallback, country-name membership for country sets (driving side).
"""
from __future__ import annotations

import numpy as np

from geoeg.eval.metrics import haversine_km
from geoeg.schema import Candidate, Clue, Region
from geoeg.tools.geocode import LEVEL_RADIUS_KM

BBOX_PAD_DEG = 0.25


def contains(r: Region, cand: Candidate) -> bool | None:
    """True / False, or None when the region cannot be tested against this candidate."""
    if r.is_set:
        if not cand.country:
            return None
        return cand.country.lower() in r.countries
    if not r.is_point:
        return None
    if r.level == "country" and r.country_code and cand.country_code:
        return r.country_code.lower() == cand.country_code.lower()
    if r.bbox:
        s, n, w, e = r.bbox
        p = BBOX_PAD_DEG
        if s - p <= cand.lat <= n + p and w - p <= cand.lon <= e + p:
            return True
        if r.level in ("poi", "city", "admin1", "country"):
            return False
    radius = r.radius_km or LEVEL_RADIUS_KM.get(r.level, 50.0)
    return haversine_km(r.lat, r.lon, cand.lat, cand.lon) <= radius


def _region_key(r: Region) -> tuple:
    if r.is_set:
        return ("set", tuple(sorted(r.countries)))
    return ("pt", r.name or f"{r.lat:.3f},{r.lon:.3f}", r.level)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or float(x.max() - x.min()) < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def score(cands: list[Candidate], clues: list[Clue], consistency: dict[str, float], w: dict) -> list[Candidate]:
    g3 = np.array([c.evidence.get("g3_sim", 0.0) for c in cands], dtype=float)
    gc = np.array([c.evidence.get("geoclip_sim", 0.0) for c in cands], dtype=float)
    g3n, gcn = _minmax(g3), _minmax(gc)
    for i, c in enumerate(cands):
        best: dict[tuple, tuple[bool, float]] = {}
        c.supporting_clues, c.contradicting_clues = [], []
        for cl in clues:
            if not cl.region:
                continue
            inside = contains(cl.region, c)
            if inside is None:
                continue
            v = cl.weight * cl.confidence * consistency.get(cl.id, 1.0)
            (c.supporting_clues if inside else c.contradicting_clues).append(cl.id)
            # Clues that resolve to the same region are one piece of evidence, not several (the VLM often
            # repeats one conclusion across clues): count only the strongest clue per region.
            k = _region_key(cl.region)
            if k not in best or v > best[k][1]:
                best[k] = (inside, v)
        sup = sum(v for inside, v in best.values() if inside)
        con = sum(v for inside, v in best.values() if not inside)
        vlm_prior = c.prior if "vlm" in c.sources else 0.0
        parts = {"clues": w["w_clues"] * sup, "contradict": -w["w_contradict"] * con,
                 "g3": w["w_g3"] * float(g3n[i]), "geoclip": w["w_geoclip"] * float(gcn[i]),
                 "vlm": w["w_vlm"] * vlm_prior}
        c.score = round(sum(parts.values()), 4)
        c.evidence["score_parts"] = {k: round(v, 4) for k, v in parts.items()}
    return sorted(cands, key=lambda c: c.score, reverse=True)
