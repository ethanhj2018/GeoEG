"""v2 decision: the VLM's direct prediction is the anchor; the evidence graph verifies, re-ranks and refines it.

`decide(features, params)` is a pure function of the per-image feature dict that pipeline.run() stores in
result["decision"]["features"], so eval/run_eval.py can grid-search `params` offline without re-running models.

1. Anchor = VLM baseline coordinate.
2. A readable place name confirmed by OSM (and by a co-occurring admin name or a second text line) that the
   baseline falls outside of replaces the baseline (strength ≥ place_decisive).
   Otherwise override only when deterministic evidence (script, calling code, plate, geocoded place name, driving side —
   all from OCR / tables / OSM, never from the VLM's own location guess) contradicts the baseline with total
   strength ≥ override_min_contra and more than it supports it. The replacement is the best-scoring candidate
   that no evidence contradicts. Optionally (retrieval_margin < inf) also when retrieval similarity strongly
   prefers another candidate.
3. Refine around the anchor: GeoCLIP-gallery points within refine_radius_km, re-ranked by G3 + GeoCLIP
   similarity with a distance penalty; move only if the gain beats refine_min_gain.
"""
from __future__ import annotations

import math

from geoeg.eval.metrics import haversine_km
from geoeg.schema import Region

DEFAULT_PARAMS = {
    "place_decisive": 4.0,            # strength at which a confirmed place name overrides the VLM (inf = never)
    "override_min_contra": 2.0,
    "retrieval_margin": math.inf,
    "retrieval_max_km": math.inf,     # retrieval may only move the anchor this far from the VLM answer
    "w_g3": 1.0, "w_gc": 1.0, "w_sup": 1.0, "w_con": 1.0, "w_prox": 0.5,
    "refine_radius_km": 0.0,          # 0 = no refinement
    "refine_alpha": 0.5,              # G3 vs GeoCLIP weight inside the local set
    "refine_dist_penalty": 0.5,
    "refine_min_gain": 0.1,
    "refine_skip_high": False,        # do not move a baseline the VLM marked "high" confidence
}

STRENGTH = {"script": 3.0, "calling_code": 3.0, "plate": 4.0, "admin": 3.0, "place": 5.0, "place_unconfirmed": 1.0,
            "driving_side": 1.0}


def region_of(ev: dict) -> Region:
    r = ev["region"]
    return r if isinstance(r, Region) else Region(**r)


def contains(r: Region, lat: float, lon: float, country: str) -> bool | None:
    if r.is_set:
        return r.has_country(country)
    if not r.is_point:
        return None
    if r.bbox:
        s, n, w, e = r.bbox
        pad = 0.25
        return s - pad <= lat <= n + pad and w - pad <= lon <= e + pad
    return haversine_km(r.lat, r.lon, lat, lon) <= (r.radius_km or 25.0)


def is_firm(ev: dict) -> bool:
    """Evidence allowed to overturn the VLM: tables, scripts, admin names and *confirmed* place names.
    A single unconfirmed name ("Nestlé", "St Clare") exists in many countries and only adds a little score."""
    return not (ev["kind"] == "place" and ev.get("confirmed") is False)


def evidence_vs(c: dict, evidence: list[dict]) -> tuple[float, float, list[str], list[str], float]:
    """→ (support, contradiction, supporting ids, contradicting ids, firm contradiction)."""
    sup = con = firm = 0.0
    s_ids, c_ids = [], []
    for ev in evidence:
        inside = contains(region_of(ev), c["lat"], c["lon"], c.get("country", ""))
        if inside is None:
            continue
        if inside:
            sup += ev["strength"]
            s_ids.append(ev["id"])
        else:
            con += ev["strength"]
            firm += ev["strength"] if is_firm(ev) else 0.0
            c_ids.append(ev["id"])
    return sup, con, s_ids, c_ids, firm


def _norm(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    return [0.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo) for v in vals]


def score_pool(f: dict, p: dict) -> list[dict]:
    pool = [dict(c) for c in f["pool"]]
    if not pool:
        return []
    g3n = _norm([c.get("g3_sim", 0.0) for c in pool])
    gcn = _norm([c.get("geoclip_sim", 0.0) for c in pool])
    b = f.get("baseline")
    for c, a, g in zip(pool, g3n, gcn):
        sup, con, s_ids, c_ids, firm = evidence_vs(c, f["evidence"])
        c["contra_firm"] = firm
        prox = math.exp(-haversine_km(c["lat"], c["lon"], b["lat"], b["lon"]) / 200.0) if b else 0.0
        c["retrieval"] = p["w_g3"] * a + p["w_gc"] * g
        c["support"], c["contra"] = sup, con
        c["supporting"], c["contradicting"] = s_ids, c_ids
        c["score"] = round(c["retrieval"] + p["w_sup"] * sup - p["w_con"] * con + p["w_prox"] * prox, 4)
    return pool


def refine(f: dict, anchor: dict, p: dict) -> tuple[dict, str]:
    R = p["refine_radius_km"]
    local = [x for x in f.get("local", {}).get(anchor["id"], []) if x["d_km"] <= R]
    if R <= 0 or not local:
        return anchor, ""
    if p["refine_skip_high"] and anchor["id"] == "baseline" and (f.get("baseline") or {}).get("confidence") == "high":
        return anchor, ""
    pts = [{"lat": anchor["lat"], "lon": anchor["lon"], "g3_sim": anchor.get("g3_sim", 0.0),
            "geoclip_sim": anchor.get("geoclip_sim", 0.0), "d_km": 0.0}] + local
    g3n = _norm([x["g3_sim"] for x in pts])
    gcn = _norm([x["geoclip_sim"] for x in pts])
    a = p["refine_alpha"]
    scores = [a * g + (1 - a) * c - p["refine_dist_penalty"] * x["d_km"] / R for x, g, c in zip(pts, g3n, gcn)]
    best = max(range(len(pts)), key=lambda i: scores[i])
    if best == 0 or scores[best] - scores[0] < p["refine_min_gain"]:
        return anchor, ""
    moved = {**anchor, "lat": pts[best]["lat"], "lon": pts[best]["lon"], "refined_from": [anchor["lat"], anchor["lon"]],
             "g3_sim": pts[best]["g3_sim"], "geoclip_sim": pts[best]["geoclip_sim"]}
    return moved, f"refined {pts[best]['d_km']:.0f} km within {R:.0f} km"


def decide(f: dict, params: dict | None = None, k: int = 5) -> dict:
    """→ {"final": cand, "top": [cand...], "action": "baseline|override|retrieval|fallback", "note": str}"""
    p = {**DEFAULT_PARAMS, **(params or {})}
    pool = score_pool(f, p)
    if not pool:
        return {"final": None, "top": [], "action": "none", "note": "no candidates"}
    by_id = {c["id"]: c for c in pool}
    base = by_id.get("baseline")
    others = sorted((c for c in pool if c["id"] != "baseline"), key=lambda c: c["score"], reverse=True)
    action, note = "baseline", "kept the VLM prediction"
    anchor = base
    # A readable, corroborated place name (OCR / transcribed text → OSM) is near-decisive: if the anchor is not
    # inside it, the place itself becomes the anchor. Several such names are disambiguated by the other evidence.
    decisive = [ev for ev in f["evidence"] if ev["kind"] == "place" and ev.get("confirmed")
                and ev["strength"] >= p["place_decisive"] and region_of(ev).level in ("poi", "city")]
    place_cands = [by_id[f"p:{ev['id']}"] for ev in decisive if f"p:{ev['id']}" in by_id]
    base_in_place = base is not None and any(
        contains(region_of(ev), base["lat"], base["lon"], base.get("country", "")) for ev in decisive)
    if place_cands and not base_in_place:
        anchor = max(place_cands, key=lambda c: (c["support"] - c["contra"], c["score"]))
        action = "place"
        note = f"readable place name {anchor['city'] or anchor['id']} overrides the VLM prediction"
    elif base is None:
        anchor, action, note = (others[0], "fallback", "no valid VLM prediction; best retrieval candidate") \
            if others else (None, "none", "")
    elif base["contra_firm"] >= p["override_min_contra"] and base["contra_firm"] > base["support"]:
        # replacement must be checkably consistent: not contradicted, and either supported or with a known
        # country (an unknown country makes every country-set test pass vacuously)
        clean = [c for c in others if c["contra"] == 0 and (c["support"] > 0 or c.get("country"))]
        if clean:
            anchor, action = clean[0], "override"
            note = f"VLM prediction contradicted by {', '.join(base['contradicting'])}; switched to {anchor['id']}"
    else:
        # retrieval may replace the VLM answer, but only with a candidate within retrieval_max_km of it: the VLM
        # keeps the coarse (country / region) call, retrieval sharpens it
        near = [c for c in others if haversine_km(c["lat"], c["lon"], base["lat"], base["lon"]) <= p["retrieval_max_km"]]
        best = near[0] if near else None                     # `others` is sorted by full score
        if best and best["retrieval"] - base["retrieval"] > p["retrieval_margin"] and best["contra"] <= base["contra"]:
            anchor, action = best, "retrieval"
            note = f"retrieval strongly prefers {anchor['id']} over the VLM prediction"
    final, rnote = refine(f, anchor, p)
    if rnote:
        note += "; " + rnote
    top = [final]
    for c in sorted(pool, key=lambda c: (c["id"] != anchor["id"], -c["score"])):
        if c["id"] == anchor["id"]:
            continue
        if all(haversine_km(c["lat"], c["lon"], t["lat"], t["lon"]) > 25 for t in top):
            top.append(c)
        if len(top) >= k:
            break
    return {"final": final, "top": top, "action": action, "note": note}
