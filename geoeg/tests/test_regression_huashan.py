"""Regression (web upload 2026-10-09): caption "华山 / HUASHAN MOUNTAIN / (陕西)" was predicted as Huangshan.

Readable place names must become candidates and win over the VLM guess (Xi'an, ~50 km off) and over GeoCLIP's
visually similar granite peaks (Huangshan). Nominatim responses are replayed from a recorded fixture, so the
test needs no network.
"""
import json
from pathlib import Path

from geoeg.conclude.decide import decide
from geoeg.config import load_config
from geoeg.eval.metrics import haversine_km
from geoeg.pipeline import TEXT, collect_evidence, place_candidates
from geoeg.tools.cache import Cache
from geoeg.tools.geosleuth import GeoSleuth

FIX = json.loads((Path(__file__).parent / "fixtures" / "huashan_nominatim.json").read_text(encoding="utf-8"))
HUASHAN = (34.48, 110.08)


class ReplayGeocoder:
    calls = 0

    def search(self, q, limit=1, level=None):
        return FIX["search"].get(q, [])[:limit]


def test_huashan_caption_wins(tmp_path):
    cfg = load_config(mode="local", lang="en")
    cfg["paths"]["geosleuth_cache"] = str(tmp_path / "gs")
    sleuth = GeoSleuth(cfg, Cache(str(tmp_path / "c.sqlite")))
    evidence, clues = collect_evidence(FIX["texts"], "unknown", sleuth, ReplayGeocoder(), cfg, TEXT["en"])
    kinds = {(e["kind"], e.get("confirmed")) for e in evidence}
    assert ("place", True) in kinds                       # the caption produced a confirmed place
    pool = [{"id": "baseline", "lat": 34.7589, "lon": 109.6541, "country": "China", "city": "Xi'an",
             "sources": ["vlm"], "g3_sim": 0.20, "geoclip_sim": 0.20},
            # GeoCLIP's favourites for this photo: Huangshan (visually similar granite peaks)
            {"id": "g1", "lat": 30.094, "lon": 118.172, "country": "", "city": "", "sources": ["geoclip"],
             "g3_sim": 0.35, "geoclip_sim": 0.40},
            {"id": "g2", "lat": 29.829, "lon": 118.265, "country": "", "city": "", "sources": ["geoclip"],
             "g3_sim": 0.30, "geoclip_sim": 0.35}]
    pool += [dict(c, g3_sim=0.25, geoclip_sim=0.25) for c in place_candidates(evidence)]
    assert any(c["id"].startswith("p:") for c in pool)
    feats = {"baseline": {"lat": 34.7589, "lon": 109.6541, "country": "China", "confidence": "medium"},
             "evidence": evidence, "pool": pool, "local": {}}
    d = decide(feats, cfg["decision"]["params"])
    top1 = d["final"]
    assert d["action"] == "place"
    assert haversine_km(top1["lat"], top1["lon"], *HUASHAN) <= 50
    assert all(haversine_km(c["lat"], c["lon"], 30.1, 118.2) > 50 for c in d["top"][:1])   # not Huangshan


def test_cjk_place_queries():
    from geoeg.tools.textclues import place_queries
    q = place_queries(["华山", "(陕西)", "HUASHAN MOUNTAIN", "ab", "1234"])
    assert q[:3] == ["华山", "陕西", "HUASHAN MOUNTAIN"]
