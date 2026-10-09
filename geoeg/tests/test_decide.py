"""v2 decision logic and deterministic text evidence."""
import math

from geoeg.conclude.decide import DEFAULT_PARAMS, decide
from geoeg.schema import Region
from geoeg.tools import textclues as tc

LEFT = ["australia", "japan", "united kingdom"]


def _f(evidence, pool, local=None, baseline=None):
    return {"baseline": baseline or {"lat": pool[0]["lat"], "lon": pool[0]["lon"], "country": pool[0]["country"],
                                     "confidence": "medium"},
            "evidence": evidence, "pool": pool, "local": local or {}}


def _pool():
    return [{"id": "baseline", "lat": 48.86, "lon": 2.35, "country": "France", "g3_sim": 0.20, "geoclip_sim": 0.20},
            {"id": "g1", "lat": 35.68, "lon": 139.69, "country": "Japan", "g3_sim": 0.30, "geoclip_sim": 0.35},
            {"id": "g2", "lat": 51.50, "lon": -0.12, "country": "", "g3_sim": 0.25, "geoclip_sim": 0.30}]


def test_default_keeps_baseline_without_evidence():
    d = decide(_f([], _pool()))
    assert d["action"] == "baseline" and d["final"]["id"] == "baseline"
    assert [c["id"] for c in d["top"]][0] == "baseline" and len(d["top"]) == 3


def test_override_only_with_strong_contradiction_and_checked_alternative():
    hira = {"id": "e1", "kind": "script", "strength": 3.0,
            "region": Region(name="script:hiragana", level="countries", countries=["japan"]).__dict__}
    d = decide(_f([hira], _pool()))
    assert d["action"] == "override" and d["final"]["id"] == "g1"
    # weak evidence (below override_min_contra) does not overturn the VLM
    weak = dict(hira, strength=1.0)
    assert decide(_f([weak], _pool()))["action"] == "baseline"
    # an alternative with unknown country is not accepted as "consistent"
    pool = _pool()
    pool[1]["country"] = ""
    assert decide(_f([hira], pool))["final"]["id"] == "baseline"


def test_right_hand_traffic_is_negated_left_set():
    r = Region(name="driving-right", level="countries", countries=LEFT, negate=True)
    assert r.has_country("Japan") is False and r.has_country("France") is True
    assert r.has_country("USA") is True and Region(level="countries", countries=["united states"]).has_country("USA")


def test_refinement_moves_only_with_enough_gain():
    pool = _pool()
    local = {"baseline": [{"lat": 48.80, "lon": 2.13, "g3_sim": 0.40, "geoclip_sim": 0.40, "d_km": 17.0},
                          {"lat": 49.0, "lon": 2.5, "g3_sim": 0.10, "geoclip_sim": 0.10, "d_km": 19.0}]}
    p = {**DEFAULT_PARAMS, "refine_radius_km": 50.0, "refine_dist_penalty": 0.1, "refine_min_gain": 0.1}
    d = decide(_f([], pool, local), p)
    assert d["final"]["refined_from"] == [48.86, 2.35] and math.isclose(d["final"]["lat"], 48.80)
    p["refine_min_gain"] = 5.0
    assert "refined_from" not in decide(_f([], pool, local), p)["final"]
    p["refine_radius_km"] = 10.0                                      # local points beyond radius are ignored
    assert "refined_from" not in decide(_f([], pool, local), {**p, "refine_min_gain": 0.0})["final"]


def test_script_detection():
    found = {s: r for s, r, _ in tc.script_regions(["ありがとう 東京", "Hello"])}
    assert "HIRAGANA" in found and "CJK" not in found and found["HIRAGANA"].countries == ["japan"]
    found = {s for s, _, _ in tc.script_regions(["Ελληνικά"])}
    assert found == {"GREEK"}
    assert tc.script_regions(["Bonjour Paris 2014"]) == []


def test_place_queries_filter():
    q = tc.place_queries(["LucT08/2014", "Denderleeuw", "12345", "Rue de la Paix", "ab"])
    assert "Denderleeuw" in q and "Rue de la Paix" in q and "12345" not in q and "ab" not in q


def test_calling_code_regex():
    assert [m.group(1) for m in tc.CALLING.finditer("Tel +32 2 555 12 12, fax +1-800")] == ["32", "1"]
