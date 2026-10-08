import pytest

from geoeg.conclude.candidates import merge
from geoeg.conclude.scoring import contains, score
from geoeg.graph.evidence_graph import build_graph, contradiction_load, relation
from geoeg.schema import Candidate, Clue, Region

BELGIUM = Region(name="Belgium", level="country", lat=50.64, lon=4.67, radius_km=180,
                 bbox=[49.49, 51.55, 2.38, 6.41], country="Belgium", country_code="be")
BRUSSELS = Region(name="Brussels, Belgium", level="city", lat=50.85, lon=4.35, radius_km=12,
                  bbox=[50.76, 50.91, 4.24, 4.48], country="Belgium", country_code="be")
JAPAN = Region(name="Japan", level="country", lat=36.5, lon=139.0, radius_km=1200, country="Japan", country_code="jp")
LEFT = Region(name="driving-left", level="countries", countries=["japan", "united kingdom"])


def test_relations():
    assert relation(BELGIUM, BRUSSELS)[0] == "refine"
    assert relation(BELGIUM, JAPAN)[0] == "contradict"
    assert relation(LEFT, JAPAN)[0] == "support"
    assert relation(LEFT, BELGIUM)[0] == "contradict"


def test_graph_edges_and_direction():
    clues = [Clue("a", "text", "法语+荷兰语路牌", 5, confidence=0.9, region=BELGIUM),
             Clue("b", "object", "布鲁塞尔地标", 4, confidence=0.8, region=BRUSSELS),
             Clue("c", "infra", "靠左行驶", 3, confidence=0.6, region=LEFT)]
    g, edges = build_graph(clues)
    types = {(e.src, e.dst): e.type for e in edges}
    assert types[("a", "b")] == "refine"            # broad → specific
    assert "contradict" in types.values()
    load = contradiction_load(g)
    assert load["c"] > 0 and load["a"] > 0
    g2, edges2 = build_graph(clues, with_contradict=False)
    assert all(e.type != "contradict" for e in edges2)


def test_contains_and_scoring_prefers_consistent_candidate():
    clues = [Clue("a", "text", "路牌", 5, confidence=0.9, region=BELGIUM),
             Clue("b", "object", "地标", 4, confidence=0.8, region=BRUSSELS)]
    brussels = Candidate(50.85, 4.35, country="Belgium", country_code="be", sources=["vlm"], prior=0.5)
    tokyo = Candidate(35.68, 139.69, country="Japan", country_code="jp", sources=["vlm"], prior=1.0)
    assert contains(BELGIUM, brussels) and not contains(BELGIUM, tokyo)
    w = {"w_clues": 1, "w_contradict": 0.5, "w_g3": 0, "w_geoclip": 0, "w_vlm": 1}
    ranked = score([tokyo, brussels], clues, {}, w)
    assert ranked[0] is brussels
    assert ranked[0].supporting_clues == ["a", "b"] and ranked[1].contradicting_clues == ["a", "b"]


def test_repeated_region_counts_once():
    w = {"w_clues": 1, "w_contradict": 0.5, "w_g3": 0, "w_geoclip": 0, "w_vlm": 0}
    echo = [Clue(f"e{i}", "object", "同一结论", 3, confidence=0.8, region=BRUSSELS) for i in range(4)]
    one = score([Candidate(50.85, 4.35)], echo[:1], {}, w)[0].score
    four = score([Candidate(50.85, 4.35)], echo, {}, w)[0].score
    assert four == one == pytest.approx(3 * 0.8)


def test_merge_unions_sources():
    a = [Candidate(50.85, 4.35, sources=["vlm"], prior=1.0)]
    b = [Candidate(50.86, 4.36, sources=["geoclip"], prior=0.02), Candidate(35.0, 139.0, sources=["geoclip"])]
    pool = merge([a, b], merge_km=25)
    assert len(pool) == 2 and pool[0].sources == ["geoclip", "vlm"]
