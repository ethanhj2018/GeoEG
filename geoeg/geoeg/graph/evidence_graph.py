"""NetworkX evidence graph. Nodes = clues; edges are added automatically from region overlap:

- support:    both regions overlap at comparable scale
- refine:     the smaller region lies inside the larger one (directed: broad → specific)
- contradict: the regions are disjoint
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import networkx as nx

from geoeg.eval.metrics import haversine_km
from geoeg.schema import Clue, Edge, Region

REFINE_RATIO = 0.5     # smaller radius must be < 50% of the larger to count as refinement


def _radius(r: Region) -> float:
    return r.radius_km or 0.0


def relation(a: Region, b: Region) -> tuple[str | None, str]:
    """Edge type between two regions (from a's point of view) and a short reason; None = unrelated/unknown."""
    if a.is_set and b.is_set:
        inter = set(a.countries) & set(b.countries)
        if not inter:
            return "contradict", "国家集合不相交"
        if set(a.countries) <= set(b.countries) or set(b.countries) <= set(a.countries):
            return "refine", f"国家集合包含（交集 {len(inter)} 国）"
        return "support", f"国家集合有交集（{len(inter)} 国）"
    if a.is_set or b.is_set:
        s, p = (a, b) if a.is_set else (b, a)
        country = p.country.strip().lower()
        if not p.is_point or not country:
            return None, ""
        inside = country in s.countries
        return ("support" if inside else "contradict"), f"{country} {'在' if inside else '不在'}国家集合内"
    if not (a.is_point and b.is_point):
        return None, ""
    d = haversine_km(a.lat, a.lon, b.lat, b.lon)
    ra, rb = _radius(a), _radius(b)
    if d > ra + rb:
        return "contradict", f"两区域相距 {d:.0f} km，超过半径和 {ra + rb:.0f} km"
    small, big = sorted((ra, rb))
    if big > 0 and small < REFINE_RATIO * big and d <= big:
        return "refine", f"小区域（{small:.0f} km）落在大区域（{big:.0f} km）内"
    return "support", f"区域重叠（距离 {d:.0f} km）"


def build_graph(clues: list[Clue], with_contradict: bool = True) -> tuple[nx.DiGraph, list[Edge]]:
    g = nx.DiGraph()
    for c in clues:
        g.add_node(c.id, **{k: v for k, v in c.to_dict().items() if k != "region"},
                   region=c.region.__dict__ if c.region else None)
    edges: list[Edge] = []
    located = [c for c in clues if c.region and (c.region.is_point or c.region.is_set)]
    for a, b in combinations(located, 2):
        typ, reason = relation(a.region, b.region)
        if typ is None or (typ == "contradict" and not with_contradict):
            continue
        src, dst = a.id, b.id
        if typ == "refine" and _radius(a.region) < _radius(b.region):
            src, dst = b.id, a.id                    # broad → specific
        if typ == "refine" and a.region.is_set and b.region.is_set and len(a.region.countries) < len(b.region.countries):
            src, dst = b.id, a.id
        e = Edge(src, dst, typ, reason)
        edges.append(e)
        g.add_edge(src, dst, type=typ, reason=reason)
    return g, edges


def contradiction_load(g: nx.DiGraph) -> dict[str, float]:
    """Per clue: Σ weight×confidence of the clues it contradicts (used to down-weight conflicting clues)."""
    load = {n: 0.0 for n in g.nodes}
    for u, v, d in g.edges(data=True):
        if d["type"] == "contradict":
            load[u] += g.nodes[v]["weight"] * g.nodes[v]["confidence"]
            load[v] += g.nodes[u]["weight"] * g.nodes[u]["confidence"]
    return load


def to_json(g: nx.DiGraph) -> dict:
    return nx.node_link_data(g, edges="edges")


def save(g: nx.DiGraph, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_json(g), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load(path: str | Path) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(Path(path).read_text(encoding="utf-8")), edges="edges")
