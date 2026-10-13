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
        if a.negate and b.negate:
            return None, ""
        if a.negate or b.negate:                      # "in S" vs "not in L"
            pos, neg = (b, a) if a.negate else (a, b)
            rest = set(pos.countries) - set(neg.countries)
            if not rest:
                return "contradict", "country sets are disjoint"
            return "support", f"{len(rest)} countries satisfy both"
        inter = set(a.countries) & set(b.countries)
        if not inter:
            return "contradict", "country sets are disjoint"
        if set(a.countries) <= set(b.countries) or set(b.countries) <= set(a.countries):
            return "refine", f"country sets nested ({len(inter)} in common)"
        return "support", f"country sets overlap ({len(inter)} in common)"
    if a.is_set or b.is_set:
        s, p = (a, b) if a.is_set else (b, a)
        inside = s.has_country(p.country) if p.is_point else None
        if inside is None:
            return None, ""
        return ("support" if inside else "contradict"), f"{p.country} {'matches' if inside else 'violates'} {s.name}"
    if not (a.is_point and b.is_point):
        return None, ""
    d = haversine_km(a.lat, a.lon, b.lat, b.lon)
    ra, rb = _radius(a), _radius(b)
    if d > ra + rb:
        return "contradict", f"regions are {d:.0f} km apart, more than the sum of radii {ra + rb:.0f} km"
    small, big = sorted((ra, rb))
    if big > 0 and small < REFINE_RATIO * big and d <= big:
        return "refine", f"smaller region ({small:.0f} km) lies inside larger region ({big:.0f} km)"
    return "support", f"regions overlap (distance {d:.0f} km)"


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
