"""Evidence-graph data model: clue nodes, typed edges, location candidates. All JSON-serialisable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

ClueCategory = Literal["object", "text", "infra", "lighting", "environment"]
EdgeType = Literal["support", "refine", "contradict"]
RegionLevel = Literal["country", "admin1", "city", "poi", "countries", "none"]

CATEGORIES = ("object", "text", "infra", "lighting", "environment")
EDGE_TYPES = ("support", "refine", "contradict")


@dataclass
class Region:
    """Where a clue points. Either a geocoded area (lat/lon + bbox/radius) or a set of countries."""
    name: str = ""
    level: str = "none"
    lat: float | None = None
    lon: float | None = None
    radius_km: float | None = None
    bbox: list[float] | None = None          # [south, north, west, east] as returned by Nominatim
    country: str = ""                         # English country name of a geocoded region
    country_code: str = ""
    countries: list[str] = field(default_factory=list)  # level == "countries": lower-case English names

    @property
    def is_point(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def is_set(self) -> bool:
        return self.level == "countries" and bool(self.countries)


@dataclass
class Clue:
    id: str
    category: str
    description: str
    weight: int = 3                           # 1–5, importance of this clue type
    conclusion: str = ""                      # what the clue suggests, free text
    confidence: float = 0.5                   # 0–1
    query: str = ""                           # short English phrase for the detector
    text: str = ""                            # visible text (text clues / OCR)
    bbox: list[float] | None = None           # [x0, y0, x1, y1] pixels in the original image
    mask_area: float | None = None            # SAM 2 mask area / image area
    region: Region | None = None
    region_query: str = ""                    # place name the VLM proposed for geocoding
    sources: list[str] = field(default_factory=list)  # vlm / gdino / sam2 / ocr / nominatim / clues.py ...
    crop: str | None = None                   # path of the saved crop

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Clue":
        d = dict(d)
        if d.get("region") is not None and not isinstance(d["region"], Region):
            d["region"] = Region(**d["region"])
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


@dataclass
class Edge:
    src: str
    dst: str
    type: str                                 # support / refine / contradict
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    lat: float
    lon: float
    country: str = ""
    country_code: str = ""
    city: str = ""
    display_name: str = ""
    score: float = 0.0
    sources: list[str] = field(default_factory=list)     # vlm / geoclip / clue:<id>
    prior: float = 0.0                                    # source prior (VLM rank / GeoCLIP prob)
    supporting_clues: list[str] = field(default_factory=list)
    contradicting_clues: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)  # g3 / geoclip similarity, score breakdown

    def to_dict(self) -> dict:
        return asdict(self)
