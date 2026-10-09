"""Deterministic location evidence from visible text and observations (no VLM location guesses involved).

Sources, strongest first:
- writing system of OCR / transcribed text (Unicode ranges → country set)
- international calling codes (+NN) and Chinese plate prefixes (geo-sleuth offline tables)
- place names in the text, geocoded by Nominatim (kept only if the hit is a settlement / admin area / country)
- driving side observed by the VLM (geo-sleuth table; "right" is modelled as "not a left-driving country",
  because the right-hand table is incomplete)
"""
from __future__ import annotations

import re
import unicodedata

from geoeg.schema import Region, normalize_country

# Unicode script → countries where that script is the main public script. Han and Latin are too broad.
SCRIPT_COUNTRIES = {
    "HANGUL": ["south korea", "north korea"],
    "HIRAGANA": ["japan"], "KATAKANA": ["japan"],
    "THAI": ["thailand"],
    "LAO": ["laos"],
    "KHMER": ["cambodia"],
    "MYANMAR": ["myanmar"],
    "GEORGIAN": ["georgia"],
    "ARMENIAN": ["armenia"],
    "HEBREW": ["israel", "palestine"],
    "GREEK": ["greece", "cyprus"],
    "ETHIOPIC": ["ethiopia", "eritrea"],
    "SINHALA": ["sri lanka"],
    "TIBETAN": ["china", "bhutan", "nepal", "india"],
    "DEVANAGARI": ["india", "nepal"],
    "BENGALI": ["bangladesh", "india"],
    "TAMIL": ["india", "sri lanka", "singapore", "malaysia"],
    "GUJARATI": ["india"], "GURMUKHI": ["india"], "KANNADA": ["india"], "MALAYALAM": ["india"],
    "TELUGU": ["india"], "ORIYA": ["india"],
    "MONGOLIAN": ["china", "mongolia"],
    "CYRILLIC": ["russia", "ukraine", "belarus", "bulgaria", "serbia", "north macedonia", "montenegro",
                 "bosnia and herzegovina", "kazakhstan", "kyrgyzstan", "tajikistan", "mongolia", "moldova"],
    "ARABIC": ["saudi arabia", "united arab emirates", "egypt", "morocco", "algeria", "tunisia", "libya", "jordan",
               "lebanon", "syria", "iraq", "kuwait", "qatar", "bahrain", "oman", "yemen", "sudan", "iran",
               "afghanistan", "pakistan", "palestine", "mauritania", "israel"],
    "CJK": ["china", "taiwan", "japan", "singapore", "malaysia", "south korea"],   # Han characters
}
_MIN_CHARS = {"CJK": 2, "CYRILLIC": 3, "ARABIC": 3}

CN_PLATE = re.compile(r"([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])\s?·?\s?([A-Z])")
CALLING = re.compile(r"(?<![\w+])\+\s?(\d{1,3})(?=[\s\-()\d])")
PLACE_TEXT = re.compile(r"[^\W\d_]{3,}", re.U)

_SETTLEMENT_TYPES = {"city", "town", "village", "municipality", "hamlet", "suburb", "borough", "state", "province",
                     "region", "county", "state_district", "country", "island", "territory"}


def script_of(ch: str) -> str | None:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    if name.startswith("CJK UNIFIED") or name.startswith("CJK COMPATIBILITY IDEOGRAPH"):
        return "CJK"
    first = name.split(" ")[0]
    return first if first in SCRIPT_COUNTRIES else None


def script_regions(texts: list[str]) -> list[tuple[str, Region, str]]:
    """[(script, region, evidence text)] for every distinctive script seen in the texts."""
    counts: dict[str, int] = {}
    sample: dict[str, str] = {}
    for t in texts:
        for ch in t:
            s = script_of(ch)
            if s:
                counts[s] = counts.get(s, 0) + 1
                sample.setdefault(s, t)
    # kana outranks Han (Japanese text mixes both): drop the broad CJK set when kana is present
    if counts.get("HIRAGANA", 0) + counts.get("KATAKANA", 0) >= 2:
        counts.pop("CJK", None)
    if counts.get("HANGUL", 0) >= 2:
        counts.pop("CJK", None)
    out = []
    for s, n in counts.items():
        if n >= _MIN_CHARS.get(s, 2):
            out.append((s, Region(name=f"script:{s.lower()}", level="countries",
                                  countries=sorted(set(SCRIPT_COUNTRIES[s]))), sample[s]))
    return out


def calling_code_regions(texts: list[str], sleuth) -> list[tuple[str, Region, str]]:
    out, seen = [], set()
    for t in texts:
        for m in CALLING.finditer(t):
            code = "+" + m.group(1)
            if code in seen:
                continue
            seen.add(code)
            try:
                res = sleuth.lookup("calling-code", code)
            except Exception:
                continue
            names = sorted({normalize_country(x["country"]) for x in res.get("matches", []) if x.get("country")})
            if names:
                out.append((code, Region(name=f"calling-code:{code}", level="countries", countries=names), t))
    return out


def plate_regions(texts: list[str], sleuth) -> list[tuple[str, Region, str]]:
    out, seen = [], set()
    for t in texts:
        for m in CN_PLATE.finditer(t):
            key = m.group(1) + m.group(2)
            if key in seen:
                continue
            seen.add(key)
            try:
                res = sleuth.lookup("plate", key)
            except Exception:
                continue
            if res.get("matches"):
                adm = res["matches"][0]
                out.append((key, Region(name=f"plate:{key} {adm.get('admin1', '')}{adm.get('admin2', '')}",
                                        level="countries", countries=["china"]), t))
    return out


_STRIP = " .,:;-|/()（）[]【】「」『』<>《》\"'“”‘’·•"
_HAS_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def _clean(t: str) -> str:
    return " ".join(t.split()).strip(_STRIP)


def place_queries(texts: list[str], max_queries: int = 8) -> list[str]:
    """Text lines worth geocoding, de-duplicated. CJK names are information-dense, so 2 characters suffice
    ("华山", "陕西"); other scripts need ≥4 characters with a run of ≥3 letters, and not mostly digits."""
    out, seen = [], set()
    for t in texts:
        q = _clean(t)
        if _HAS_CJK.search(q):
            ok = 2 <= len(q) <= 30 and len(_HAS_CJK.findall(q)) >= 2
        else:
            ok = 4 <= len(q) <= 40 and bool(PLACE_TEXT.search(q))
        if not ok or sum(c.isdigit() for c in q) > len(q) / 3:
            continue
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out[:max_queries]


# OSM categories that can name a location in a caption or sign. Shops, amenities, roads and buildings are
# excluded: "Paris Bakery" in Tokyo must not become evidence for Paris.
_PLACE_CATEGORIES = {"place", "boundary", "natural", "tourism", "historic", "leisure", "landuse", "railway",
                     "aerialway", "mountain_pass", "waterway"}
_ADMIN_TYPES = {"country", "state", "province", "region", "state_district", "county", "city", "municipality"}
_CITY_TYPES = {"city", "town", "village", "municipality", "hamlet", "suburb", "borough", "city_district", "quarter"}


def _level(h: dict) -> str:
    at = h["addresstype"]
    if at == "country":
        return "country"
    if at in {"state", "province", "region", "state_district", "county"}:
        return "admin1"
    if at in _CITY_TYPES:
        return "city"
    return "poi"


def _region(h: dict, level: str) -> Region:
    radius = h["radius_km"] or 0.0
    if level == "poi":
        radius = max(radius, 10.0)            # a peak / attraction / station point stands for its surroundings
    return Region(name=h["name"], level=level, lat=h["lat"], lon=h["lon"], radius_km=radius, bbox=h["bbox"],
                  country=h["country"], country_code=h["country_code"], importance=h.get("importance"))


def _inside(h: dict, r: Region) -> bool:
    if not r.bbox:
        return False
    s, n, w, e = r.bbox
    return s <= h["lat"] <= n and w <= h["lon"] <= e


def place_evidence(texts: list[str], geo, min_importance: float) -> list[dict]:
    """Geocode visible place names. Returns [{"query", "region", "confirmed", "kind": "admin"|"place"}].

    1. Every text line is geocoded on its own (OSM names in all languages, so Chinese works directly).
    2. Admin areas (country / province / state with importance ≥ 0.5, e.g. "陕西" → Shaanxi) become evidence and are
       used to disambiguate the other names: "<name> <admin text>" is queried as well, and hits inside the admin
       area win. Without such context a hit needs importance ≥ min_importance (bare "华山" matches several small
       peaks across China; "华山 陕西" lands at Mount Hua).
    3. Names whose hits agree across queries (within 30 km) or fall inside a named admin area are "confirmed".
    """
    queries = place_queries(texts)
    hits = {q: [h for h in geo.search(q, limit=5) if h.get("category", "place") in _PLACE_CATEGORIES] for q in queries}
    admins: list[tuple[str, Region]] = []
    for q, hs in hits.items():
        h = next((h for h in hs if h["addresstype"] in _ADMIN_TYPES - {"city", "municipality"}
                  and h["importance"] >= 0.5), None)
        if h:
            admins.append((q, _region(h, _level(h))))
    out = [{"query": q, "region": r, "confirmed": True, "kind": "admin"} for q, r in admins]
    admin_qs = {q for q, _ in admins}
    for q in queries:
        if q in admin_qs:
            continue
        pool = [(h, q) for h in hits[q]]
        for aq, _ in admins[:2]:
            combo = f"{q} {aq}"
            pool += [(h, combo) for h in geo.search(combo, limit=5) if h.get("category", "place") in _PLACE_CATEGORIES]
        best, best_key = None, None
        # agreement must come from a *different* name: "St Clare" vs "ST CLARE" (or "St Clare Hotel") are the
        # same text twice, not independent evidence
        others = [q2 for q2 in hits if q2 != q and not _same_name(q, q2)]
        for h, src in pool:
            in_admin = any(_inside(h, r) for _, r in admins)
            agree = sum(1 for q2 in others
                        for h2 in hits[q2] if haversine(h["lat"], h["lon"], h2["lat"], h2["lon"]) <= 30)
            if not (in_admin or agree or h["importance"] >= min_importance):
                continue
            key = (in_admin, agree > 0, h["importance"])
            if best is None or key > best_key:
                best, best_key = h, key
        if best is not None:
            out.append({"query": q, "region": _region(best, _level(best)),
                        "confirmed": bool(best_key[0] or best_key[1]), "kind": "place"})
    return out


def _tokens(q: str) -> set[str]:
    t = re.findall(r"[^\W\d_]+", q.lower())
    cjk = set(_HAS_CJK.findall(q))
    return {w for w in t if not _HAS_CJK.search(w)} | cjk


def _same_name(a: str, b: str) -> bool:
    """True if two text lines share a word (or, for CJK, half their characters): likely the same name."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    common = ta & tb
    if _HAS_CJK.search(a) and _HAS_CJK.search(b):
        return len(common) >= min(len(ta), len(tb)) / 2
    return bool(common)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from geoeg.eval.metrics import haversine_km
    return haversine_km(lat1, lon1, lat2, lon2)


def driving_side_region(side: str, sleuth) -> Region | None:
    if side not in ("left", "right"):
        return None
    left = [normalize_country(c) for c in sleuth.driving_side_countries("left")]
    return Region(name=f"driving-{side}", level="countries", countries=sorted(set(left)), negate=(side == "right"))
