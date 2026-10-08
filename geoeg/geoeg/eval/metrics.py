"""Great-circle distance and the standard Acc@{1,25,200,750,2500} km geolocation metrics."""
from __future__ import annotations

import math
from typing import Iterable

EARTH_RADIUS_KM = 6371.0088
THRESHOLDS_KM = (1, 25, 200, 750, 2500)   # street / city / region / country / continent


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def accuracy_at(distances_km: Iterable[float], thresholds=THRESHOLDS_KM) -> dict[str, float]:
    """Fraction of predictions within each threshold. A None distance (no prediction) counts as a miss."""
    d = list(distances_km)
    n = len(d)
    if n == 0:
        return {f"acc@{t}km": 0.0 for t in thresholds}
    return {f"acc@{t}km": sum(1 for x in d if x is not None and x <= t) / n for t in thresholds}


def summarize(distances_km: Iterable[float]) -> dict:
    d = [x for x in distances_km]
    valid = sorted(x for x in d if x is not None)
    med = valid[len(valid) // 2] if valid else None
    return {"n": len(d), "median_km": med, **accuracy_at(d)}
