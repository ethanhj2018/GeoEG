"""Thin wrappers around the vendored geo-sleuth scripts (MIT, see geosleuth/LICENSE and THIRD_PARTY_NOTICES.md).

The scripts are kept byte-identical and run as CLI subprocesses (their own sys.path/data layout), JSON out.
Offline tables (clues.py, sun.py) need no network; osm.py calls Overpass, so it is rate-limited and cached here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from geoeg.tools.cache import Cache, RateLimiter

SCRIPTS = Path(__file__).with_name("geosleuth") / "scripts"


def _run(script: str, args: list[str], cwd: str, timeout: int = 120) -> dict:
    Path(cwd).mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True,
                       cwd=cwd, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{script} {' '.join(args)} failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


class GeoSleuth:
    def __init__(self, cfg: dict, cache: Cache):
        self.cwd = cfg["paths"]["geosleuth_cache"]
        self.cache = cache
        self.overpass_limiter = RateLimiter(cfg["overpass"]["min_interval_s"])

    def driving_side_countries(self, side: str) -> list[str]:
        """Lower-case English country names that drive on `side` (left/right). Offline table."""
        req = ["lookup", "driving-side", side, "--json"]
        out = self.cache.cached("geosleuth:clues", req, lambda: _run("clues.py", req, self.cwd))
        return sorted({m["country"].lower() for m in out.get("matches", []) if m.get("country")})

    def lookup(self, kind: str, value: str) -> dict:
        """clues.py lookup <kind> <value> (plate / area-code / calling-code / admin ...). Offline table."""
        req = ["lookup", kind, value, "--json"]
        return self.cache.cached("geosleuth:clues", req, lambda: _run("clues.py", req, self.cwd))

    def sun_pos(self, lat: float, lon: float, time_iso: str, tz: str = "UTC") -> dict:
        req = ["pos", "--at", f"{lat},{lon}", "--time", time_iso, "--tz", tz]
        return _run("sun.py", req, self.cwd)

    def osm_find(self, bbox: tuple[float, float, float, float], osm_filter: str) -> dict:
        """Overpass: features matching `osm_filter` (e.g. '["railway"="tram"]') inside bbox (s,w,n,e)."""
        req = ["find", "--bbox", ",".join(f"{v:.5f}" for v in bbox), osm_filter]

        def fetch():
            self.overpass_limiter.wait()
            return _run("osm.py", req, self.cwd, timeout=400)
        return self.cache.cached("geosleuth:osm", req, fetch)
