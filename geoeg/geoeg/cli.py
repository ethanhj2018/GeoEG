"""python -m geoeg.cli <image> [--mode local|hybrid] [--lang en|zh] [--config config.yaml] [--json]"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from geoeg.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geoeg", description="GeoEG: geolocate a single image (evidence graph)")
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["local", "hybrid"], help="override MODE in config.yaml")
    ap.add_argument("--lang", choices=["en", "zh"], help="override output_language in config.yaml")
    ap.add_argument("--config", help="config file path (default: config.yaml in the project root)")
    ap.add_argument("--json", action="store_true", help="print the full result JSON to stdout")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s: %(message)s")

    try:
        cfg = load_config(args.config, mode=args.mode, lang=args.lang)
        from geoeg.pipeline import run
        res = run(args.image, cfg)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return 0
    print(f"Mode {res['mode']}  model {res['model']}  {res['n_candidates']} candidates  run dir {res['run_dir']}")
    print(f"{'#':<3}{'Score':>7}  {'Lat':>9} {'Lon':>10}  Place")
    for i, c in enumerate(res["top5"], 1):
        place = ", ".join(x for x in (c["city"], c["country"]) if x) or "-"
        print(f"{i:<3}{c['score']:>7.2f}  {c['lat']:>9.4f} {c['lon']:>10.4f}  {place}")
    print("\nExplanation (#1): " + res["explanations"][0] if res["explanations"] else "")
    t = res["timings_s"]
    print("\nTimings (s): " + "  ".join(f"{k}={v}" for k, v in t.items()))
    print(f"Peak card VRAM {res['gpu']['card_peak_mib']} MiB (budget {res['gpu']['budget_mib']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
