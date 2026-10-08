"""python -m geoeg.cli <image> [--mode local|hybrid] [--lang en|zh] [--config config.yaml] [--json]"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from geoeg.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geoeg", description="GeoEG：单张图片地理定位（证据图）")
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["local", "hybrid"], help="覆盖 config.yaml 的 MODE")
    ap.add_argument("--lang", choices=["en", "zh"], help="覆盖 config.yaml 的 output_language")
    ap.add_argument("--config", help="配置文件路径（默认项目根目录的 config.yaml）")
    ap.add_argument("--json", action="store_true", help="把完整结果 JSON 打到标准输出")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s: %(message)s")

    try:
        cfg = load_config(args.config, mode=args.mode, lang=args.lang)
        from geoeg.pipeline import run
        res = run(args.image, cfg)
    except ConfigError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"找不到文件：{e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return 0
    print(f"模式 {res['mode']}  模型 {res['model']}  候选 {res['n_candidates']} 个  运行目录 {res['run_dir']}")
    print(f"{'#':<3}{'得分':>7}  {'纬度':>9} {'经度':>10}  地点")
    for i, c in enumerate(res["top5"], 1):
        place = ", ".join(x for x in (c["city"], c["country"]) if x) or "-"
        print(f"{i:<3}{c['score']:>7.2f}  {c['lat']:>9.4f} {c['lon']:>10.4f}  {place}")
    print("\n解释（第 1 名）：" + res["explanations"][0] if res["explanations"] else "")
    t = res["timings_s"]
    print("\n耗时(s)：" + "  ".join(f"{k}={v}" for k, v in t.items()))
    print(f"整卡显存峰值 {res['gpu']['card_peak_mib']} MiB（预算 {res['gpu']['budget_mib']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
