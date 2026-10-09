"""Benchmark on IM2GPS3K: (a) VLM direct prediction, (b) GeoCLIP alone, (c) GeoEG v1 (legacy_v1), (d) GeoEG v2.

    python -m geoeg.eval.run_eval split  [--seed 2026 --dev 100 --test 200]
    python -m geoeg.eval.run_eval run    --split dev|test --method v2|v1 [--limit N]
    python -m geoeg.eval.run_eval tune   # grid-search decide() params on dev only
    python -m geoeg.eval.run_eval report # metrics on test only → eval/results/

Data and per-image records live in the WSL filesystem (~/geoeg_data, override with GEOEG_DATA); only the
small CSV / Markdown summaries are written to geoeg/eval/results/.

Evaluation settings (stated in the report): Grounding DINO and SAM 2 are off (they only produce boxes/masks for
display; v2 evidence comes from OCR and text, so decisions are unaffected), v2 skips naming the Top-5 by reverse
geocoding, models are loaded per image (VRAM budget), and every external call goes through the SQLite cache.
(a) and (b) come from the same v2 runs: (a) is v2's step 1a (same prompt as the web baseline), (b) GeoCLIP's
gallery top-1. (d) re-applies decide() with the dev-tuned parameters to the stored per-image features, which is
exactly what pipeline.run() computes with those parameters in config.yaml.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

from geoeg.config import ROOT, load_config
from geoeg.conclude.decide import DEFAULT_PARAMS, decide
from geoeg.eval.metrics import THRESHOLDS_KM, haversine_km, summarize

DATA = Path(__import__("os").environ.get("GEOEG_DATA", "~/geoeg_data")).expanduser()
DS = DATA / "im2gps3k"
WORK = DATA / "eval"
RESULTS = ROOT / "geoeg" / "eval" / "results"
SPLIT_FILE = DS / "split.json"


# --- data -----------------------------------------------------------------------------------------------------
def load_meta() -> dict[str, dict]:
    path = DS / "im2gps3k_places365.csv"
    if not path.exists():
        sys.exit(f"缺少 {path}：先下载 IM2GPS3K（见 README）")
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img = DS / "images" / r["IMG_ID"]
            if img.exists():
                rows[r["IMG_ID"]] = {"id": r["IMG_ID"], "image": str(img), "lat": float(r["LAT"]), "lon": float(r["LON"])}
    return rows


def cmd_split(args) -> None:
    meta = load_meta()
    ids = sorted(meta)
    random.Random(args.seed).shuffle(ids)
    split = {"seed": args.seed, "source_n": len(ids), "dev": ids[:args.dev], "test": ids[args.dev:args.dev + args.test]}
    SPLIT_FILE.write_text(json.dumps(split, indent=1), encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"{len(ids)} images with ground truth; dev={len(split['dev'])} test={len(split['test'])} (seed {args.seed})")


def split_items(name: str, limit: int | None = None) -> list[dict]:
    meta = load_meta()
    ids = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))[name]
    return [meta[i] for i in ids[:limit] if i in meta]


def eval_cfg(runs_dir: Path) -> dict:
    cfg = load_config(mode="local", lang="en")
    cfg["switches"].update(detect=False, segment=False, name_results=False)
    cfg["paths"]["runs"] = str(runs_dir)
    return cfg


def prefetch_ocr(items: list[dict], cfg: dict) -> None:
    """One PaddleOCR worker for all images (avoids a ~7 s cold start per image); fills the shared cache."""
    from geoeg.extract.ocr import run_ocr
    from geoeg.pipeline import _sha
    from geoeg.tools.cache import Cache
    cache = Cache(cfg["paths"]["cache"])
    todo = [it for it in items if cache.get("ocr", {"image": _sha(it["image"]), "rec": cfg["ocr"]["rec_model"]}) is None]
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        cfg2 = copy.deepcopy(cfg)
        cfg2["ocr"]["timeout_s"] = 1800
        out = run_ocr([it["image"] for it in chunk], cfg2)
        for it, res in zip(chunk, out["results"]):
            cache.set("ocr", {"image": _sha(it["image"]), "rec": cfg["ocr"]["rec_model"]},
                      {"results": [res], "elapsed_s": None})
        print(f"OCR prefetch {i + len(chunk)}/{len(todo)}", flush=True)


# --- running --------------------------------------------------------------------------------------------------
def records_path(split: str, method: str) -> Path:
    return WORK / f"{split}_{method}.jsonl"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def cmd_run(args) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    items = split_items(args.split, args.limit)
    cfg = eval_cfg(WORK / "runs" / args.method)
    if cfg["switches"]["ocr"]:
        prefetch_ocr(items, cfg)
    out = records_path(args.split, args.method)
    if args.retry_failed and out.exists():
        keep = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("ok")]
        out.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    done = done_ids(out)
    if args.method == "v2":
        from geoeg.pipeline import run
    else:
        from geoeg.legacy_v1.pipeline import run
    t_all = time.time()
    n = 0
    for it in items:
        if it["id"] in done:
            continue
        t0 = time.time()
        try:
            res = run(it["image"], copy.deepcopy(cfg))
            rec = {"id": it["id"], "lat": it["lat"], "lon": it["lon"], "ok": True,
                   "pred": [res["top5"][0]["lat"], res["top5"][0]["lon"]] if res["top5"] else None,
                   "timings": res["timings_s"], "gpu_peak": res["gpu"]["card_peak_mib"]}
            if args.method == "v2":
                b, g = res.get("baseline"), res.get("geoclip_top1")
                rec.update(baseline=[b["lat"], b["lon"]] if b else None, geoclip=[g["lat"], g["lon"]] if g else None,
                           features=res["decision"]["features"], action=res["decision"]["action"])
        except Exception as e:                                  # record and continue; counted as a miss
            rec = {"id": it["id"], "lat": it["lat"], "lon": it["lon"], "ok": False, "error": repr(e)[:500]}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        el = time.time() - t_all
        print(f"[{args.method}/{args.split}] {len(done) + n}/{len(items)} {it['id']} {time.time() - t0:.1f}s "
              f"(avg {el / n:.1f}s, eta {(len(items) - len(done) - n) * el / n / 60:.0f} min)", flush=True)


# --- scoring --------------------------------------------------------------------------------------------------
def load_records(split: str, method: str) -> list[dict]:
    p = records_path(split, method)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def dist(rec: dict, pred) -> float | None:
    return None if not pred else haversine_km(pred[0], pred[1], rec["lat"], rec["lon"])


def v2_pred(rec: dict, params: dict):
    if not rec.get("ok"):
        return None
    d = decide(rec["features"], params, k=1)
    return [d["final"]["lat"], d["final"]["lon"]] if d["final"] else None


def objective(dists: list[float | None]) -> float:
    """Mean of Acc@{1,25,200,750,2500} — all five scales weigh the same."""
    s = summarize(dists)
    return sum(s[f"acc@{t}km"] for t in THRESHOLDS_KM) / len(THRESHOLDS_KM)


# Two-stage grid (anchor choice, then refinement around the chosen anchor) keeps the search small.
GRID_ANCHOR = {
    "place_decisive": [4.0, math.inf],
    "override_min_contra": [1.0, 2.0, 3.0, math.inf],
    "retrieval_margin": [0.25, 0.5, 1.0, math.inf],
    "retrieval_max_km": [250.0, 750.0, 2500.0, math.inf],
    "w_g3": [0.5, 1.0, 2.0],
    "w_gc": [0.5, 1.0],
    "w_prox": [0.0, 0.5],
}
GRID_REFINE = {
    "refine_radius_km": [0.0, 10.0, 50.0, 200.0],
    "refine_alpha": [0.0, 0.5, 1.0],
    "refine_dist_penalty": [0.0, 0.5],
    "refine_min_gain": [0.0, 0.1],
    "refine_skip_high": [False],
}


def cmd_tune(args) -> None:
    recs = [r for r in load_records("dev", "v2") if r.get("ok")]
    if not recs:
        sys.exit("先跑 dev：run --split dev --method v2")
    rows = []
    base_obj = objective([dist(r, r.get("baseline")) for r in recs])
    best = (base_obj, dict(DEFAULT_PARAMS, override_min_contra=math.inf, place_decisive=math.inf,
                           refine_radius_km=0.0))               # pure baseline is always a candidate
    # exhaustive joint grid (the anchor and refinement parameters interact, so a staged search gets stuck)
    grid = {**GRID_ANCHOR, **GRID_REFINE}
    keys = list(grid)
    ri = keys.index("refine_radius_km")
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = {**DEFAULT_PARAMS, **dict(zip(keys, combo))}
        if combo[ri] == 0 and (p["refine_alpha"], p["refine_dist_penalty"], p["refine_min_gain"]) != (0.0, 0.0, 0.0):
            continue                                            # refinement params are irrelevant without refinement
        ds = [dist(r, v2_pred(r, p)) for r in recs]
        obj = objective(ds)
        rows.append({**{k: p[k] for k in keys}, "objective": round(obj, 4), "median_km": summarize(ds)["median_km"]})
        if obj > best[0] + 1e-12 or (abs(obj - best[0]) < 1e-12 and _simpler(p, best[1])):
            best = (obj, p)
    print(f"joint grid: {len(rows)} combos, best objective {best[0]:.4f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "dev_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["objective"]))
    tuned = {k: best[1][k] for k in DEFAULT_PARAMS}
    s_best = summarize([dist(r, v2_pred(r, tuned)) for r in recs])
    s_base = summarize([dist(r, r.get("baseline")) for r in recs])
    print("dev baseline:", s_base)
    print("dev tuned:   ", s_best)
    (RESULTS / "tuned_params.json").write_text(json.dumps(
        {"params": tuned, "dev_objective": round(best[0], 4), "dev_baseline_objective": round(base_obj, 4),
         "n_dev": len(recs), "grid_size": len(rows)}, indent=1, default=str), encoding="utf-8")
    print(f"dev: baseline objective {base_obj:.4f}, best {best[0]:.4f} over {len(rows)} combos")
    print(json.dumps(tuned, default=str))


def _simpler(p: dict, q: dict) -> bool:
    """Tie-break towards fewer deviations from the baseline (no override / no refinement)."""
    def c(x):
        return ((x["override_min_contra"] != math.inf) + (x["retrieval_margin"] != math.inf)
                + (x["refine_radius_km"] > 0) - (x["place_decisive"] != math.inf))   # keep the place rule on ties
    return c(p) < c(q)


def cmd_report(args) -> None:
    tuned = json.loads((RESULTS / "tuned_params.json").read_text(encoding="utf-8"))["params"]
    tuned = {k: (math.inf if v in ("inf", "Infinity") else v) for k, v in tuned.items()}
    v2 = {r["id"]: r for r in load_records("test", "v2")}
    v1 = {r["id"]: r for r in load_records("test", "v1")}
    ids = sorted(set(v2) & set(v1)) if v1 else sorted(v2)
    methods = {
        "a_vlm_direct": lambda r: v2[r].get("baseline"),
        "b_geoclip": lambda r: v2[r].get("geoclip"),
        "c_geoeg_v1": (lambda r: v1[r].get("pred")) if v1 else None,
        "d_geoeg_v2_tuned": lambda r: v2_pred(v2[r], tuned),
        "d0_geoeg_v2_default": lambda r: v2_pred(v2[r], DEFAULT_PARAMS),
    }
    per_image, table = [], []
    for name, fn in methods.items():
        if fn is None:
            continue
        ds = [dist(v2[i], fn(i)) for i in ids]
        s = summarize(ds)
        table.append({"method": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
        for i, d in zip(ids, ds):
            per_image.append({"id": i, "method": name, "error_km": None if d is None else round(d, 2)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with (RESULTS / "test_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "method", "error_km"])
        w.writeheader()
        w.writerows(per_image)
    actions = {}
    for i in ids:
        if v2[i].get("ok"):
            a = decide(v2[i]["features"], tuned, k=1)["action"]
            actions[a] = actions.get(a, 0) + 1
    t2 = [v2[i]["timings"]["total"] for i in ids if v2[i].get("ok")]
    t1 = [v1[i]["timings"]["total"] for i in ids if v1 and v1[i].get("ok")]
    md = ["# IM2GPS3K 测试集结果", "",
          f"- 测试集 {len(ids)} 张（seed 见 split.json；开发集只用于调参，未参与本表）",
          f"- 失败（计为未命中）：v2 {sum(not v2[i].get('ok') for i in ids)}，"
          f"v1 {sum(not v1[i].get('ok') for i in ids) if v1 else '-'}",
          "- 评测设置：Grounding DINO / SAM 2 关闭（只影响显示）；v2 不对 Top-5 反向地理编码取地名；外部调用全部走缓存。", "",
          "| 方法 | n | 中位误差 km | Acc@1km | Acc@25km | Acc@200km | Acc@750km | Acc@2500km |", "|---|---|---|---|---|---|---|---|"]
    for r in table:
        med = "-" if r["median_km"] is None else f"{r['median_km']:.0f}"
        md.append(f"| {r['method']} | {r['n']} | {med} | " + " | ".join(f"{100 * r[f'acc@{t}km']:.1f}%" for t in THRESHOLDS_KM) + " |")
    md += ["", f"v2 决策分布（调参后）：{actions}", "",
           f"调参结果：`{json.dumps(tuned, default=str)}`", "",
           f"评测模式下单张耗时（中位数）：v2 {statistics.median(t2):.1f}s" + (f"，v1 {statistics.median(t1):.1f}s" if t1 else "")]
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(prog="run_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--dev", type=int, default=100)
    s.add_argument("--test", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test"], required=True)
    r.add_argument("--method", choices=["v2", "v1"], required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--retry-failed", action="store_true", help="re-run images whose record has ok=false")
    sub.add_parser("tune")
    sub.add_parser("report")
    args = ap.parse_args()
    {"split": cmd_split, "run": cmd_run, "tune": cmd_tune, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
