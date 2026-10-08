"""One image → clues → isolation (detect / segment / OCR) → tool checks → evidence graph → candidates & scoring
→ Top-5 + evidence-graph JSON + explanation. Every run writes runs/<time>_<image>/ (log.jsonl, result.json, graph.json, crops/, masks/).
User-facing text (clue descriptions, summary, explanations) follows cfg["output_language"] (en | zh)."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from geoeg.conclude import candidates as cands_mod
from geoeg.conclude import scoring
from geoeg.config import load_config
from geoeg.extract.vlm import VLMClient
from geoeg.gpu import GPUManager
from geoeg.graph import evidence_graph as eg
from geoeg.runlog import RunLog
from geoeg.schema import Candidate, Clue, Region
from geoeg.tools.cache import Cache

log = logging.getLogger("geoeg.pipeline")

TEXT = {
    "en": {"ocr": "OCR text: {t}", "drive": "Vehicles drive on the {side}",
           "drive_concl": "One of {n} countries/territories that drive on the {side}",
           "side": {"left": "left", "right": "right"},
           "head": "{place} ({lat:.4f}, {lon:.4f}), score {score:.2f}. ", "support": "Supporting clues: {s}. ",
           "sep": "; ", "contra": "Contradicted by: {s}. ", "sims": "Image–location similarity: {s}. ",
           "src": "Sources: {s}.", "lsep": ", "},
    "zh": {"ocr": "OCR 文字：{t}", "drive": "车辆靠{side}行驶", "drive_concl": "{n} 个{side}行国家/地区之一",
           "side": {"left": "左", "right": "右"},
           "head": "{place}（{lat:.4f}, {lon:.4f}），得分 {score:.2f}。", "support": "支持线索：{s}。",
           "sep": "；", "contra": "与之矛盾：{s}。", "sims": "图像-坐标相似度：{s}。", "src": "来源：{s}。", "lsep": "、"},
}


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clues_from_vlm(out: dict) -> list[Clue]:
    clues = []
    for i, c in enumerate(out.get("clues", []), 1):
        clues.append(Clue(
            id=f"c{i}", category=c.get("category", "object"), description=c.get("description", ""),
            weight=int(min(5, max(1, int(c.get("weight", 3))))), conclusion=c.get("conclusion", ""),
            confidence=float(min(1.0, max(0.0, float(c.get("confidence", 0.5))))), query=c.get("query", ""),
            text=c.get("text", ""), region_query=c.get("region_query", "") if c.get("region_level") != "none" else "",
            region=Region(level=c.get("region_level", "none")) if c.get("region_level", "none") != "none" else None,
            sources=["vlm"]))
    return clues


def _box_contains(box: list[float], pt: tuple[float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def run(image_path: str, cfg: dict | None = None, mode: str | None = None) -> dict:
    cfg = cfg or load_config(mode=mode)
    if mode:
        cfg["MODE"] = mode
    vlm = VLMClient(cfg)                       # fails fast (ConfigError) for hybrid without key
    sw = cfg["switches"]
    tx = TEXT[cfg.get("output_language", "en")]
    image = Image.open(image_path).convert("RGB")
    rl = RunLog(cfg["paths"]["runs"], image_path)
    cache = Cache(cfg["paths"]["cache"])
    gpu = GPUManager(cfg["gpu"]["budget_mib"]).start()
    t_start = time.perf_counter()
    rl.event("start", image=str(image_path), mode=cfg["MODE"], model=vlm.model, size=image.size)

    try:
        # 1. clue extraction ----------------------------------------------------------------------
        with rl.timed("1_vlm_extract"):
            req = {"model": vlm.model, "image": _sha(image_path), "prompt": hashlib.sha256(vlm.prompt.encode()).hexdigest(),
                   "max_side": cfg["vlm"]["max_side"]}
            vout = cache.cached("vlm", req, lambda: dict(zip(("json", "raw"), vlm.extract(image))))
            clues = _clues_from_vlm(vout["json"])
            rl.event("vlm_output", raw=vout["raw"])

        # 2. isolation: detect → segment → OCR ----------------------------------------------------
        if sw["detect"]:
            from geoeg.extract.detect import Detector
            m = cfg["models"]
            with rl.timed("2a_detect"), gpu.use("grounding_dino", lambda: Detector(
                    m["gdino"], m["gdino_box_threshold"], m["gdino_text_threshold"])) as det:
                for c in clues:
                    if not c.query.isascii():                     # Grounding DINO only understands English
                        rl.event("detect_skip", clue=c.id, query=c.query)
                        continue
                    hit = det.ground(image, c.query)
                    if hit:
                        c.bbox = hit["bbox"]
                        c.sources.append(f"gdino:{hit['score']}")
                rl.event("detect", boxes={c.id: c.bbox for c in clues})
        boxed = [c for c in clues if c.bbox]
        if sw["segment"] and boxed:
            from geoeg.extract.segment import Segmenter, save_crop
            with rl.timed("2b_segment"), gpu.use("sam2", lambda: Segmenter(cfg["models"]["sam2"])) as seg:
                for c, s in zip(boxed, seg.segment(image, [c.bbox for c in boxed])):
                    c.mask_area = s["mask_area"]
                    c.sources.append(f"sam2:{s['score']}")
                    c.crop = save_crop(image, c.bbox, s["mask"], str(rl.dir / "crops" / f"{c.id}.jpg"))
                    # full-size single-channel mask: 255 inside, 0 outside (the web page overlays it)
                    Image.fromarray(s["mask"].astype(np.uint8) * 255).save(rl.dir / "masks" / f"{c.id}.png")
        if sw["ocr"]:
            from geoeg.extract.ocr import run_ocr
            with rl.timed("2c_ocr"):
                gpu._current = "paddleocr(subprocess)"
                try:
                    ocr = cache.cached("ocr", {"image": _sha(image_path), "rec": cfg["ocr"]["rec_model"]},
                                       lambda: run_ocr([str(Path(image_path).resolve())], cfg))
                finally:
                    gpu._current = None
                lines = [l for l in ocr["results"][0]["lines"] if l["score"] >= cfg["ocr"]["min_score"]
                         and len(l["text"].strip()) >= 2]
                rl.event("ocr", lines=lines)
                n = len(clues)
                for l in lines:
                    center = ((l["box"][0] + l["box"][2]) / 2, (l["box"][1] + l["box"][3]) / 2)
                    owner = next((c for c in clues if c.category == "text" and c.bbox and _box_contains(c.bbox, center)), None)
                    if owner:
                        owner.text = (owner.text + " | " if owner.text else "") + l["text"]
                        owner.sources.append("ocr")
                    else:
                        n += 1
                        clues.append(Clue(id=f"o{n}", category="text", description=tx["ocr"].format(t=l["text"]),
                                          weight=2, confidence=round(0.5 * l["score"], 2), text=l["text"],
                                          bbox=l["box"], sources=[f"ocr:{l['score']}"]))

        # 3. tool checks: geocode clue regions, driving-side table -----------------------------------
        with rl.timed("3_tools"):
            if sw["geocode"]:
                from geoeg.tools.geocode import Geocoder
                geo = Geocoder(cfg, cache)
                for c in clues:
                    if not c.region_query:
                        continue
                    try:
                        hits = geo.search(c.region_query, level=c.region.level if c.region else None)
                    except Exception as e:                         # network errors must not kill the run
                        rl.event("geocode_error", clue=c.id, query=c.region_query, error=repr(e))
                        continue
                    if hits:
                        h = hits[0]
                        c.region = Region(name=h["name"], level=c.region.level if c.region else "city",
                                          lat=h["lat"], lon=h["lon"], radius_km=h["radius_km"], bbox=h["bbox"],
                                          country=h["country"], country_code=h["country_code"])
                        c.sources.append("nominatim")
                rl.event("geocode", regions={c.id: c.region for c in clues if c.region}, calls=geo.calls)
            side = vout["json"].get("driving_side", "unknown")
            if sw["driving_side"] and side in ("left", "right"):
                from geoeg.tools.geosleuth import GeoSleuth
                countries = GeoSleuth(cfg, cache).driving_side_countries(side)
                sd = tx["side"][side]
                clues.append(Clue(id="ds", category="infra", description=tx["drive"].format(side=sd),
                                  weight=3, confidence=0.6, conclusion=tx["drive_concl"].format(n=len(countries), side=sd),
                                  region=Region(name=f"driving-{side}", level="countries", countries=countries),
                                  sources=["vlm", "geo-sleuth:clues.py"]))

        # 4. evidence graph -----------------------------------------------------------------------
        with rl.timed("4_graph"):
            graph, edges = eg.build_graph(clues, with_contradict=sw["contradict_edges"])
            load = eg.contradiction_load(graph)
            consistency = {k: 1.0 / (1.0 + v / 5.0) for k, v in load.items()}
            rl.event("graph", edges=[e.to_dict() for e in edges])

        # 5. candidates + similarity evidence + scoring ---------------------------------------------
        cc = cfg["candidates"]
        with rl.timed("5a_candidates_geoclip"):
            groups = [cands_mod.from_vlm(vout["json"].get("guesses", []), cc["vlm_top_k"]), cands_mod.from_clues(clues)]
            if sw["geoclip"]:
                from geoeg.tools.geoclip_prior import GeoCLIPPrior
                with gpu.use("geoclip", GeoCLIPPrior) as gc:
                    groups.append(cands_mod.from_geoclip(gc.top_k(image, cc["geoclip_top_k"])))
                    pool = cands_mod.merge(groups, cc["merge_km"])
                    for c, s in zip(pool, gc.similarity(image, [(c.lat, c.lon) for c in pool])):
                        c.evidence["geoclip_sim"] = round(float(s), 4)
            else:
                pool = cands_mod.merge(groups, cc["merge_km"])
        if sw["g3"] and pool:
            from geoeg.tools.g3_scorer import G3Scorer
            m = cfg["models"]
            with rl.timed("5b_g3"), gpu.use("g3", lambda: G3Scorer(m["g3_repo"], m["g3_ckpt"])) as g3:
                for c, s in zip(pool, g3.similarity(image, [(c.lat, c.lon) for c in pool])):
                    c.evidence["g3_sim"] = round(float(s), 4)
        with rl.timed("5c_score"):
            ranked = scoring.score(pool, clues, consistency, cfg["scoring"])
            top_n = ranked[:cc["rescore_top_n"]]
            if sw["geocode"]:
                for c in top_n:                                   # name the finalists, then re-check country-level clues
                    try:
                        rev = geo.reverse(c.lat, c.lon)
                    except Exception as e:
                        rl.event("reverse_error", error=repr(e))
                        continue
                    if rev:
                        c.country, c.country_code = rev["country"], rev["country_code"]
                        c.city, c.display_name = rev["city"], rev["display_name"]
                top_n = scoring.score(top_n, clues, consistency, cfg["scoring"])
            top = top_n[:cc["output_top_k"]]

        result = {
            "image": str(image_path), "mode": cfg["MODE"], "model": vlm.model, "run_dir": str(rl.dir),
            "top5": [c.to_dict() for c in top],
            "explanations": [explain(c, clues, cfg.get("output_language", "en")) for c in top],
            "language": cfg.get("output_language", "en"),
            "clues": [c.to_dict() for c in clues],
            "edges": [e.to_dict() for e in edges],
            "summary": vout["json"].get("summary", ""),
            "driving_side": side,
            "n_candidates": len(pool),
        }
    finally:
        gpu.stop()
    result["timings_s"] = {**rl.timings, "total": round(time.perf_counter() - t_start, 2)}
    result["gpu"] = gpu.report()
    result["cache"] = {"hits": cache.hits, "misses": cache.misses}
    eg.save(graph, rl.dir / "graph.json")
    rl.write_json("result.json", result)
    rl.event("done", timings=result["timings_s"], gpu=result["gpu"])
    return result


def explain(c: Candidate, clues: list[Clue], lang: str = "en") -> str:
    tx = TEXT[lang]
    by_id = {cl.id: cl for cl in clues}
    place = c.display_name or ", ".join(x for x in (c.city, c.country) if x) or f"{c.lat:.3f}, {c.lon:.3f}"
    parts = [tx["head"].format(place=place, lat=c.lat, lon=c.lon, score=c.score)]
    if c.supporting_clues:
        s = tx["sep"].join(f"[{i}] {by_id[i].description} → {by_id[i].conclusion or by_id[i].region.name}"
                           for i in c.supporting_clues if i in by_id)
        parts.append(tx["support"].format(s=s))
    if c.contradicting_clues:
        parts.append(tx["contra"].format(s=tx["lsep"].join(f"[{i}]" for i in c.contradicting_clues)))
    ev = c.evidence
    sims = [f"{k} {ev[k]:.3f}" for k in ("g3_sim", "geoclip_sim") if k in ev]
    if sims:
        parts.append(tx["sims"].format(s=tx["lsep"].join(sims)))
    parts.append(tx["src"].format(s=tx["lsep"].join(c.sources)))
    return "".join(parts).strip()
