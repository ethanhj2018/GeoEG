"""GeoEG v2: one image → VLM direct prediction (anchor) + description-only observations → isolation (detect / segment
/ OCR) → deterministic evidence (script, calling code, plate, geocoded place names, driving side) → retrieval pool
(GeoCLIP gallery, G3) → evidence graph over clues and candidates → decide (verify / override / refine) → Top-5.

Every run writes runs/<time>_<image>/ (log.jsonl, result.json, graph.json, crops/, masks/).
User-facing text (clue descriptions, summary, explanations) follows cfg["output_language"] (en | zh).
result.json keeps the v1 keys (top5, explanations, clues, edges, summary, timings_s, gpu ...) and adds
"baseline" and "decision" (incl. the feature dict that eval/run_eval.py re-scores offline).
"""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from geoeg.conclude import candidates as cands_mod
from geoeg.conclude.decide import STRENGTH, contains, decide, region_of
from geoeg.config import load_config
from geoeg.extract.vlm import VLMClient
from geoeg.gpu import GPUManager
from geoeg.graph import evidence_graph as eg
from geoeg.runlog import RunLog
from geoeg.schema import Candidate, Clue, Edge
from geoeg.tools import textclues as tc
from geoeg.tools.cache import Cache

log = logging.getLogger("geoeg.pipeline")

LOCAL_RADIUS_KM = 200.0     # local refinement sets are precomputed up to this radius (decide() may use less)
LOCAL_POINTS = 40           # GeoCLIP gallery points kept per centre
LOCAL_CENTRES = 10          # centres: baseline + best retrieval candidates

TEXT = {
    "en": {"ocr": "OCR text: {t}", "drive": "Vehicles drive on the {side}",
           "side": {"left": "left", "right": "right"},
           "drive_concl": {"left": "Left-hand traffic country ({n} in table)",
                           "right": "Not a left-hand traffic country"},
           "script": "Writing system: {s}", "script_concl": "Script used mainly in: {c}",
           "calling": "Calling code {code} in “{t}”", "calling_concl": "Calling code of: {c}",
           "plate": "Chinese plate prefix {k} in “{t}”", "plate_concl": "{r}",
           "place": "Place name in text: “{t}”", "place_concl": "Geocoded (OSM): {n}",
           "baseline_name": "VLM direct prediction",
           "head": "{place} ({lat:.4f}, {lon:.4f}), score {score:.2f}. ", "support": "Supporting clues: {s}. ",
           "sep": "; ", "contra": "Contradicted by: {s}. ", "sims": "Image–location similarity: {s}. ",
           "src": "Sources: {s}.", "lsep": ", ", "decision": "Decision: {n}. "},
    "zh": {"ocr": "OCR 文字：{t}", "drive": "车辆靠{side}行驶",
           "side": {"left": "左", "right": "右"},
           "drive_concl": {"left": "靠左行驶的国家/地区（表中 {n} 个）", "right": "不是靠左行驶的国家/地区"},
           "script": "文字系统：{s}", "script_concl": "主要使用该文字的国家：{c}",
           "calling": "“{t}”中的国际区号 {code}", "calling_concl": "该区号属于：{c}",
           "plate": "“{t}”中的中国车牌前缀 {k}", "plate_concl": "{r}",
           "place": "文字中的地名：“{t}”", "place_concl": "OSM 地理编码：{n}",
           "baseline_name": "VLM 直接预测",
           "head": "{place}（{lat:.4f}, {lon:.4f}），得分 {score:.2f}。", "support": "支持线索：{s}。",
           "sep": "；", "contra": "与之矛盾：{s}。", "sims": "图像-坐标相似度：{s}。", "src": "来源：{s}。", "lsep": "、",
           "decision": "决策：{n}。"},
}


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _box_contains(box: list[float], pt: tuple[float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _valid_latlon(d: dict) -> tuple[float, float] | None:
    try:
        lat, lon = float(d["latitude"]), float(d["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _haversine_np(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat), np.radians(lats)
    dp, dl = p2 - p1, np.radians(lons - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def run(image_path: str, cfg: dict | None = None, mode: str | None = None, models: dict | None = None) -> dict:
    """`models`: optional dict to keep GPU models resident across calls (batch evaluation)."""
    cfg = cfg or load_config(mode=mode)
    if mode:
        cfg["MODE"] = mode
    vlm = VLMClient(cfg)                       # fails fast (ConfigError) for hybrid without key
    sw = cfg["switches"]
    lang = cfg.get("output_language", "en")
    tx = TEXT[lang]
    image = Image.open(image_path).convert("RGB")
    rl = RunLog(cfg["paths"]["runs"], image_path)
    cache = Cache(cfg["paths"]["cache"])
    gpu = GPUManager(cfg["gpu"]["budget_mib"]).start()
    t_start = time.perf_counter()
    rl.event("start", image=str(image_path), mode=cfg["MODE"], model=vlm.model, size=image.size, version=2)
    img_sha = _sha(image_path)

    try:
        # 1. in parallel: direct prediction (baseline, same prompt as the web page), description-only
        #    observations (both on the vLLM server) and OCR (separate worker process) -------------------------
        def timed(name, fn):
            t0 = time.perf_counter()
            try:
                return fn()
            finally:
                rl.timings[name] = round(time.perf_counter() - t0, 2)

        def tolerant(call, empty: dict):
            # A malformed / truncated VLM reply must not sink the image: continue without that part
            # (no baseline → retrieval fallback; no observations → OCR-only evidence). Not cached.
            try:
                return dict(zip(("json", "raw"), call()))
            except ValueError as e:
                rl.event("vlm_parse_error", error=repr(e)[:500])
                return {"json": empty, "raw": f"<unparseable: {repr(e)[:300]}>", "_failed": True}

        def do_baseline():
            req = {"model": vlm.model, "image": img_sha, "max_side": cfg["vlm"]["max_side"],
                   "prompt": hashlib.sha256(vlm.baseline_prompt.encode()).hexdigest()}
            out = cache.get("vlm_baseline", req)
            if out is None:
                out = tolerant(lambda: vlm.predict_location(image), {})
                if not out.get("_failed"):
                    cache.set("vlm_baseline", req, out)
            return out

        def do_clues():
            req = {"model": vlm.model, "image": img_sha, "max_side": cfg["vlm"]["max_side"],
                   "prompt": hashlib.sha256(vlm.prompt.encode()).hexdigest()}
            out = cache.get("vlm_clues", req)
            if out is None:
                out = tolerant(lambda: vlm.extract(image), {"clues": [], "driving_side": "unknown"})
                if not out.get("_failed"):
                    cache.set("vlm_clues", req, out)
            return out

        def do_ocr():
            from geoeg.extract.ocr import run_ocr
            return cache.cached("ocr", {"image": img_sha, "rec": cfg["ocr"]["rec_model"]},
                                lambda: run_ocr([str(Path(image_path).resolve())], cfg))

        t_par = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_base = ex.submit(timed, "1a_vlm_baseline", do_baseline)
            f_clue = ex.submit(timed, "1b_vlm_clues", do_clues)
            f_ocr = ex.submit(timed, "2c_ocr", do_ocr) if sw["ocr"] else None
            bout, vout = f_base.result(), f_clue.result()
            ocr = f_ocr.result() if f_ocr else None
        rl.timings["1_parallel_wall"] = round(time.perf_counter() - t_par, 2)
        bj = bout["json"]
        ll = _valid_latlon(bj)
        baseline = None if ll is None else {
            "lat": ll[0], "lon": ll[1], "country": bj.get("country", ""), "region": bj.get("region", ""),
            "city": bj.get("city", ""), "place": bj.get("place", ""), "confidence": bj.get("confidence", "")}
        rl.event("vlm_baseline", raw=bout["raw"])
        clues = [Clue(id=f"c{i}", category=c.get("category", "object"), description=c.get("description", ""),
                      weight=2, confidence=0.5, query=c.get("query", ""), text=c.get("text", ""), sources=["vlm"])
                 for i, c in enumerate(vout["json"].get("clues", []), 1)]
        rl.event("vlm_clues", raw=vout["raw"])

        # 2. isolation: detect → segment → OCR --------------------------------------------------------------
        if sw["detect"]:
            from geoeg.extract.detect import Detector
            m = cfg["models"]
            with rl.timed("2a_detect"), gpu.use("grounding_dino", lambda: Detector(
                    m["gdino"], m["gdino_box_threshold"], m["gdino_text_threshold"]), keep=models) as det:
                for c in clues:
                    if c.category not in ("text", "object", "infra"):
                        continue                                  # sky / light / terrain: whole-image clues
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
            with rl.timed("2b_segment"), gpu.use("sam2", lambda: Segmenter(cfg["models"]["sam2"]), keep=models) as seg:
                for c, s in zip(boxed, seg.segment(image, [c.bbox for c in boxed])):
                    c.mask_area = s["mask_area"]
                    c.sources.append(f"sam2:{s['score']}")
                    c.crop = save_crop(image, c.bbox, s["mask"], str(rl.dir / "crops" / f"{c.id}.jpg"))
                    # full-size single-channel mask: 255 inside, 0 outside (the web page overlays it)
                    Image.fromarray(s["mask"].astype(np.uint8) * 255).save(rl.dir / "masks" / f"{c.id}.png")
        ocr_lines: list[dict] = []
        if ocr is not None:
            ocr_lines = [l for l in ocr["results"][0]["lines"] if l["score"] >= cfg["ocr"]["min_score"]
                         and len(l["text"].strip()) >= 2]
            rl.event("ocr", lines=ocr_lines)
            n = 0
            for l in ocr_lines:
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

        # 3. deterministic evidence ---------------------------------------------------------------------------
        with rl.timed("3_evidence"):
            from geoeg.tools.geosleuth import GeoSleuth
            sleuth = GeoSleuth(cfg, cache)
            texts = [l["text"] for l in ocr_lines] + [c.text for c in clues if c.text and c.category == "text"]
            geo = None
            if sw["geocode"]:
                from geoeg.tools.geocode import Geocoder
                geo = Geocoder(cfg, cache)
            side = vout["json"].get("driving_side", "unknown")
            evidence, ev_clues = collect_evidence(texts, side if sw["driving_side"] else "unknown", sleuth, geo, cfg, tx,
                                                  on_error=lambda e: rl.event("geocode_error", error=repr(e)))
            clues += ev_clues
            rl.event("evidence", evidence=evidence)

        # 4. retrieval pool: baseline + GeoCLIP gallery + geocoded places; G3 / GeoCLIP similarity --------------
        cc = cfg["candidates"]
        pool: list[dict] = []
        if baseline:
            pool.append({"id": "baseline", **{k: baseline[k] for k in ("lat", "lon", "country")},
                         "city": baseline["city"], "sources": ["vlm"]})
        pool += place_candidates(evidence)         # geocoded place names are always candidates
        geoclip_top = []
        local: dict[str, list[dict]] = {}
        with rl.timed("4a_geoclip"):
            if sw["geoclip"]:
                from geoeg.tools.geoclip_prior import GeoCLIPPrior
                with gpu.use("geoclip", GeoCLIPPrior, keep=models) as gc:
                    gal, sims, probs = gc.gallery_scores(image)
                    order = np.argsort(-probs)[:cc["geoclip_top_k"]]
                    geoclip_top = [{"lat": float(gal[i, 0]), "lon": float(gal[i, 1]), "prob": float(probs[i])} for i in order]
                    clusters = cands_mod.merge([cands_mod.from_geoclip(geoclip_top)], cc["merge_km"])
                    for i, c in enumerate(clusters, 1):
                        pool.append({"id": f"g{i}", "lat": c.lat, "lon": c.lon, "country": "", "city": "",
                                     "sources": ["geoclip"], "geoclip_prob": c.prior})
                    pool_sims = gc.similarity(image, [(c["lat"], c["lon"]) for c in pool])
                    for c, s in zip(pool, pool_sims):
                        c["geoclip_sim"] = round(float(s), 4)
                    # local refinement sets around the baseline and the best retrieval candidates
                    centres = ([c for c in pool if c["id"] == "baseline"]
                               + sorted((c for c in pool if c["id"] != "baseline"),
                                        key=lambda c: -c["geoclip_sim"])[:LOCAL_CENTRES - 1])
                    for c in centres:
                        d = _haversine_np(c["lat"], c["lon"], gal[:, 0], gal[:, 1])
                        idx = np.where((d <= LOCAL_RADIUS_KM) & (d > 0.5))[0]
                        idx = idx[np.argsort(-sims[idx])[:LOCAL_POINTS]]
                        local[c["id"]] = [{"lat": float(gal[i, 0]), "lon": float(gal[i, 1]),
                                           "geoclip_sim": round(float(sims[i]), 4), "d_km": round(float(d[i]), 2)}
                                          for i in idx]
        if sw["g3"] and pool:
            from geoeg.tools.g3_scorer import G3Scorer
            m = cfg["models"]
            with rl.timed("4b_g3"), gpu.use("g3", lambda: G3Scorer(m["g3_repo"], m["g3_ckpt"]), keep=models) as g3:
                flat = [(c["lat"], c["lon"]) for c in pool] + [(x["lat"], x["lon"]) for v in local.values() for x in v]
                s = g3.similarity(image, flat)
                for c, v in zip(pool, s[:len(pool)]):
                    c["g3_sim"] = round(float(v), 4)
                i = len(pool)
                for v in local.values():
                    for x in v:
                        x["g3_sim"] = round(float(s[i]), 4)
                        i += 1

        # 5. decide: verify / override / refine ------------------------------------------------------------------
        with rl.timed("5_decide"):
            features = {"baseline": baseline, "evidence": evidence, "pool": pool, "local": local}
            base_c = next((c for c in pool if c["id"] == "baseline"), None)
            if base_c and geo is not None and any(
                    contains(region_of(ev), base_c["lat"], base_c["lon"], base_c["country"]) is False for ev in evidence):
                # the baseline is contradicted somewhere: name the strongest alternatives so country-level
                # evidence can be checked against them (only then; keeps Nominatim traffic low)
                alts = sorted((c for c in pool if c["id"] != "baseline" and not c["country"]),
                              key=lambda c: -(c.get("g3_sim", 0) + c.get("geoclip_sim", 0)))[:8]
                for c in alts:
                    try:
                        rev = geo.reverse(c["lat"], c["lon"], zoom=3)
                    except Exception as e:
                        rl.event("reverse_error", error=repr(e))
                        continue
                    c["country"] = rev.get("country", "")
            params = cfg["decision"]["params"]
            dec = decide(features, params, k=cc["output_top_k"])
            rl.event("decision", action=dec["action"], note=dec["note"])

        # 6. names for the output + evidence graph ---------------------------------------------------------------
        with rl.timed("6_output"):
            top = [_to_candidate(c) for c in dec["top"]]
            if sw.get("name_results", True) and geo is not None:
                for c, raw in zip(top, dec["top"]):
                    if raw["id"] == "baseline" and not raw.get("refined_from") and baseline:
                        continue                                 # the VLM already named it
                    try:
                        rev = geo.reverse(c.lat, c.lon)
                    except Exception as e:
                        rl.event("reverse_error", error=repr(e))
                        continue
                    if rev:
                        c.country, c.country_code = rev["country"], rev["country_code"]
                        c.city, c.display_name = rev["city"], rev["display_name"]
            graph, edges = eg.build_graph(clues, with_contradict=sw["contradict_edges"])
            for raw in dec["top"]:
                cid = f"cand:{raw['id']}"
                graph.add_node(cid, kind="candidate", lat=raw["lat"], lon=raw["lon"], score=raw.get("score"))
                for ev_id, typ in [(i, "support") for i in raw.get("supporting", [])] + \
                                  [(i, "contradict") for i in raw.get("contradicting", [])]:
                    if typ == "contradict" and not sw["contradict_edges"]:
                        continue
                    e = Edge(ev_id, cid, typ, f"evidence {typ}s candidate {raw['id']}")
                    edges.append(e)
                    graph.add_edge(ev_id, cid, type=typ, reason=e.reason)

        explanations = [explain(c, clues, lang) for c in top]
        if explanations:
            explanations[0] = tx["decision"].format(n=dec["note"]) + explanations[0]
        result = {
            "image": str(image_path), "mode": cfg["MODE"], "model": vlm.model, "run_dir": str(rl.dir),
            "version": 2,
            "top5": [c.to_dict() for c in top],
            "explanations": explanations,
            "language": lang,
            "clues": [c.to_dict() for c in clues],
            "edges": [e.to_dict() for e in edges],
            "summary": bj.get("reasoning", ""),
            "driving_side": side,
            "n_candidates": len(pool),
            "baseline": baseline,
            "geoclip_top1": geoclip_top[0] if geoclip_top else None,
            "decision": {"action": dec["action"], "note": dec["note"], "params": params, "features": features},
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


def collect_evidence(texts: list[str], side: str, sleuth, geo, cfg: dict, tx: dict, on_error=None
                     ) -> tuple[list[dict], list[Clue]]:
    """Deterministic evidence from visible text and the observed driving side → (evidence dicts, clue nodes).

    Sources: writing system, calling codes, Chinese plates (offline tables), place names geocoded by OSM
    (`geo` may be None to skip), driving side. Never uses the VLM's own location guess.
    """
    evidence: list[dict] = []
    clues: list[Clue] = []

    def add(kind: str, region, desc: str, concl: str, src: list[str], strength=None, **extra) -> None:
        cid = f"e{len(evidence) + 1}"
        strength = STRENGTH[kind] if strength is None else strength
        clues.append(Clue(id=cid, category="text" if kind != "driving_side" else "infra", description=desc,
                          weight=min(5, max(1, round(strength))), conclusion=concl,
                          confidence=round(min(1.0, strength / 4), 2), region=region, text="", sources=src))
        evidence.append({"id": cid, "kind": kind, "strength": round(strength, 3), "region": region.__dict__, **extra})

    for script, region, t in tc.script_regions(texts):
        add("script", region, tx["script"].format(s=script.title()),
            tx["script_concl"].format(c=", ".join(region.countries)), ["unicode-script"])
    for code, region, t in tc.calling_code_regions(texts, sleuth):
        add("calling_code", region, tx["calling"].format(code=code, t=t),
            tx["calling_concl"].format(c=", ".join(region.countries)), ["geo-sleuth:clues.py"])
    for key, region, t in tc.plate_regions(texts, sleuth):
        add("plate", region, tx["plate"].format(k=key, t=t), tx["plate_concl"].format(r=region.name),
            ["geo-sleuth:clues.py"])
    if geo is not None:
        try:
            places = tc.place_evidence(texts, geo, cfg["decision"]["place_min_importance"])
        except Exception as e:                             # network errors must not kill the run
            if on_error:
                on_error(e)
            places = []
        for pe in places:
            r = pe["region"]
            if pe["kind"] == "admin":
                strength = STRENGTH["admin"]
            elif pe["confirmed"]:
                strength = STRENGTH["place"]               # readable, corroborated place name: near-decisive
            else:
                strength = STRENGTH["place_unconfirmed"] * min(1.0, (r.importance or 0.0) / 0.5)
            add("place", r, tx["place"].format(t=pe["query"]), tx["place_concl"].format(n=r.name),
                ["ocr/vlm-text", "nominatim"], strength=strength, confirmed=pe["confirmed"])
    region = tc.driving_side_region(side, sleuth)
    if region:
        add("driving_side", region, tx["drive"].format(side=tx["side"][side]),
            tx["drive_concl"][side].format(n=len(region.countries)), ["vlm-observation", "geo-sleuth:clues.py"])
    return evidence, clues


def place_candidates(evidence: list[dict]) -> list[dict]:
    out = []
    for ev in evidence:
        r = region_of(ev)
        if ev["kind"] == "place" and r.is_point and r.level in ("poi", "city"):
            out.append({"id": f"p:{ev['id']}", "lat": r.lat, "lon": r.lon, "country": r.country,
                        "city": r.name.split(",")[0], "sources": [f"clue:{ev['id']}"]})
    return out


def _to_candidate(raw: dict) -> Candidate:
    srcs = list(raw.get("sources", []))
    if raw.get("refined_from"):
        srcs.append("refined:g3+geoclip")
    return Candidate(
        lat=raw["lat"], lon=raw["lon"], country=raw.get("country", ""), city=raw.get("city", ""),
        display_name=", ".join(x for x in (raw.get("city", ""), raw.get("country", "")) if x)
        if raw["id"] == "baseline" else "",
        score=float(raw.get("score", 0.0)), sources=srcs, prior=0.0,
        supporting_clues=list(raw.get("supporting", [])), contradicting_clues=list(raw.get("contradicting", [])),
        evidence={k: raw[k] for k in ("g3_sim", "geoclip_sim", "geoclip_prob", "retrieval", "support", "contra",
                                      "refined_from") if k in raw} | {"candidate_id": raw["id"]})


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
    srcs = [tx["baseline_name"] if s == "vlm" else s for s in c.sources]
    parts.append(tx["src"].format(s=tx["lsep"].join(srcs)))
    return "".join(parts).strip()
