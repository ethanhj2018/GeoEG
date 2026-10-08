"""GeoEG local web page backend. Two engines for an uploaded image:
  geoeg    (default) the full GeoEG pipeline (../geoeg: VLM clues -> Grounding DINO -> SAM 2 -> OCR -> evidence graph
           -> candidates), imported in-process; each run lands in geoeg/runs/<name>/ and is served by /api/run/<name>
  baseline one direct call to the local Qwen2.5-VL-7B (vLLM, start_vllm.sh) for place + coordinates + clue boxes
Run (conda env geoeg, inside WSL): uvicorn server:app --host 0.0.0.0 --port 7860   (or use start_web.sh)

Clue format the page draws (both sources produce it):
  {"id", "category", "description", "conclusion", "confidence" 0-1, "text",
   "bbox": [x0, y0, x1, y1] in original-image pixels or null,
   "mask": URL of a full-size grayscale PNG (white = clue) or null, "crop": URL or null, "sources": [...]}
"""
import base64, hashlib, io, json, math, os, re, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "qwen2.5-vl-7b-awq")
MAX_SIDE = 1280  # vLLM caps pixels anyway; shrinking here keeps the request small
# must match --mm-processor-kwargs in scripts/start_vllm.sh: Qwen2.5-VL boxes are in the resized input's pixels
MIN_PIXELS, MAX_PIXELS, PATCH = 200704, 1003520, 28
GEOEG_ROOT = Path(os.environ.get("GEOEG_ROOT", Path(__file__).resolve().parent.parent / "geoeg"))
RUNS_DIR = GEOEG_ROOT / "runs"
UPLOADS = Path(__file__).resolve().parent / "uploads"  # GeoEG runs keep a path to their image, so uploads persist
GEOEG_LOCK = threading.Lock()                         # one pipeline run at a time: it loads models onto the GPU
CATEGORIES = ["text", "object", "infra", "environment", "lighting"]  # same set as geoeg/geoeg/schema.py

PROMPT = """You are an expert at image geolocation (like GeoGuessr). Look carefully at this photo: text and signs,
language, architecture, vegetation, road markings, vehicles, terrain, climate. Decide where it was most likely taken.
Give your best precise coordinates for the spot itself, not just the city center.
For each clue you used, give the bounding box of the image region that shows it as bbox_2d [x1, y1, x2, y2]
in pixel coordinates of this image (use null for whole-image clues such as climate or lighting).
Write all text values in English. Reply with ONLY a JSON object, no other text, with these keys:
{"country": "...", "region": "state/province", "city": "...", "place": "specific landmark or area, or empty",
 "latitude": <number>, "longitude": <number>, "confidence": "high|medium|low",
 "clues": [{"category": "text|object|infra|environment|lighting", "description": "what is visible",
            "conclusion": "what it suggests about the location", "text": "visible text, or empty",
            "confidence": <0-1>, "bbox_2d": [x1, y1, x2, y2] or null}],
 "reasoning": "one or two sentences"}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "country": {"type": "string"}, "region": {"type": "string"}, "city": {"type": "string"},
        "place": {"type": "string"}, "latitude": {"type": "number"}, "longitude": {"type": "number"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "clues": {"type": "array", "maxItems": 6, "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "description": {"type": "string"}, "conclusion": {"type": "string"}, "text": {"type": "string"},
                "confidence": {"type": "number"},
                "bbox_2d": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                      {"type": "null"}]},
            },
            "required": ["category", "description", "conclusion", "text", "confidence", "bbox_2d"],
        }},
        "reasoning": {"type": "string"},
    },
    "required": ["country", "region", "city", "place", "latitude", "longitude", "confidence", "clues", "reasoning"],
}

app = FastAPI(title="GeoEG demo")
HERE = Path(__file__).parent


def load_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unrecognized image format")


def to_jpeg_b64(img: Image.Image) -> tuple[str, tuple[int, int]]:
    img = img.copy()
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), img.size


def model_input_size(w: int, h: int) -> tuple[int, int]:
    """Qwen2.5-VL smart_resize: the size vLLM actually feeds the model, which its boxes refer to."""
    hb, wb = max(PATCH, round(h / PATCH) * PATCH), max(PATCH, round(w / PATCH) * PATCH)
    if hb * wb > MAX_PIXELS:
        beta = math.sqrt(h * w / MAX_PIXELS)
        hb, wb = math.floor(h / beta / PATCH) * PATCH, math.floor(w / beta / PATCH) * PATCH
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb, wb = math.ceil(h * beta / PATCH) * PATCH, math.ceil(w * beta / PATCH) * PATCH
    return wb, hb


def call_vllm(b64: str, structured: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    if structured:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "geo", "schema": SCHEMA}}
    req = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def parse_answer(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in answer")
    return json.loads(m.group(0))


def scale_box(box, sx: float, sy: float, w: int, h: int):
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((x0 * sx, x1 * sx))
    y0, y1 = sorted((y0 * sy, y1 * sy))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/health")
def health():
    try:
        urllib.request.urlopen(VLLM_URL.replace("/v1/chat/completions", "/health"), timeout=3)
        return {"model_ready": True, "model": MODEL}
    except Exception:
        return {"model_ready": False, "model": MODEL}


@app.post("/api/predict")
def predict(file: UploadFile = File(...), engine: str = Query("geoeg", pattern="^(geoeg|baseline)$")):
    raw = file.file.read()
    img = load_image(raw)
    if engine == "geoeg":
        return predict_geoeg(raw, img)
    b64, sent = to_jpeg_b64(img)
    t0 = time.time()
    try:
        try:
            resp = call_vllm(b64, structured=True)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            resp = call_vllm(b64, structured=False)  # server without guided decoding
    except urllib.error.URLError as e:
        raise HTTPException(503, f"Model server not ready ({VLLM_URL}): {e}")
    text = resp["choices"][0]["message"]["content"]
    try:
        result = parse_answer(text)
    except Exception:
        raise HTTPException(502, f"Could not parse model output: {text[:500]}")
    for k in ("latitude", "longitude"):
        try:
            result[k] = float(result.get(k))
        except (TypeError, ValueError):
            result[k] = None

    W, H = img.size
    mw, mh = model_input_size(*sent)
    clues = []
    for i, c in enumerate(result.get("clues") or [], 1):
        if isinstance(c, str):
            c = {"description": c}
        clues.append({
            "id": f"c{i}", "category": c.get("category") or "object", "description": c.get("description", ""),
            "conclusion": c.get("conclusion", ""), "text": c.get("text", ""),
            "confidence": c.get("confidence"), "bbox": scale_box(c.get("bbox_2d"), W / mw, H / mh, W, H),
            "mask": None, "crop": None, "sources": ["vlm"],
        })
    result["clues"] = clues
    result["image_size"] = [W, H]
    result["source"] = "baseline"
    result["elapsed_ms"] = round((time.time() - t0) * 1000)
    result["model"] = MODEL
    result["raw"] = text
    return result


# ---- GeoEG pipeline runs (geoeg/runs/<name>/result.json) --------------------------------------------------

def predict_geoeg(raw: bytes, img: Image.Image) -> dict:
    UPLOADS.mkdir(exist_ok=True)
    path = UPLOADS / f"{hashlib.sha256(raw).hexdigest()[:16]}.jpg"
    if not path.exists():
        img.save(path, quality=95)
    if str(GEOEG_ROOT) not in sys.path:
        sys.path.insert(0, str(GEOEG_ROOT))
    from geoeg.config import ConfigError, load_config
    from geoeg.pipeline import run
    if not GEOEG_LOCK.acquire(timeout=600):
        raise HTTPException(503, "GeoEG is busy with another image, try again shortly")
    try:
        res = run(str(path), load_config(mode="local"))
    except ConfigError as e:
        raise HTTPException(500, f"GeoEG config error: {e}")
    except urllib.error.URLError as e:
        raise HTTPException(503, f"Model server not ready ({VLLM_URL}): {e}")
    finally:
        GEOEG_LOCK.release()
    return get_run(Path(res["run_dir"]).name)


def run_dir(name: str) -> Path:
    d = (RUNS_DIR / name).resolve()
    if d.parent != RUNS_DIR.resolve() or not (d / "result.json").is_file():
        raise HTTPException(404, f"No GeoEG run named {name}")
    return d


@app.get("/api/runs")
def list_runs():
    if not RUNS_DIR.is_dir():
        return []
    return sorted((p.name for p in RUNS_DIR.iterdir() if (p / "result.json").is_file()), reverse=True)


@app.get("/api/run/{name}")
def get_run(name: str):
    d = run_dir(name)
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    with Image.open(r["image"]) as im:
        W, H = im.size
    clues = []
    for c in r.get("clues", []):
        mask = c.get("mask") or str(d / "masks" / f"{c['id']}.png")
        clues.append({
            "id": c["id"], "category": c.get("category", "object"), "description": c.get("description", ""),
            "conclusion": c.get("conclusion", "") or ((c.get("region") or {}).get("name", "")),
            "text": c.get("text", ""), "confidence": c.get("confidence"), "bbox": c.get("bbox"),
            "mask": f"/api/run/{name}/file/masks/{c['id']}.png" if Path(mask).is_file() else None,
            "crop": f"/api/run/{name}/file/crops/{c['id']}.jpg" if c.get("crop") and Path(c["crop"]).is_file() else None,
            "sources": c.get("sources", []),
        })
    top5 = r.get("top5") or []
    top = top5[0] if top5 else {}
    return {
        "source": "geoeg", "run": name, "model": r.get("model", ""), "image": f"/api/run/{name}/image",
        "image_size": [W, H], "clues": clues,
        "place": top.get("display_name", ""), "city": top.get("city", ""), "country": top.get("country", ""),
        "latitude": top.get("lat"), "longitude": top.get("lon"), "confidence": None,
        "reasoning": r.get("summary", ""), "elapsed_ms": round(r.get("timings_s", {}).get("total", 0) * 1000),
        "top5": [{"lat": c["lat"], "lon": c["lon"], "score": c.get("score"),
                  "name": c.get("display_name") or ", ".join(x for x in (c.get("city"), c.get("country")) if x)}
                 for c in top5],
        "explanations": r.get("explanations", []),
        "raw": json.dumps({k: r.get(k) for k in ("top5", "edges", "timings_s", "gpu")}, ensure_ascii=False, indent=1),
    }


@app.get("/api/run/{name}/image")
def run_image(name: str):
    r = json.loads((run_dir(name) / "result.json").read_text(encoding="utf-8"))
    return FileResponse(r["image"])


@app.get("/api/run/{name}/file/{kind}/{fname}")
def run_file(name: str, kind: str, fname: str):
    if kind not in ("crops", "masks") or "/" in fname or ".." in fname:
        raise HTTPException(404)
    p = run_dir(name) / kind / fname
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)
