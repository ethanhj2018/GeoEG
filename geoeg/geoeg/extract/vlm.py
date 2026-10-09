"""OpenAI-compatible VLM client: local vLLM (MODE=local) or a cloud API (MODE=hybrid).

Two separate calls (v2):
- `predict_location`: the direct-prediction baseline. Same prompt as the web page's "baseline" engine
  (web/server.py) so GeoEG starts from exactly the answer the baseline would give.
- `extract`: describe what is visible ONLY. No per-clue location conclusions, so a wrong guess by the VLM
  cannot be copied into every clue. Clue regions come from deterministic tools (OCR, geocoder, lookup tables).
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request

from PIL import Image

from geoeg.config import resolve_vlm

LANGUAGES = {"en": "English", "zh": "Simplified Chinese"}

# --- 1. direct location prediction (keep in sync with web/server.py PROMPT / SCHEMA) ----------------------
BASELINE_PROMPT = """You are an expert at image geolocation (like GeoGuessr). Look carefully at this photo: text and signs,
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

_CATS = ["text", "object", "infra", "environment", "lighting"]
BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "country": {"type": "string"}, "region": {"type": "string"}, "city": {"type": "string"},
        "place": {"type": "string"}, "latitude": {"type": "number"}, "longitude": {"type": "number"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "clues": {"type": "array", "maxItems": 6, "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": _CATS},
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


def build_baseline_prompt(lang: str) -> str:
    _check(lang)
    # en is byte-identical to the web baseline; zh only swaps the output-language sentence
    return BASELINE_PROMPT if lang == "en" else BASELINE_PROMPT.replace(
        "Write all text values in English.", "Write all text values in Simplified Chinese, except country, region "
        "and city, which stay in English.")


# --- 2. description-only clue extraction ------------------------------------------------------------------
CLUE_PROMPT = """You are helping a geolocation system by describing what is VISIBLE in this photo. Do NOT say or guess
where the photo was taken; another component does that. Just report observations.

List 4 to 10 distinct visual observations that a geolocation expert would care about: all readable text and signs
(transcribe exactly), the writing system/script, architecture, road markings, vehicles and license plates,
infrastructure (poles, bollards, rails), vegetation, terrain, sky/sun/lighting.

For each observation:
- category: object | text | infra | lighting | environment
- description: what is visible, in {language}. No place names unless they are literally written in the image.
- query: MUST be English: a short noun phrase naming the visible thing, for an object detector
  (e.g. "street sign", "red tile roof", "church tower").
- text: the exact visible text for text observations, otherwise "".
Also report driving_side: left | right | unknown, ONLY if moving traffic or road markings make it visible.
Reply with ONLY one JSON object."""

CLUE_SCHEMA = {
    "type": "object",
    "properties": {
        "clues": {"type": "array", "maxItems": 10, "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["object", "text", "infra", "lighting", "environment"]},
                "description": {"type": "string"}, "query": {"type": "string"}, "text": {"type": "string"},
            },
            "required": ["category", "description", "query", "text"]}},
        "driving_side": {"type": "string", "enum": ["left", "right", "unknown"]},
    },
    "required": ["clues", "driving_side"],
}


def build_prompt(lang: str) -> str:
    _check(lang)
    return CLUE_PROMPT.replace("{language}", LANGUAGES[lang])


def _check(lang: str) -> None:
    if lang not in LANGUAGES:
        raise ValueError(f"output_language must be one of {sorted(LANGUAGES)}, got {lang!r}")


class VLMClient:
    def __init__(self, cfg: dict):
        self.conn = resolve_vlm(cfg)   # raises ConfigError for hybrid without key
        self.max_side = cfg["vlm"]["max_side"]
        self.max_tokens = cfg["vlm"]["max_tokens"]
        self.timeout = cfg["vlm"]["timeout_s"]
        lang = cfg.get("output_language", "en")
        self.prompt = build_prompt(lang)
        self.baseline_prompt = build_baseline_prompt(lang)

    @property
    def model(self) -> str:
        return self.conn["model"]

    def _b64(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        img.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.conn["base_url"] + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.conn['api_key']}"})
        return json.load(urllib.request.urlopen(req, timeout=self.timeout))

    def chat_json(self, image: Image.Image, prompt: str, schema: dict | None, max_tokens: int | None = None
                  ) -> tuple[dict, str]:
        body = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._b64(image)}"}},
                {"type": "text", "text": prompt}]}],
        }
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "geoeg", "schema": schema}}
        try:
            resp = self._post(body)
        except urllib.error.HTTPError as e:
            if e.code != 400 or not schema:
                raise
            body.pop("response_format")          # server without guided decoding
            resp = self._post(body)
        text = resp["choices"][0]["message"]["content"] or ""
        return parse_json(text), text

    def extract(self, image: Image.Image) -> tuple[dict, str]:
        """Description-only observations (no locations)."""
        return self.chat_json(image, self.prompt, CLUE_SCHEMA)

    def predict_location(self, image: Image.Image) -> tuple[dict, str]:
        """Direct coordinate prediction, same request as the web baseline (max_tokens 1024)."""
        return self.chat_json(image, self.baseline_prompt, BASELINE_SCHEMA, max_tokens=1024)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"VLM 输出里没有 JSON：{text[:300]}")
    return json.loads(m.group(0))
