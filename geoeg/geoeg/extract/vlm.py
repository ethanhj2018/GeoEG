"""OpenAI-compatible VLM client: local vLLM (MODE=local) or a cloud API (MODE=hybrid).

One call extracts structured clues + up to 5 direct location guesses as JSON.
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

PROMPT = """You are an expert image geolocation analyst. Inspect this photo and list the concrete visual clues that
help determine where it was taken: text/signs/language, architecture, road markings, vehicles and plates, infrastructure
(poles, bollards, rails), vegetation, terrain, climate, sun/lighting. Then give your best location guesses.

Rules:
- 4 to 10 clues, each about a DIFFERENT visible thing. Do not repeat a clue.
- category: one of object | text | infra | lighting | environment.
- query: MUST be English: a short noun phrase naming the visible thing, for an object detector
  (e.g. "street sign", "red tile roof", "church tower").
- text: the exact visible text for text clues, otherwise "".
- weight: 1-5, how discriminative this kind of clue is (5 = readable place name, 1 = generic).
- confidence: 0-1, how sure you are about the conclusion.
- region_query: MUST be English: the place this clue alone points to, as a name a geocoder understands
  (e.g. "Belgium", "Bavaria, Germany", "Kyoto, Japan"). Use "" if the clue alone points nowhere specific.
  Never write vague directions like "southern Germany"; name a country, state/province or city.
- region_level: country | admin1 | city | poi | none (matching region_query).
- description, conclusion and summary in {language}; query and region_query in English only.
- driving_side: left | right | unknown (from vehicles/road markings, only if visible).
- guesses: up to 5 locations, best first, with decimal latitude/longitude.
- summary: one or two sentences on where the photo was most likely taken and why.
Reply with ONLY one JSON object."""

LANGUAGES = {"en": "English", "zh": "Simplified Chinese"}


def build_prompt(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"output_language must be one of {sorted(LANGUAGES)}, got {lang!r}")
    return PROMPT.replace("{language}", LANGUAGES[lang])

SCHEMA = {
    "type": "object",
    "properties": {
        "clues": {"type": "array", "maxItems": 10, "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["object", "text", "infra", "lighting", "environment"]},
                "description": {"type": "string"}, "query": {"type": "string"}, "text": {"type": "string"},
                "weight": {"type": "integer", "minimum": 1, "maximum": 5},
                "conclusion": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "region_query": {"type": "string"},
                "region_level": {"type": "string", "enum": ["country", "admin1", "city", "poi", "none"]},
            },
            "required": ["category", "description", "query", "text", "weight", "conclusion", "confidence",
                         "region_query", "region_level"]}},
        "driving_side": {"type": "string", "enum": ["left", "right", "unknown"]},
        "guesses": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {"country": {"type": "string"}, "region": {"type": "string"}, "city": {"type": "string"},
                           "latitude": {"type": "number"}, "longitude": {"type": "number"},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["country", "region", "city", "latitude", "longitude", "confidence"]}},
        "summary": {"type": "string"},
    },
    "required": ["clues", "driving_side", "guesses", "summary"],
}


class VLMClient:
    def __init__(self, cfg: dict):
        self.conn = resolve_vlm(cfg)   # raises ConfigError for hybrid without key
        self.max_side = cfg["vlm"]["max_side"]
        self.max_tokens = cfg["vlm"]["max_tokens"]
        self.timeout = cfg["vlm"]["timeout_s"]
        self.prompt = build_prompt(cfg.get("output_language", "en"))

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

    def chat_json(self, image: Image.Image, prompt: str, schema: dict | None) -> tuple[dict, str]:
        body = {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
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
        return self.chat_json(image, self.prompt, SCHEMA)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"VLM 输出里没有 JSON：{text[:300]}")
    return json.loads(m.group(0))
