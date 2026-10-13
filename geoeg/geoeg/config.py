"""Load config.yaml + .env and resolve project-relative paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # .../geoeg (project root, holds config.yaml)


class ConfigError(RuntimeError):
    """Raised for missing keys / unusable settings; carries a user-facing message."""


def load_config(path: str | os.PathLike | None = None, mode: str | None = None, lang: str | None = None) -> dict:
    load_dotenv(ROOT / ".env", override=False)
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if mode:
        cfg["MODE"] = mode
    if lang:
        cfg["output_language"] = lang
    cfg.setdefault("output_language", "en")
    if cfg["output_language"] not in ("en", "zh"):
        raise ConfigError(f"output_language must be en or zh, got: {cfg['output_language']!r}")
    if cfg["MODE"] not in ("local", "hybrid"):
        raise ConfigError(f"MODE must be local or hybrid, got: {cfg['MODE']!r}")
    for k, v in cfg["paths"].items():
        p = Path(os.path.expanduser(v))
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    for k in ("g3_repo", "g3_ckpt"):
        cfg["models"][k] = os.path.expanduser(cfg["models"][k])
    cfg["ocr"]["conda"] = os.path.expanduser(cfg["ocr"]["conda"])
    return cfg


def resolve_vlm(cfg: dict) -> dict:
    """Return {base_url, model, api_key} for the current MODE, or raise ConfigError with a clear hint."""
    v = cfg["vlm"]
    if cfg["MODE"] == "local":
        return {"base_url": v["local"]["base_url"].rstrip("/"), "model": v["local"]["model"], "api_key": "EMPTY"}
    h = v["hybrid"]
    vals = {k: os.environ.get(h[f"{k}_env"], "").strip() for k in ("base_url", "model", "api_key")}
    missing = [h[f"{k}_env"] for k, val in vals.items() if not val]
    if missing:
        raise ConfigError(
            "hybrid mode needs cloud API settings, but .env is missing: " + ", ".join(missing)
            + f". Copy {ROOT / '.env.example'} to .env and fill it in, or use --mode local.")
    vals["base_url"] = vals["base_url"].rstrip("/")
    return vals
