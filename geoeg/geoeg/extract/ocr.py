"""PaddleOCR via a subprocess in the separate geoeg-ocrtest env (`conda run`), JSON in / JSON out."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

WORKER = Path(__file__).with_name("ocr_worker.py")


def run_ocr(images: list[str], cfg: dict) -> dict:
    o = cfg["ocr"]
    with tempfile.TemporaryDirectory(prefix="geoeg_ocr_") as tmp:
        req, out = Path(tmp) / "req.json", Path(tmp) / "out.json"
        req.write_text(json.dumps({"images": images, "det_model": o["det_model"], "rec_model": o["rec_model"]}),
                       encoding="utf-8")
        cmd = [o["conda"], "run", "-n", o["env"], "--no-capture-output",
               "python", str(WORKER), "--in", str(req), "--out", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=o["timeout_s"])
        if p.returncode or not out.exists():
            raise RuntimeError(f"OCR worker 失败（exit {p.returncode}）：{(p.stderr or p.stdout)[-800:]}")
        return json.loads(out.read_text(encoding="utf-8"))
