"""PaddleOCR in the separate geoeg-ocrtest env (`conda run`), JSON in / JSON out.

Default: a long-lived worker process (ocr.persistent) keeps the model warm, so only the first image pays the
~7 s start-up; it exits by itself after ocr.idle_exit_s seconds without requests, which releases its ~0.8 GB of
VRAM. Falls back to a one-shot subprocess if the worker cannot be used.
"""
from __future__ import annotations

import atexit
import json
import logging
import select
import subprocess
import tempfile
import threading
import time
from pathlib import Path

WORKER = Path(__file__).with_name("ocr_worker.py")
MARK = "@@GEOEG_OCR@@"
log = logging.getLogger("geoeg.ocr")


class _Server:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def _start(self, cfg: dict) -> None:
        o = cfg["ocr"]
        self.proc = subprocess.Popen(
            [o["conda"], "run", "-n", o["env"], "--no-capture-output", "python", "-u", str(WORKER), "--serve",
             "--idle-exit", str(o.get("idle_exit_s", 600))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def request(self, images: list[str], cfg: dict) -> dict:
        o = cfg["ocr"]
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start(cfg)
            req = {"images": images, "det_model": o["det_model"], "rec_model": o["rec_model"]}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + o["timeout_s"]
            while True:
                left = deadline - time.time()
                if left <= 0:
                    self.close()
                    raise TimeoutError("OCR worker timed out")
                ready, _, _ = select.select([self.proc.stdout], [], [], left)
                if not ready:
                    continue
                line = self.proc.stdout.readline()
                if not line:
                    self.close()
                    raise RuntimeError("OCR worker exited")
                if line.startswith(MARK):
                    resp = json.loads(line[len(MARK):])
                    if "error" in resp:
                        raise RuntimeError(f"OCR worker error: {resp['error']}")
                    return resp

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


_SERVER = _Server()
atexit.register(_SERVER.close)


def run_ocr(images: list[str], cfg: dict) -> dict:
    if cfg["ocr"].get("persistent", True):
        try:
            return _SERVER.request(images, cfg)
        except Exception as e:                       # fall back to a one-shot worker
            log.warning("persistent OCR worker failed (%r); using a one-shot worker", e)
    return run_ocr_once(images, cfg)


def run_ocr_once(images: list[str], cfg: dict) -> dict:
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
