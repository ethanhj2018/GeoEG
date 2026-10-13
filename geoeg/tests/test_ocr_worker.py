"""Persistent PaddleOCR worker: warm second request, same output as the one-shot worker."""
import os
import time
from pathlib import Path

import pytest

from geoeg.config import load_config
from geoeg.eval.datasets import holo_examples


def _sample():
    s = {x["id"]: x for x in holo_examples()} if Path(os.path.expanduser("~/geoeg_refs/HoloGeo")).exists() else {}
    return s.get("36fd4851ae54d3ff")


def test_persistent_worker_matches_one_shot():
    cfg = load_config()
    if not Path(cfg["ocr"]["conda"]).exists() or _sample() is None:
        pytest.skip("needs the geoeg-ocrtest env and sample images")
    from geoeg.extract import ocr
    img = _sample()["image"]
    first = ocr._SERVER.request([img], cfg)                 # may include model start-up
    t0 = time.time()
    second = ocr._SERVER.request([img], cfg)
    warm = time.time() - t0
    once = ocr.run_ocr_once([img], cfg)
    texts = lambda r: [l["text"] for l in r["results"][0]["lines"]]
    assert texts(first) == texts(second) == texts(once)
    assert warm < 5.0, f"warm request took {warm:.1f}s"
    ocr._SERVER.close()
