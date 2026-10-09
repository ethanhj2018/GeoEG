"""End-to-end smoke test on one HoloGeo sample image (MODE=local). Skipped when vLLM / CUDA / the sample are absent."""
import json
import os
import urllib.request
from pathlib import Path

import pytest

from geoeg.config import ConfigError, load_config
from geoeg.eval.datasets import holo_examples
from geoeg.eval.metrics import haversine_km


def _vllm_up(cfg) -> bool:
    try:
        urllib.request.urlopen(cfg["vlm"]["local"]["base_url"] + "/models", timeout=3)
        return True
    except Exception:
        return False


def test_hybrid_without_key_is_a_clear_config_error(monkeypatch):
    for k in ("GEOEG_API_BASE", "GEOEG_API_MODEL", "GEOEG_API_KEY"):
        monkeypatch.setenv(k, "")
    cfg = load_config(mode="hybrid")
    from geoeg.pipeline import run
    with pytest.raises(ConfigError, match="GEOEG_API_KEY"):
        run("does-not-matter.jpg", cfg)


def test_end_to_end_local():
    torch = pytest.importorskip("torch")
    cfg = load_config(mode="local")
    if not torch.cuda.is_available() or not _vllm_up(cfg):
        pytest.skip("需要 CUDA 和在跑的本地 vLLM")
    samples = {s["id"]: s for s in holo_examples()} if Path(os.path.expanduser("~/geoeg_refs/HoloGeo")).exists() else {}
    if not samples:
        pytest.skip("缺少 HoloGeo 样例图")
    s = samples.get("36fd4851ae54d3ff") or next(iter(samples.values()))
    from geoeg.pipeline import run
    res = run(s["image"], cfg)
    assert 1 <= len(res["top5"]) <= 5
    assert all(-90 <= c["lat"] <= 90 and -180 <= c["lon"] <= 180 for c in res["top5"])
    assert res["clues"] and res["explanations"]
    run_dir = Path(res["run_dir"])
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    assert {"nodes", "edges"} <= set(graph)
    assert (run_dir / "log.jsonl").stat().st_size > 0
    # The whole-card figure includes vLLM, the desktop and any other GPU process (web server, a batch evaluation),
    # so it is not asserted here; the run log warns on budget overruns. GeoEG's own models must stay small.
    g = res["gpu"]
    assert max(g["per_model_own_peak_reserved_mib"].values()) <= 1600
    d = haversine_km(res["top5"][0]["lat"], res["top5"][0]["lon"], s["lat"], s["lon"])
    print(f"top-1 error {d:.1f} km; timings {res['timings_s']}")
