"""output_language switch (prompt + explanations) and SAM 2 mask files."""
import json
import urllib.request

import numpy as np
import pytest
from PIL import Image

from geoeg.config import ConfigError, load_config
from geoeg.eval.datasets import holo_examples
from geoeg.extract.vlm import VLMClient, build_prompt
from geoeg.pipeline import explain
from geoeg.schema import Candidate, Clue, Region


def test_prompt_language_switch():
    assert "in English;" in build_prompt("en") and "Simplified Chinese" not in build_prompt("en")
    assert "in Simplified Chinese;" in build_prompt("zh")
    for lang in ("en", "zh"):    # detector phrases and geocoder names stay English either way
        assert "query and region_query in English only" in build_prompt(lang)
    with pytest.raises(ValueError):
        build_prompt("fr")


def test_config_lang_override_and_validation():
    assert load_config(lang="zh")["output_language"] == "zh"
    assert VLMClient(load_config(mode="local", lang="zh")).prompt == build_prompt("zh")
    assert VLMClient(load_config(mode="local", lang="en")).prompt == build_prompt("en")
    with pytest.raises(ConfigError):
        load_config(lang="de")


def test_explanation_language():
    cl = Clue("c1", "text", "Dutch street sign", 4, conclusion="Netherlands", confidence=0.8,
              region=Region(name="Netherlands", level="country"))
    c = Candidate(52.37, 4.9, country="Netherlands", city="Amsterdam", score=3.2, sources=["vlm"],
                  supporting_clues=["c1"], evidence={"g3_sim": 0.31})
    en, zh = explain(c, [cl], "en"), explain(c, [cl], "zh")
    assert "Supporting clues" in en and "score 3.20" in en
    assert all(w not in en for w in ("支持线索", "得分", "来源"))
    assert "支持线索" in zh and "得分" in zh


def _vllm_up(cfg) -> bool:
    try:
        urllib.request.urlopen(cfg["vlm"]["local"]["base_url"] + "/models", timeout=3)
        return True
    except Exception:
        return False


def test_masks_written_and_english_output():
    torch = pytest.importorskip("torch")
    cfg = load_config(mode="local", lang="en")
    if not torch.cuda.is_available() or not _vllm_up(cfg):
        pytest.skip("需要 CUDA 和在跑的本地 vLLM")
    samples = {s["id"]: s for s in holo_examples()}
    if "3947712380" not in samples:
        pytest.skip("缺少 HoloGeo 样例图")
    from geoeg.pipeline import run
    res = run(samples["3947712380"]["image"], cfg)
    assert res["language"] == "en"
    vlm_clues = [c for c in res["clues"] if c["id"].startswith("c")]
    # English output: VLM descriptions carry no CJK characters
    assert vlm_clues and not any(any("一" <= ch <= "鿿" for ch in c["description"]) for c in vlm_clues)
    masked = [c for c in res["clues"] if c["mask_area"] is not None]
    assert masked, "expected at least one segmented clue"
    w, h = Image.open(res["image"]).size
    from pathlib import Path
    for c in masked:
        m = np.array(Image.open(Path(res["run_dir"]) / "masks" / f"{c['id']}.png"))
        assert m.shape == (h, w) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 255}
    unmasked = [c for c in res["clues"] if c["mask_area"] is None]
    assert all(not (Path(res["run_dir"]) / "masks" / f"{c['id']}.png").exists() for c in unmasked)
    print(json.dumps([c["description"] for c in vlm_clues[:3]], ensure_ascii=False))
