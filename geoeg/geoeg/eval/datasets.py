"""Dataset loaders. IM2GPS3K / YFCC4K are not downloaded yet: the loaders define the interface and fail clearly.

Every loader yields dicts {"id", "image", "lat", "lon"} (image = local file path).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterator

DATA_ROOT = Path(os.environ.get("GEOEG_DATA", "~/geoeg_data")).expanduser()


def holo_examples(root: str = "~/geoeg_refs/HoloGeo/Data_Examples") -> Iterator[dict]:
    """The 5 BF30K sample images shipped with HoloGeo (used by the smoke test)."""
    root = Path(root).expanduser()
    for line in (root / "Examples_BF30K.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        img = root / "Fig_BF30K" / (r["image_id"].split("/")[-1] + ".jpg")
        if img.exists():
            yield {"id": img.stem, "image": str(img), "lat": r["latitude"], "lon": r["longitude"],
                   "label": r.get("original_location", "")}


def _csv_dataset(name: str, img_dir: str, meta: str, id_col: str, lat_col: str, lon_col: str,
                 split: str | None, limit: int | None) -> Iterator[dict]:
    root = DATA_ROOT / name
    meta_path, images = root / meta, root / img_dir
    if not meta_path.exists() or not images.exists():
        raise FileNotFoundError(
            f"{name} is not downloaded: need {meta_path} and {images}/. Set GEOEG_DATA to the data root; "
            f"the metadata CSV is from TIBHannover/GeoEstimation and the image archive is linked in the G3 repo README.")
    with meta_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split == "dev":           # fixed dev subset: first 500 rows in file order
        rows = rows[:500]
    for r in rows[:limit] if limit else rows:
        yield {"id": r[id_col], "image": str(images / r[id_col]), "lat": float(r[lat_col]), "lon": float(r[lon_col])}


def im2gps3k(split: str | None = None, limit: int | None = None) -> Iterator[dict]:
    return _csv_dataset("im2gps3k", "images", "im2gps3k_places365.csv", "IMG_ID", "LAT", "LON", split, limit)


def yfcc4k(split: str | None = None, limit: int | None = None) -> Iterator[dict]:
    return _csv_dataset("yfcc4k", "images", "yfcc4k_places365.csv", "IMG_ID", "LAT", "LON", split, limit)


DATASETS = {"holo_examples": lambda split=None, limit=None: holo_examples(), "im2gps3k": im2gps3k, "yfcc4k": yfcc4k}
