"""SAM 2 box-prompted segmentation: isolate each grounded clue and save a masked crop."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class Segmenter:
    def __init__(self, model_id: str, device: str = "cuda"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    @torch.no_grad()
    def segment(self, image: Image.Image, boxes: list[list[float]]) -> list[dict]:
        """For each box: {"mask_area": fraction of image, "score": float, "mask": HxW bool array}."""
        if not boxes:
            return []
        arr = np.array(image.convert("RGB"))
        self.predictor.set_image(arr)
        masks, scores, _ = self.predictor.predict(box=np.array(boxes, dtype=np.float32), multimask_output=False)
        masks = masks.reshape(len(boxes), *arr.shape[:2]).astype(bool)
        scores = np.asarray(scores).reshape(len(boxes))
        total = arr.shape[0] * arr.shape[1]
        return [{"mask_area": round(float(m.sum()) / total, 4), "score": round(float(s), 3), "mask": m}
                for m, s in zip(masks, scores)]


def save_crop(image: Image.Image, bbox: list[float], mask: np.ndarray | None, path: str, pad: int = 8) -> str:
    """Crop the bbox; pixels outside the mask are dimmed so the isolated object stays readable in context."""
    arr = np.array(image.convert("RGB")).astype(np.float32)
    if mask is not None:
        arr[~mask] *= 0.35
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = bbox
    box = (max(0, int(x0) - pad), max(0, int(y0) - pad), min(w, int(x1) + pad), min(h, int(y1) + pad))
    Image.fromarray(arr.astype(np.uint8)).crop(box).save(path, quality=90)
    return path
