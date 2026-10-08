"""G3 (NeurIPS 2024, Apache-2.0) image–GPS similarity as a candidate scorer.

The model code is imported from the local clone (~/geoeg_refs/G3, utils/G3.py + utils/rff), not copied.
Only the vision tower runs, in fp16; the text tower and text heads are dropped after loading g3.pth.
The location encoder stays fp32 (random Fourier features need the precision).
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image


class G3Scorer:
    def __init__(self, repo: str, ckpt: str, device: str = "cuda"):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from utils.G3 import G3   # noqa: E402  (from the G3 clone)
        model = G3(device)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"g3.pth 与模型结构不符：missing={missing[:5]} unexpected={unexpected[:5]}")
        for name in ("text_model", "text_projection", "text_projection_else", "vision_projection_else_1"):
            delattr(model, name)
        model.vision_model.half()
        model.vision_projection.half()
        model.vision_projection_else_2.half()
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def similarity(self, image: Image.Image, latlons: list[tuple[float, float]]) -> np.ndarray:
        if not latlons:
            return np.zeros(0)
        m = self.model
        px = m.vision_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        px = px.to(self.device, dtype=torch.float16)
        img = m.vision_projection_else_2(m.vision_projection(m.vision_model(px)[1])).float()
        img = img / img.norm(p=2, dim=-1, keepdim=True)
        gps = torch.tensor(latlons, dtype=torch.float32, device=self.device)
        loc = m.location_projection_else(m.location_encoder(gps).reshape(len(latlons), -1))
        loc = loc / loc.norm(p=2, dim=-1, keepdim=True)
        return (img @ loc.T).squeeze(0).cpu().numpy()
