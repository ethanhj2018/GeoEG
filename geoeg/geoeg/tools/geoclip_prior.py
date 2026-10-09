"""GeoCLIP: Top-K coordinate prior from its GPS gallery + cosine similarity for arbitrary candidates.

VRAM: the CLIP text tower is dropped and the vision tower runs in fp16; the 100k-point GPS gallery is encoded
in chunks instead of in one batch (GeoCLIP.predict does it in one go, which peaks at several GB).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GALLERY_CHUNK = 16384


class GeoCLIPPrior:
    def __init__(self, device: str = "cuda"):
        from geoclip import GeoCLIP
        from geoclip.model.image_encoder import _extract_image_features
        model = GeoCLIP()
        clip = model.image_encoder.CLIP
        for name in ("text_model", "text_projection"):
            if hasattr(clip, name):
                delattr(clip, name)
        clip.half()
        self.model = model.to(device).eval()
        self._extract = _extract_image_features
        self.device = device

    @torch.no_grad()
    def _image_features(self, image: Image.Image) -> torch.Tensor:
        enc = self.model.image_encoder
        px = enc.preprocess_image(image.convert("RGB")).to(self.device, dtype=torch.float16)
        x = self._extract(enc.CLIP.get_image_features(pixel_values=px)).float()
        return F.normalize(enc.mlp(x), dim=1)

    @torch.no_grad()
    def _location_features(self, gps: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.normalize(self.model.location_encoder(chunk), dim=1)
                          for chunk in gps.split(GALLERY_CHUNK)])

    @torch.no_grad()
    def _gallery_features(self) -> torch.Tensor:
        if getattr(self, "_gal_feats", None) is None:      # 100k × 512 fp32 ≈ 200 MB, computed once per load
            self._gal_feats = self._location_features(self.model.gps_gallery.to(self.device))
        return self._gal_feats

    @torch.no_grad()
    def gallery_scores(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(gallery lat/lon N×2, cosine similarity N, softmax probability N) over GeoCLIP's GPS gallery."""
        img = self._image_features(image)
        sims = (img @ self._gallery_features().T).squeeze(0)
        probs = (self.model.logit_scale.exp() * sims).softmax(dim=-1)
        return self.model.gps_gallery.cpu().numpy(), sims.float().cpu().numpy(), probs.float().cpu().numpy()

    @torch.no_grad()
    def top_k(self, image: Image.Image, k: int = 20) -> list[dict]:
        gallery, _, probs = self.gallery_scores(image)
        idx = np.argsort(-probs)[:k]
        return [{"lat": float(gallery[i, 0]), "lon": float(gallery[i, 1]), "prob": float(probs[i])} for i in idx]

    @torch.no_grad()
    def similarity(self, image: Image.Image, latlons: list[tuple[float, float]]) -> np.ndarray:
        if not latlons:
            return np.zeros(0)
        img = self._image_features(image)
        loc = self._location_features(torch.tensor(latlons, dtype=torch.float32, device=self.device))
        return (img @ loc.T).squeeze(0).float().cpu().numpy()
