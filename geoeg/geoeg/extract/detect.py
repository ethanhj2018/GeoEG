"""Grounding DINO (transformers implementation): ground each clue's query phrase to a box."""
from __future__ import annotations

import torch
from PIL import Image


class Detector:
    def __init__(self, model_id: str, box_threshold: float = 0.3, text_threshold: float = 0.25, device: str = "cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        # fp16 halves the weights and activations (VRAM budget is shared with vLLM)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, dtype=torch.float16).to(device).eval()
        self.device = device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    @torch.no_grad()
    def ground(self, image: Image.Image, phrase: str) -> dict | None:
        """Best box for one phrase: {"bbox": [x0,y0,x1,y1], "score": float, "label": str} or None."""
        phrase = phrase.strip().lower().rstrip(".")
        if not phrase:
            return None
        inputs = self.processor(images=image, text=phrase + ".", return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        out = self.model(**inputs)
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]])[0]
        if len(res["scores"]) == 0:
            return None
        i = int(res["scores"].argmax())
        return {"bbox": [round(float(x), 1) for x in res["boxes"][i].tolist()],
                "score": round(float(res["scores"][i]), 3), "label": res["text_labels"][i]}
