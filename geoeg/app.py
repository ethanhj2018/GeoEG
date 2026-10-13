"""Minimal Gradio UI: upload an image → Top-5 table + evidence-graph JSON. Listens on 127.0.0.1 only.

Run inside WSL (conda env geoeg):  python app.py   →  http://127.0.0.1:7861
(7860 is left for the web/ demo.)
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from geoeg.config import ConfigError, load_config
from geoeg.pipeline import run


def predict(image_path: str, mode: str):
    if not image_path:
        raise gr.Error("Please upload an image first")
    try:
        res = run(image_path, load_config(mode=mode))
    except ConfigError as e:
        raise gr.Error(str(e))
    rows = [[i, round(c["score"], 2), round(c["lat"], 4), round(c["lon"], 4),
             ", ".join(x for x in (c["city"], c["country"]) if x), ", ".join(c["supporting_clues"])]
            for i, c in enumerate(res["top5"], 1)]
    graph = json.loads((Path(res["run_dir"]) / "graph.json").read_text(encoding="utf-8"))
    info = "\n\n".join(f"**{i}.** {e}" for i, e in enumerate(res["explanations"], 1))
    info += f"\n\nTime {res['timings_s']['total']} s · peak card VRAM {res['gpu']['card_peak_mib']} MiB · `{res['run_dir']}`"
    return rows, info, graph


with gr.Blocks(title="GeoEG") as demo:
    gr.Markdown("## GeoEG: image geolocation with an evidence graph")
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type="filepath", label="Image")
            mode = gr.Radio(["local", "hybrid"], value="local", label="Mode")
            btn = gr.Button("Locate", variant="primary")
        with gr.Column(scale=2):
            table = gr.Dataframe(headers=["#", "Score", "Lat", "Lon", "Place", "Supporting clues"], label="Top-5")
            expl = gr.Markdown()
            graph = gr.JSON(label="Evidence graph")
    btn.click(predict, [img, mode], [table, expl, graph])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
